from kubernetes import client, config
import pickle
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import networkx as nx
from sklearn.neighbors import kneighbors_graph
from numpy import linalg as LA
import time
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the graph object from the pickle file
G = pickle.load(open('/app/algorithm/QuizGraphLearning.pickle', 'rb'))

for iter_node in G.nodes(): 
    nodefeatures = np.array([np.mean(G.nodes[iter_node]["Xtrain"]),np.mean(G.nodes[iter_node]["ytrain"])])
    G.nodes[iter_node]['coords'] = nodefeatures 
    G.nodes[iter_node]["name"] = str(iter_node)

# Load Kubernetes configuration
config.load_incluster_config()

# Get pod's name and IP addresses {pod_name:IP}
def get_pod_info(label_selector, namespace="fed-relax"):
    config.load_incluster_config()
    v1 = client.CoreV1Api()

    pod_info = {}
    pods = v1.list_namespaced_pod(namespace=namespace, label_selector=label_selector)

    for pod in pods.items:
        pod_info[pod.metadata.name] = pod.status.pod_ip

    return pod_info


def create_or_update_configmap(configmap_name, configmap_data, namespace="fed-relax"):
    v1 = client.CoreV1Api()
    
    # Check if the ConfigMap already exists
    try:
        existing_configmap = v1.read_namespaced_config_map(name=configmap_name, namespace=namespace)
        existing_configmap.data = {key: str(value) for key, value in configmap_data.items()}
        v1.replace_namespaced_config_map(name=configmap_name, namespace=namespace, body=existing_configmap)
    except client.rest.ApiException as e:
        if e.status == 404:
            # ConfigMap doesn't exist, create a new one
            if not configmap_name:
                print("Error: ConfigMap name is empty")
                return
            configmap_body = client.V1ConfigMap(metadata=client.V1ObjectMeta(name=configmap_name), 
                                                data={key: str(value) for key, value in configmap_data.items()})
            v1.create_namespaced_config_map(namespace=namespace, body=configmap_body)
        else:
            print(f"Error during ConfigMap operation: {e}")



def update_pod_attributes(pod_name, configmap_name, namespace="fed-relax"):
    v1 = client.CoreV1Api()
    pod = v1.read_namespaced_pod(name=pod_name, namespace=namespace)

    # Debug: Print the contents of the Pod
    # print("Pod object before update:")
    # print(pod)

    # Update pod attributes to reference the ConfigMap
    new_labels = {"configmap-name": configmap_name}
    pod.metadata.labels.update(new_labels)

    # Apply the changes
    try:
        v1.replace_namespaced_pod(name=pod_name, namespace=namespace, body=pod)
        # Debug: Print the contents of the Pod
        # print("Pod object after update:")
        # print(pod)
    except Exception as e:
        print(f"Error updating Pod attributes: {e}")


def init_attributes():
# Should modify and scale later
    num_pods = min(len(list(get_pod_info(label_selector="app=fedrelax-client"))), len(G.nodes()))

    # Iterate through the nodes and update Kubernetes pods
    for iter_node in range(num_pods):
        i = iter_node
        node_features = np.array([np.mean(G.nodes[iter_node]["Xtrain"]), np.mean(G.nodes[iter_node]["ytrain"])])
        
        pod_name = list(get_pod_info(label_selector="app=fedrelax-client"))[i]
        pod_ip = list(get_pod_info(label_selector="app=fedrelax-client").values())[i]

        # Construct the new attributes based on 'node_features'
        configmap_data = {
            "coords": node_features.tolist(),
            "Xtrain": G.nodes[iter_node]["Xtrain"].tolist(),
            "ytrain": G.nodes[iter_node]["ytrain"].tolist(),
        }

        configmap_name = f"node-configmap-{iter_node}"

        # Create or update ConfigMap
        create_or_update_configmap(configmap_name, configmap_data)
        print(f"ConfigMap Data for {configmap_name}: {configmap_data}")

        # Update the Kubernetes pod with the new attributes
        update_pod_attributes(pod_name, configmap_name)
        print(f"ConfigMap {configmap_name} created/updated successfully.")


def add_edges_k8s(graphin, nrneighbors=3, pos='coords', refdistance=1, namespace="fed-relax"):
    edges = graphin.edges()
    graphin.remove_edges_from(edges)

    # Get pod information from the Kubernetes cluster
    pod_info = get_pod_info(namespace=namespace, label_selector="app=fedrelax-client")

    # Build up a numpy array tmp_data which has one row for each pod in graphin
    num_nodes = len(graphin.nodes())
    node_shape = len(next(iter(graphin.nodes(data=True)))[1].get(pos, []))
    tmp_data = np.zeros((num_nodes, node_shape))

    # Iterate over the nodes and their attributes
    for node_index, node_attr in graphin.nodes(data=True):
        # Check if the 'coords' attribute exists for the node and has the correct shape
        if pos in node_attr and len(node_attr[pos]) == node_shape:
            tmp_data[node_index, :] = node_attr[pos]
        else:
            # Handle the case where the attribute is missing or has an incorrect shape for a node
            print(f"Attribute '{pos}' not found or has incorrect shape for node {node_index}")

    # Create a connectivity matrix using k-neighbors
    A = kneighbors_graph(tmp_data, nrneighbors, mode='connectivity', include_self=False)
    A = A.toarray()

    for iter_i in range(num_nodes):
        for iter_j in range(num_nodes):
            # Add an edge between nodes i,j if entry A_i,j is non-zero
            if A[iter_i, iter_j] > 0:
                graphin.add_edge(iter_i, iter_j)
                # Use the Euclidean distance between node attribute selected by parameter "pos" to compute the edge weight
                graphin.edges[(iter_i, iter_j)]["weight"] = np.exp(
                    -LA.norm(tmp_data[iter_i, :] - tmp_data[iter_j, :], 2) / refdistance)


def visualize_and_save_graph(graph, output_path):
    pos = nx.spring_layout(graph)  # Compute layout for visualization
    plt.figure(figsize=(10, 10))
    nx.draw(graph, pos, with_labels=True, node_size=3000, node_color='skyblue', font_size=10, font_weight='bold')
    plt.title("Graph Visualization")
    plt.savefig(output_path) 
    print(f"Image is successfully saved in {output_path}")
    plt.show() 

# FedRelax main algorithm
def fed_relax_k8s(graph, X_test, regparam=0, maxiter=100, namespace="fed-relax"):
    # Determine the number of data points in the test set
    testsize = X_test.shape[0]

    # Attach a DecisionTreeRegressor as the local model to each node in G
    pod_names = list(get_pod_info(namespace="fed-relax", label_selector="app=fedrelax-client"))
    num_nodes = len(graph.nodes())

    if not pod_names:
        print("Error: No pod names available.")
        return

    for node_i in range(min(num_nodes, len(pod_names))):
        pod_name = pod_names[node_i]
        print(f"Selected pod name for node {node_i}: {pod_name}")  # Debug: Print selected pod name

        # Construct the new attributes based on 'node_features'
        new_pod_attributes = {
            "Xtrain": graph.nodes[node_i]["Xtrain"].tolist(),
            "ytrain": graph.nodes[node_i]["ytrain"].tolist(),
            "model": None,  # Initialize the model attribute
            "sample_weight": np.ones((len(graph.nodes[node_i]["ytrain"]), 1)).tolist(),
        }

        # Update the Kubernetes pod with the new attributes
        update_pod_attributes(pod_name, new_pod_attributes, namespace)

    # Repeat the local updates (simultaneously at all nodes) for maxiter iterations
    for iter_GD in range(maxiter):
        # Iterate over all nodes in the graph
        for node_i in graph.nodes(data=False):
            if node_i >= len(pod_names):
                print(f"Error: Node index {node_i} is out of range.")
                continue

            pod_name_i = pod_names[node_i]

            # Share predictions with neighbors
            for node_j in graph[node_i]:
                if node_j >= len(pod_names):
                    print(f"Error: Node index {node_j} is out of range.")
                    continue

                pod_name_j = pod_names[node_j]

                # Add the predictions of the current hypothesis at node j as labels
                neighbourpred = graph.nodes[node_j]["model"].predict(X_test).reshape(-1, 1)

                # Update the Xtrain, ytrain, and sample_weight attributes of node_i
                new_attributes_i = {
                    "Xtrain": np.vstack((graph.nodes[node_i]["Xtrain"], X_test)).tolist(),
                    "ytrain": np.vstack((graph.nodes[node_i]["ytrain"], neighbourpred)).tolist(),
                    "sample_weight": np.vstack((
                        graph.nodes[node_i]["sample_weight"],
                        (regparam * len(graph.nodes[node_i]["ytrain"]) / testsize) * graph.edges[
                            (node_i, node_j)]["weight"] * np.ones((len(neighbourpred), 1))
                    )).tolist(),
                }

                # Update the attributes of node_i in the Kubernetes pod
                update_pod_attributes(pod_name_i, new_attributes_i, namespace)

                # Augment local dataset at node i by a new dataset obtained from the features of the test set
                graph.nodes[node_i]["Xtrain"] = new_attributes_i["Xtrain"]
                graph.nodes[node_i]["ytrain"] = new_attributes_i["ytrain"]
                graph.nodes[node_i]["sample_weight"] = new_attributes_i["sample_weight"]

            # Fit the local model with the augmented dataset and sample weights
            graph.nodes[node_i]["model"].fit(
                np.array(graph.nodes[node_i]["Xtrain"]),
                np.array(graph.nodes[node_i]["ytrain"]),
                sample_weight=np.array(graph.nodes[node_i]["sample_weight"]).reshape(-1)
            )


# Initialize pod attributes
init_attributes()

# Filter the graph to include only the first four nodes
subgraph_nodes = list(G.nodes())[:3]
subgraph = G.subgraph(subgraph_nodes).copy()

# Call the visualization function before adding edges
visualize_and_save_graph(subgraph, '/app/init_graph.png')

# Add edges based on the pod coordinates
add_edges_k8s(subgraph, nrneighbors=2, pos='coords', refdistance=100)

# Call the visualization function after adding edges
visualize_and_save_graph(subgraph, '/app/after_graph.png')

# Generate global test_set
X_test = np.arange(0.0, 1, 0.1)[:, np.newaxis]

# Get the start time
st = time.time()

# Run FedRelax on Kubernetes
fed_relax_k8s(subgraph, X_test, 0.1, 100, namespace="fed-relax")

end = time.time()
print("runtime of FedRelax ", end - st)

# Compute node-wise train and val errors
for iter_node in subgraph.nodes():
    trained_local_model = subgraph.nodes[iter_node]["model"]
    train_features = subgraph.nodes[iter_node]["Xtrain"]
    train_labels = subgraph.nodes[iter_node]["ytrain"]
    subgraph.nodes[iter_node]["trainerr"] = mean_squared_error(train_labels, trained_local_model.predict(train_features))
    # Assuming you have validation data for each node
    val_features = subgraph.nodes[iter_node]["Xval"]
    val_labels = subgraph.nodes[iter_node]["yval"]
    subgraph.nodes[iter_node]["valerr"] = mean_squared_error(val_labels, trained_local_model.predict(val_features))
