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

    # Update pod attributes to reference the ConfigMap
    new_labels = {"configmap-name": configmap_name}
    pod.metadata.labels.update(new_labels)

    # Apply the changes
    v1.replace_namespaced_pod(name=pod_name, namespace=namespace, body=pod)


def init_attributes():
    # Should modify and scale later
    num_pods = min(len(list(get_pod_info(label_selector="app=fedrelax-client"))), len(G.nodes()))
    pod_info = {}

    # Iterate through the nodes and update Kubernetes pods
    for iter_node in range(num_pods):
        i = iter_node
        node_features = np.array([np.mean(G.nodes[iter_node]["Xtrain"]), np.mean(G.nodes[iter_node]["ytrain"])])
        pod_name = list(get_pod_info(label_selector="app=fedrelax-client"))[i]
        pod_ip = list(get_pod_info(label_selector="app=fedrelax-client").values())[i]

        # Construct the new attributes based on 'node_features'
        configmap_data = {
            "coords": node_features,
            "Xtrain": G.nodes[iter_node]["Xtrain"],
            "ytrain": G.nodes[iter_node]["ytrain"],
        }

        configmap_name = f"node-configmap-{iter_node}"

        # Create or update ConfigMap
        create_or_update_configmap(configmap_name, configmap_data)
        print(f"ConfigMap Data for {configmap_name}: {configmap_data}")
        print("Shape of ConfigMap Data, Xtrain", np.array(configmap_data["Xtrain"]).shape)
        print("Shape of ConfigMap Data, ytrain", np.array(configmap_data["ytrain"]).shape)

        # Update the Kubernetes pod with the new attributes
        update_pod_attributes(pod_name, configmap_name)
        print(f"ConfigMap {configmap_name} created/updated successfully.")\

        # Store pod attributes
        pod_info[pod_name] = {"coords": node_features, "Xtrain": np.array(configmap_data["Xtrain"]), "ytrain": np.array(configmap_data["Xtrain"])}
    
    return pod_info


def add_edges_k8s(pod_info,namespace="fed-relax", nrneighbors=1, pos='coords', refdistance=1):
    """
    Add edges to the graph based on pod attributes retrieved from Kubernetes config maps
    using k-nearest neighbors approach.
    """
    # Get pod information from Kubernetes cluster
    # pod_info = get_pod_attributes(namespace=namespace)
    
    # Initialize empty graph
    graph = nx.Graph()
    
    # Add nodes to the graph with pod attributes
    for pod_name, attributes in pod_info.items():
        graph.add_node(pod_name, **attributes)
    
    # Build a numpy array containing node positions
    node_positions = np.array([attributes["coords"] for attributes in pod_info.values()], dtype=float)
    
    # Calculate k-nearest neighbors graph
    A = kneighbors_graph(node_positions, n_neighbors=nrneighbors, mode='connectivity', include_self=False)
    
    # Iterate over the k-nearest neighbors graph and add edges with weights
    for i in range(len(pod_info)):
        for j in range(len(pod_info)):
            if A[i, j] > 0:
                pod_name_i = list(pod_info.keys())[i]
                pod_name_j = list(pod_info.keys())[j]
                
                # Calculate the Euclidean distance between pods based on their positions
                distance = np.linalg.norm(node_positions[i] - node_positions[j])
                
                # Add edge with weight based on distance
                graph.add_edge(pod_name_i, pod_name_j, weight=distance)
    
    return graph

# Function to visualize the graph and save the image
def visualize_and_save_graph(graph, output_path):
    pos = nx.spring_layout(graph)  # Compute layout for visualization
    plt.figure(figsize=(10, 10))
    nx.draw(graph, pos, with_labels=True, node_size=3000, node_color='skyblue', font_size=10, font_weight='bold')
    plt.title("Graph Visualization")
    plt.savefig(output_path)  # Save the image to a file
    print(f"Image is successfully saved in {output_path}")
    plt.show()  # Display the graph

# Fed Relax's main function
def FedRelax(Xtest, updated_graph, namespace="fed-relax", label_selector="app=fedrelax-client", regparam=0, maxiter=100):
    # Determine the number of data points in the test set
    testsize = Xtest.shape[0]
    
    # Attach a DecisionTreeRegressor as the local model to each node in G
    for node_i in G.nodes(data=False): 
        G.nodes[node_i]["model"] = DecisionTreeRegressor(max_depth=4).fit(np.array(G.nodes[node_i]["Xtrain"]).reshape(-1,1), np.array(G.nodes[node_i]["ytrain"]))
        G.nodes[node_i]["sample_weight"] = np.ones((len(G.nodes[node_i]["ytrain"]), 1))  # Initialize sample weights
    
    # Repeat the local updates (simultaneously at all nodes) for maxiter iterations
    for iter_GD in range(maxiter):
        # Iterate over all nodes in the graph
        for node_i in G.nodes(data=False):
            # Share predictions with neighbors
            for node_j in G[node_i]:
                # Add the predictions of the current hypothesis at node j as labels
                neighbourpred = G.nodes[node_j]["model"].predict(Xtest).reshape(-1, 1)
                neighbourpred = np.tile(neighbourpred, (1, len(G.nodes[node_i]["ytrain"][0])))
                G.nodes[node_i]["ytrain"] = np.vstack((G.nodes[node_i]["ytrain"], neighbourpred))
                
                # Augment local dataset at node i by a new dataset obtained from the features of the test set
                G.nodes[node_i]["Xtrain"] = np.vstack((G.nodes[node_i]["Xtrain"], Xtest))
                
                # Set sample weights of added local dataset according to edge weight of edge i <-> j
                # and GTV regularization parameter
                sampleweightaug = (regparam * len(G.nodes[node_i]["ytrain"]) / testsize)
                G.nodes[node_i]["sample_weight"] = np.vstack((G.nodes[node_i]["sample_weight"], sampleweightaug * G.edges[(node_i, node_j)]["weight"] * np.ones((len(neighbourpred), 1))))
            
            # Fit the local model with the augmented dataset and sample weights
            G.nodes[node_i]["model"].fit(G.nodes[node_i]["Xtrain"], G.nodes[node_i]["ytrain"], sample_weight=G.nodes[node_i]["sample_weight"].reshape(-1))
    
    return G


def PlotFinalGraph(graphin, pos='coord', annotate="name", output_path='/app/final.png'):
    # the numpy array x will hold the horizontal coord of markers for each node in emp. graph graphin
    x = np.zeros(len(graphin.nodes))
    # vertical coords of markers
    y = np.zeros(len(graphin.nodes))

    for iter_node in graphin.nodes:
        x[iter_node] = graphin.nodes[iter_node][pos][0]
        y[iter_node] = graphin.nodes[iter_node][pos][1]

    # standardize the coordinates of node markers
    x = (x - np.min(x, axis=0)) / np.std(x, axis=0) + 1
    y = (y - np.min(y, axis=0)) / np.std(y, axis=0) + 1

    # create a figure with prescribed dimensions
    fig, ax = plt.subplots(figsize=(10, 10))

    # generate a scatter plot with each marker representing a node in graphin
    ax.scatter(x, y, 300, marker='o', color='Black')

    # draw links between nodes if they are connected by an edge
    for edge in graphin.edges:
        node1 = edge[0]
        node2 = edge[1]
        if graphin.has_edge(node1, node2):  # Check if the edge exists
            ax.plot([x[node1], x[node2]], [y[node1], y[node2]], c='black', lw=2)

    # annotate each marker by the node attribute whose name is stored in the input parameter "annotate"
    for iter_node in graphin.nodes:
        ax.annotate(str(graphin.nodes[iter_node][annotate]), (x[iter_node] + 0.2, 0.995 * y[iter_node]), c="red")
    ax.set_ylim(0.9 * np.min(y), 1.1 * np.max(y))
    ax.set_xlim(0.9 * np.min(x), 1.1 * np.max(x))

    plt.savefig(output_path)  # Save the image to a file
    print(f"Final graph is successfully saved in {output_path}")

# Initialize pod attributes
pod_info = init_attributes()

# Filter the graph to include only the first three nodes
subgraph_nodes = list(G.nodes())[:3]
subgraph = G.subgraph(subgraph_nodes).copy()

# Call the visualization function before adding edges
visualize_and_save_graph(subgraph, '/app/init_graph.png')

# Add edges based on the pod coordinates
updated_graph = add_edges_k8s(pod_info)
print("Update graph", updated_graph.nodes())
i = 'client-f44c98887-jw7wx'
#for iter_node in G.nodes(): 
#    print(G.nodes[iter_node])
print("Attributes of graphs after update")
print((np.array(updated_graph.nodes[i]["Xtrain"])).shape)
print((np.array(updated_graph.nodes[i]["ytrain"])).shape)
print((np.array(updated_graph.nodes[i]["coords"])).shape)

# Call the visualization function after adding edges
visualize_and_save_graph(updated_graph, '/app/after_graph.png')

# Generate global test_set
X_test = np.arange(0.0, 1, 0.1).reshape(-1, 1) 

# Get the start time
st = time.time()

# Run FedRelax on Kubernetes
final_graph = FedRelax(X_test, updated_graph)

# Plot the graph
visualize_and_save_graph(updated_graph, '/app/final_graph.png')
PlotFinalGraph(final_graph,pos='coords',annotate='name')

end = time.time()
print("runtime of FedRelax ", end - st)

# Compute node-wise train and val errors
for iter_node in final_graph.nodes():
    if "model" in final_graph.nodes[iter_node]:  # Check if 'model' exists in the node attributes
        trained_local_model = final_graph.nodes[iter_node]["model"]
        train_features = final_graph.nodes[iter_node]["Xtrain"]
        train_labels = final_graph.nodes[iter_node]["ytrain"]
        final_graph.nodes[iter_node]["trainerr"] = mean_squared_error(train_labels, trained_local_model.predict(train_features))
        # Assuming you have validation data for each node
        val_features = final_graph.nodes[iter_node]["Xval"]
        val_labels = final_graph.nodes[iter_node]["yval"]
        final_graph.nodes[iter_node]["valerr"] = mean_squared_error(val_labels, trained_local_model.predict(val_features))
    else:
        print(f"Warning: 'model' attribute not found for node {iter_node}")