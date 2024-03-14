from kubernetes import client, config
import pickle
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import networkx as nx
from sklearn.neighbors import kneighbors_graph
from numpy import linalg as LA
import time
from sklearn.metrics import mean_squared_error

# Load the graph object from the pickle file
G = pickle.load(open('/app/algorithm/QuizGraphLearning.pickle', 'rb'))

for iter_node in G.nodes(): 
    nodefeatures = np.array([np.mean(G.nodes[iter_node]["Xtrain"]),np.mean(G.nodes[iter_node]["ytrain"])])
    G.nodes[iter_node]['coords'] = nodefeatures 
    G.nodes[iter_node]["name"] = str(iter_node)

# Load Kubernetes configuration
config.load_incluster_config()

# Get pod's name and IP addresses {pod_name:IP}
def get_pod_info(namespace="fed-relax"):
    config.load_incluster_config()
    v1 = client.CoreV1Api()

    pod_info = {}
    pods = v1.list_namespaced_pod(namespace=namespace, label_selector="app=fedrelax-client")

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
    num_pods = min(len(list(get_pod_info())), len(G.nodes()))

    # Iterate through the nodes and update Kubernetes pods
    for iter_node in range(num_pods):
        i = iter_node
        node_features = np.array([np.mean(G.nodes[iter_node]["Xtrain"]), np.mean(G.nodes[iter_node]["ytrain"])])
        
        pod_name = list(get_pod_info())[i]
        pod_ip = list(get_pod_info().values())[i]

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
    pod_info = get_pod_info(namespace)

    # Build up a numpy array tmp_data which has one row for each pod in graphinloa
    tmp_data = np.zeros((len(graphin.nodes()), len(next(iter(graphin.nodes().values())))))
    
    # Iterate over the nodes and their attributes
    for iter_node in graphin.nodes(data=True):
        node_index, node_attr = iter_node
        # Check if the 'coords' attribute exists for the node
        if pos in node_attr:
            # Each row of tmp_data holds the numpy array stored in the node attribute selected by parameter "pos"
            tmp_data[node_index, :] = node_attr[pos]
        else:
            # Handle the case where the attribute is missing for a node
            print(f"Attribute '{pos}' not found for node {node_index}")

    # Create a connectivity matrix using k-neighbors
    A = kneighbors_graph(tmp_data, nrneighbors, mode='connectivity', include_self=False)
    A = A.toarray()

    for iter_i in range(len(graphin.nodes())):
        for iter_j in range(len(graphin.nodes())):
            # Add an edge between nodes i,j if entry A_i,j is non-zero
            if A[iter_i, iter_j] > 0:
                graphin.add_edge(iter_i, iter_j)
                # Use the Euclidean distance between node attribute selected by parameter "pos"
                # to compute the edge weight
                graphin.edges[(iter_i, iter_j)]["weight"] = np.exp(
                    -LA.norm(tmp_data[iter_i, :] - tmp_data[iter_j, :], 2) / refdistance)


# Initialize pod attributes
init_attributes()

# Add edges based on the pod coordinates
add_edges_k8s(G, nrneighbors=2, pos='coords', refdistance=100)