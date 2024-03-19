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


def get_pod_attributes(namespace="fed-relax", label_selector="app=fedrelax-client"):
    """
    Retrieve pod attributes from Kubernetes config maps.
    """
    # Load Kubernetes configuration
    config.load_incluster_config()
    
    # Create Kubernetes API client
    api_instance = client.CoreV1Api()
    
    # Retrieve list of pods matching the label selector
    pods = api_instance.list_namespaced_pod(namespace=namespace, label_selector=label_selector)
    
    pod_info = {}
    
    # Iterate over pods and retrieve attributes from config maps
    for pod in pods.items:
        # Get pod name and namespace
        pod_name = pod.metadata.name
        
        # Retrieve config map associated with the pod
        config_map_name = pod.metadata.labels.get("configmap-name")
        if config_map_name:
            # Retrieve config map data
            config_map = api_instance.read_namespaced_config_map(name=config_map_name, namespace=namespace)
            
            # Extract relevant attributes from the config map data
            coords_str = config_map.data.get("coords")
            # Remove '[' and ']' characters and then split by ','
            coords_str = coords_str.strip('[]')
            coords = [float(coord) for coord in coords_str.split(',')]
            
            # Store pod attributes
            pod_info[pod_name] = {"coords": coords}
    
    return pod_info


def add_edges_k8s(namespace="fed-relax"):
    """
    Add edges to the graph based on pod attributes retrieved from Kubernetes config maps.
    """
    # Get pod information from Kubernetes cluster
    pod_info = get_pod_attributes(namespace=namespace)
    
    # Initialize empty graph
    graph = nx.Graph()
    
    # Add nodes to the graph with pod attributes
    for pod_name, attributes in pod_info.items():
        graph.add_node(pod_name, **attributes)
    
    # Calculate distances and add edges based on pod attributes
    for pod_name1, attr1 in pod_info.items():
        for pod_name2, attr2 in pod_info.items():
            if pod_name1 != pod_name2:
                # Calculate distance between pods based on attributes (e.g., coordinates)
                distance = np.linalg.norm(np.array(attr1["coords"]) - np.array(attr2["coords"]))
                
                # Add edge with weight based on distance
                graph.add_edge(pod_name1, pod_name2, weight=distance)
    
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

# Initialize pod attributes
init_attributes()

# Filter the graph to include only the first four nodes
subgraph_nodes = list(G.nodes())[:4]
subgraph = G.subgraph(subgraph_nodes).copy()

# Call the visualization function before adding edges
visualize_and_save_graph(subgraph, '/app/init_graph.png')

# Add edges based on the pod coordinates
updated_graph = add_edges_k8s()

# Call the visualization function after adding edges
visualize_and_save_graph(updated_graph, '/app/after_graph.png')