from turtle import pd
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
import base64

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
            "coords": base64.b64encode(pickle.dumps(np.array(node_features))).decode('utf-8'),
            "Xtrain": base64.b64encode(pickle.dumps(G.nodes[iter_node]["Xtrain"])).decode('utf-8'),
            "ytrain": base64.b64encode(pickle.dumps(G.nodes[iter_node]["ytrain"])).decode('utf-8'),
            "Xval": base64.b64encode(pickle.dumps(G.nodes[iter_node]["Xval"])).decode('utf-8'),
            "yval": base64.b64encode(pickle.dumps(G.nodes[iter_node]["yval"])).decode('utf-8'),
        }
        configmap_name = f"node-configmap-{iter_node}"

        # Create or update ConfigMap
        create_or_update_configmap(configmap_name, configmap_data)
        print(f"ConfigMap Data for {configmap_name}: {configmap_data}")
        print("Shape of ConfigMap Data, Xtrain", np.array(configmap_data["Xtrain"]).shape)
        print("Shape of ConfigMap Data, ytrain", np.array(configmap_data["ytrain"]).shape)

        # Update the Kubernetes pod with the new attributes
        update_pod_attributes(pod_name, configmap_name)
        print(f"ConfigMap {configmap_name} created/updated successfully.")

init_attributes()