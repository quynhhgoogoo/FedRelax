from kubernetes import client, config
import pickle
import networkx as nx
import numpy as np

# Load the graph object from the pickle file
G = pickle.load(open('/app/algorithm/QuizGraphLearning.pickle', 'rb'))

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

