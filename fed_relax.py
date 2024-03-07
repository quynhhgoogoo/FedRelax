from kubernetes import client, config
import pickle
import numpy as np

# Load the graph object from the pickle file
G = pickle.load(open('/algorithm/QuizGraphLearning.pickle', 'rb'))

# Load Kubernetes configuration
config.load_kube_config()

# Get pod's name and IP addresses {pod_name:IP}
def get_pod_info(namespace="fed-relax"):
    config.load_incluster_config()
    v1 = client.CoreV1Api()

    pod_info = {}
    pods = v1.list_namespaced_pod(namespace=namespace)

    for pod in pods.items:
        pod_info[pod.metadata.name] = pod.status.pod_ip

    return pod_info


def update_pod_attributes(pod_name, new_attributes):
    v1 = client.CoreV1Api()
    pod = v1.read_namespaced_pod(name=pod_name, namespace="fed-relax")
    # Update pod attributes
    pod.metadata.labels.update(new_attributes)
    # Apply the changes
    v1.replace_namespaced_pod(name=pod_name, namespace="fed-relax", body=pod)


# Iterate through the nodes and update Kubernetes pods
for i, iter_node in enumerate(G.nodes()):
    node_features = np.array([np.mean(G.nodes[iter_node]["Xtrain"]), np.mean(G.nodes[iter_node]["ytrain"])])
    
    pod_name = list(get_pod_info)[i]
    pod_ip = list(get_pod_info.values())[i]

    # Construct the new attributes based on 'node_features'
    new_pod_attributes = {
        "coords": node_features.tolist(),
        "name": pod_ip,
        "Xtrain": G.nodes[iter_node]["Xtrain"].tolist(),
        "ytrain": G.nodes[iter_node]["ytrain"].tolist(),
    }

    # Update the Kubernetes pod with the new attributes
    update_pod_attributes(pod_name, new_pod_attributes)
