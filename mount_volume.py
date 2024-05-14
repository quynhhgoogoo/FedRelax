import subprocess
from kubernetes import client, config
import pickle
import numpy as np
import base64

# Load the graph object from the pickle file
G = pickle.load(open('/app/algorithm/QuizGraphLearning.pickle', 'rb'))

for iter_node in G.nodes(): 
    nodefeatures = np.array([np.mean(G.nodes[iter_node]["Xtrain"]), np.mean(G.nodes[iter_node]["ytrain"])])
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

# Mount volume into each pod
def mount_volume(pod_name, volume_name, volume_mount_path, data):
    mount_command = [
        "/bin/sh",
        "-c",
        f"kubectl exec -n fed-relax {pod_name} -- mkdir -p {volume_mount_path} && kubectl cp {data} fed-relax/{pod_name}:{volume_mount_path}"
    ]
    subprocess.run(mount_command)

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

        # Mount volumes with the necessary data into the pods
        mount_volume(pod_name, "Xtrain", "/app/data/Xtrain", pickle.dumps(G.nodes[iter_node]["Xtrain"]))
        mount_volume(pod_name, "ytrain", "/app/data/ytrain", pickle.dumps(G.nodes[iter_node]["ytrain"]))
        mount_volume(pod_name, "Xval", "/app/data/Xval", pickle.dumps(G.nodes[iter_node]["Xval"]))
        mount_volume(pod_name, "yval", "/app/data/yval", pickle.dumps(G.nodes[iter_node]["yval"]))

        print(f"Volumes mounted successfully into pod {pod_name}")

init_attributes()
