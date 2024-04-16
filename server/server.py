from kubernetes import client, config
import socket
import pickle
import numpy as np
import base64

node_attributes = {}

def get_node_attributes(pod_selector="app=fedrelax-client", namespace="fed-relax"):
    config.load_incluster_config()
    v1 = client.CoreV1Api()
    attributes = {}
    pods = v1.list_namespaced_pod(namespace=namespace, label_selector=pod_selector)

    for pod in pods.items:
        # Get the ConfigMap name associated with the pod
        configmap_name = pod.metadata.labels.get("configmap-name")

        if not configmap_name:
            print(f"Warning: Pod {pod.metadata.name} doesn't have a configmap-name label.")
            continue

        # Get the ConfigMap data
        try:
            configmap = v1.read_namespaced_config_map(name=configmap_name, namespace=namespace)
        except client.rest.ApiException as e:
            print(f"Error retrieving ConfigMap {configmap_name} for pod {pod.metadata.name}: {e}")
            continue

        # Extract relevant attributes from the ConfigMap data
        node_attributes[pod.metadata.name] = {
            "coords": pickle.loads(base64.b64decode(configmap.data["coords"])),
        }

    return node_attributes

# Set up socket server
SRV = socket.gethostbyname(socket.gethostname())  # Get pod's IP address
PORT = 3000
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((SRV, PORT))
server_socket.listen()
print(f'Listening for connections on {SRV}:{PORT}...')

def receive_predictions(client_socket):
    try:
        message_header = client_socket.recv(4)
        if not len(message_header):
            return False
        message_length = int.from_bytes(message_header, byteorder='big')
        data = client_socket.recv(message_length)
        return data
    except:
        return False

# Initialize variables to store received predictions and node attributes
all_predictions = []
node_attributes = get_node_attributes()

while True:
    client_socket, _ = server_socket.accept()
    predictions_data = receive_predictions(client_socket)
    if predictions_data:
        # Basic check for base64 format (without external library)
        if len(predictions_data) % 4 == 0 and all(char in base64.b64alphabet or char in base64.urlsafe_b64alphabet for char in predictions_data.decode()):
            try:
                predictions = pickle.loads(base64.b64decode(predictions_data))
                all_predictions.append(predictions)
                print("Received predictions:", predictions)
            except pickle.UnpicklingError as e:
                print("Error while unpickling predictions data:", e)
        else:
            print("Received invalid data format (likely not base64 encoded).")
    else:
        client_socket.close()
