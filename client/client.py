import os
import socket
import pickle
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import sys
import base64
import random
import json
from kubernetes import client, config

def get_pod_name():
    # Get the pod name from the environment variable
    pod_name = os.environ.get('POD_NAME')

    if pod_name:
        print("Pod name:", pod_name)
    else:
        print("Pod name not found.")
    return pod_name

def get_configmap_data(pod_name, namespace="fed-relax"):
    config.load_incluster_config()
    v1 = client.CoreV1Api()

    # Get the ConfigMap name associated with the specified pod
    pod = v1.read_namespaced_pod(name=pod_name, namespace=namespace)
    configmap_name = pod.metadata.labels.get("configmap-name")

    if not configmap_name:
        print(f"Warning: Pod {pod_name} doesn't have a configmap-name label.")
        return None

    try:
        # Get the ConfigMap data
        configmap = v1.read_namespaced_config_map(name=configmap_name, namespace=namespace)
    except client.rest.ApiException as e:
        print(f"Error retrieving ConfigMap {configmap_name} for pod {pod_name}: {e}")
        return None
    
    # Extract relevant attributes from the ConfigMap data
    pod_attributes = {
        "coords": pickle.loads(base64.b64decode(configmap.data["coords"])),
        "Xtrain": pickle.loads(base64.b64decode(configmap.data["Xtrain"])),
        "ytrain": pickle.loads(base64.b64decode(configmap.data["ytrain"])),
        "Xval": pickle.loads(base64.b64decode(configmap.data["Xval"])),
        "yval": pickle.loads(base64.b64decode(configmap.data["yval"])),
    }

    if pod_attributes:
        print(f"Attributes for pod {pod_name}: {pod_attributes}")
    else:
        print(f"No attributes found for pod {pod_name}.")

    return pod_attributes

def train_local_model(Xtrain, ytrain, max_depth=4):
    model = DecisionTreeRegressor(max_depth=max_depth)
    model.fit(Xtrain, ytrain)
    return model

def send_predictions_to_server(predictions, peer_ip, port=3000):
    # Establish connection to the server using socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client_socket.connect((peer_ip, port))
    except Exception as e:
        print(f"Error connecting to server: {e}")
        return
    
    try:
        # Convert predictions data to JSON
        message = json.dumps(predictions.tolist())
        # Add message header: length of message as 2-byte big-endian integer
        message_header = len(message).to_bytes(2, byteorder='big')
        message_with_header = message_header + message.encode()

        client_socket.send(message_with_header)
        print("Predictions sent", predictions, type(predictions))

        # Receive acknowledgment from server
        ack = client_socket.recv(1024)
        print("Server acknowledgment:", ack.decode('utf-8'))
    except Exception as e:
        print(f"Error sending predictions to server: {e}")
    finally:
        # Close the socket connection
        client_socket.close()
def get_random_server_pod_ip(namespace="fed-relax"):
    config.load_incluster_config()
    v1 = client.CoreV1Api()

    # List the pods in the server deployment
    server_pods = v1.list_namespaced_pod(namespace=namespace, label_selector="app=fedrelax-server").items
    
    if not server_pods:
        print("No server pods found in the namespace.")
        return None

    # Choose a random pod from the list of server pods
    random_pod = random.choice(server_pods)

    # Retrieve the IP address of the chosen pod
    pod_ip = random_pod.status.pod_ip
    
    if pod_ip:
        print("Random server pod IP:", pod_ip)
    else:
        print("Failed to retrieve the IP address of a random server pod.")
    
    return pod_ip

# Load data and attributes from ConfigMap
pod_name = get_pod_name()
print(pod_name)
pod_hash = pod_name[:-4]
configmap_name = f"node-configmap-{pod_hash}"
configmap_data = get_configmap_data(pod_name)
print("Config Map data: ", configmap_data)

if configmap_data:
    Xtrain = configmap_data["Xtrain"]
    ytrain = configmap_data["ytrain"]

    # Train local model
    local_model = train_local_model(Xtrain, ytrain)

    # Generate predictions on a test set (replace with actual test set generation)
    Xtest = np.arange(0.0, 1, 0.1).reshape(-1, 1)
    predictions = local_model.predict(Xtest)

    # Get peer pod's IP address
    pod_ip = get_random_server_pod_ip()

    # Send predictions to the server for aggregation
    print("Predictions from model", predictions, type(predictions))
    send_predictions_to_server(predictions, pod_ip)
else:
    print("Error: ConfigMap data not found.")
