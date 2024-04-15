import os
import socket
import pickle
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import sys
import base64
from kubernetes import client, config

def get_pod_name():
    # Get the pod name from the environment variable
    pod_name = os.environ.get('POD_NAME')

    if pod_name:
        print("Pod name:", pod_name)
    else:
        print("Pod name not found.")
    return pod_name

def get_configmap_data(pod_name, configmap_name, namespace="fed-relax"):
    v1 = client.CoreV1Api()
    try:
        configmap = v1.read_namespaced_config_map(name=configmap_name, namespace=namespace)
    except client.rest.ApiException as e:
        print(f"Error retrieving ConfigMap {configmap_name} for pod {pod_name}: {e}")
        return None
    return {key: pickle.loads(base64.b64decode(value)) for key, value in configmap.data.items()}

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
        # Send predictions data
        message = pickle.dumps(predictions)
        client_socket.send(message)

        # Receive acknowledgment from server
        ack = client_socket.recv(1024)
        print("Server acknowledgment:", ack.decode('utf-8'))
    except Exception as e:
        print(f"Error sending predictions to server: {e}")
    finally:
        # Close the socket connection
        client_socket.close()

# Load data and attributes from ConfigMap
pod_name = get_pod_name()
print(pod_name)
pod_hash = pod_name[:-4]
configmap_name = f"node-configmap-{pod_hash}"
configmap_data = get_configmap_data(pod_name, configmap_name)

if configmap_data:
    Xtrain = configmap_data["Xtrain"]
    ytrain = configmap_data["ytrain"]

    # Train local model
    local_model = train_local_model(Xtrain, ytrain)

    # Generate predictions on a test set (replace with actual test set generation)
    Xtest = np.arange(0.0, 1, 0.1).reshape(-1, 1)
    predictions = local_model.predict(Xtest)

    # Get peer pod's IP address
    pod_ip = socket.gethostbyname(socket.gethostname())

    # Send predictions to the server for aggregation
    send_predictions_to_server(predictions, pod_ip)
else:
    print("Error: ConfigMap data not found.")
