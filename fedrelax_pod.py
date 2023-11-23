from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import os
from kubernetes import client, config
import socket
import random
import time
import threading
import joblib

# Get all pod's ip in namespace
def get_pod_ip_addresses():
    config.load_kube_config()  # Load your Kubernetes configuration (e.g., ~/.kube/config)

    v1 = client.CoreV1Api()

    pod_ip_addresses = {}
    pods = v1.list_namespaced_pod(namespace="fed-relax")

    for pod in pods.items:
        pod_ip_addresses[pod.metadata.name] = pod.status.pod_ip

    return pod_ip_addresses

# Send parameters to pod's namespace
peer_ips = get_pod_ip_addresses() 
peer_port = 8000

def send_to_pod(pod_index, pod_ip, model_params):
    # Allow pods to send parameters to other pods in the same ns
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((pod_ip, peer_port))
    client_socket.send(str(model_params).encode())
    print(f"Send to pod {pod_index} received: {str(model_params).encode()}")


def receive_from_pod(pod_index, pod_ip):
    # Allow pods to receive parameters from other pods in same ns
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((pod_ip, peer_port))
    data = client_socket.recv(1024)
    print(f"Pod {pod_index} received: {data.decode()}")

    return data

# Load train data to the pod
df = pd.read_csv("data/train.csv")
# Only load a part of data to current pod 
df = df.iloc[:1000,:]
X1 = df['tweet'].to_numpy()
y1 = df['label'].to_numpy().reshape(-1,)
local_model = LogisticRegression()
local_model.fit(X1, y1)


# Apply FedRelax algorithm
# Initialize global model weights as the local model's weights
global_model_weights = local_model.coef_

# Number of pods
num_pods = 2

# Number of iterations for FedRelax
num_iterations = 10

for iteration in range(num_iterations):
    # Share and receive global model weights from other pods
    for i, pod_id in enumerate(peer_ips):
        # Send your global model weights to another pods
        send_to_pod(i, pod_id, global_model_weights)
        # Receive the global model weights from another pod
        received_weights = receive_from_pod(i, pod_id)
        # Update your global model weights based on received_weights
        global_model_weights = (global_model_weights + received_weights) / 2  # You can adjust this aggregation method

    # Update your local model with the global weights
    local_model.coef_ = global_model_weights
    local_model.fit(X1, y1)

# Final global model is in global_model_weights

# Evaluate model
test_df = pd.read_csv("test.csv")
X_test = test_df['tweet'].to_numpy()
y_test = test_df['label'].to_numpy().reshape(-1,)

# Initialize a list to store local evaluation results
local_evaluation_results = []

# Evaluate local models
for i in range(num_pods):
    # Assuming 'local_model' is your trained model in each pod
    y_pred = local_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Print or store the local evaluation result
    print(f"Local Accuracy for Pod {i}: {accuracy}")
    local_evaluation_results.append(accuracy)