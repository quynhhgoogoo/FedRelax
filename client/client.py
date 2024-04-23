import os
import socket
import pickle
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import sys
import base64
import random
import json
import time
import subprocess
from kubernetes import client, config


def check_job_completion(job_name="init-attributes-job", namespace="fed-relax"):
    config.load_incluster_config()  # Load incluster config if running inside a pod
    api_instance = client.BatchV1Api()
    try:
        job = api_instance.read_namespaced_job(job_name, namespace)
        if job.status.succeeded is not None and job.status.succeeded > 0:
            return True
    except Exception as e:
        print(f"Error checking job status: {str(e)}")
    return False


def wait_for_job_completion(job_name="init-attributes-job", namespace="fed-relax"):
    while not check_job_completion(job_name, namespace):
        print(f"Waiting for Job {job_name} to complete...")
        time.sleep(10)
    print(f"Job {job_name} completed successfully!")


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

    # Extract relevant attributes and training data
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


def send_model_update_to_server(coords, model_params, Xtrain, ytrain, sample_weight, peer_ip, port=3000):
    # Establish connection to the server using socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client_socket.connect((peer_ip, port))
    except Exception as e:
        print(f"Error connecting to server: {e}")
        return

    try:
        model_params_encoded = pickle.dumps(model_params)
        model_params_encoded_str = base64.b64encode(model_params_encoded).decode('utf-8')
        coords_encoded_str = base64.b64encode(pickle.dumps(coords)).decode('utf-8')

        if isinstance(sample_weight, np.ndarray):
            # Convert sample_weight to a list
            sample_weight_list = sample_weight.tolist()
        else:
            # No conversion needed if it's already a list
            sample_weight_list = sample_weight

        # Create a dictionary containing model parameters, training data, and sample weights
        client_update = {
            "pod_name": get_pod_name(),
            "coords": coords_encoded_str,
            "model_params": model_params_encoded_str,
            "sample_weight": sample_weight_list,  # Use the converted list
        }

        # Add message header: length of message as 2-byte big-endian integer
        message_header = len(json.dumps(client_update)).to_bytes(2, byteorder='big')
        message_with_header = message_header + json.dumps(client_update).encode()

        client_socket.send(message_with_header)
        print("Model update sent to server")

        # Receive acknowledgment from server
        ack = client_socket.recv(1024)
        print("Server acknowledgment:", ack.decode('utf-8'))

    except Exception as e:
        print(f"Error sending model update to server: {e}")
        # Close the socket connection in case of an error
        client_socket.close()
        print("Connection closed due to error")
    
    finally:
        client_socket.close()


# TODO: Should be replaced by load balancer (Optional)
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

    
# Wait for job completion (initialization job in this example)
wait_for_job_completion()

# Get pod name and ConfigMap data
pod_name = get_pod_name()
configmap_data = get_configmap_data(pod_name)

# Train local model if data is available
if configmap_data:
    Xtrain = configmap_data["Xtrain"]
    ytrain = configmap_data["ytrain"]
    coords = configmap_data["coords"]

    # Train a local model (replace with your actual model training)
    local_model = train_local_model(Xtrain, ytrain)

    # Get sample weights (e.g., set equal weights for all data points)
    sample_weight = np.ones(len(Xtrain))

    # Get server pod IP address (you can use service discovery mechanisms)
    server_ip = get_random_server_pod_ip(namespace="fed-relax")

    # Send model update to the server
    send_model_update_to_server(coords, local_model, Xtrain, ytrain, sample_weight, server_ip)
else:
    print("Error: ConfigMap data not found.")
