import os
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
import signal
import sys
import datetime
import select
import requests
from flask import Flask, request, jsonify

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


def send_model_update_to_server(coords, model_params, Xtrain, ytrain, sample_weight, peer_ip="server-service", port=3000):
    try:
        model_params_encoded = base64.b64encode(pickle.dumps(model_params)).decode('utf-8')
        coords_encoded = base64.b64encode(pickle.dumps(coords)).decode('utf-8')

        if isinstance(sample_weight, np.ndarray):
            # Convert sample_weight to a list
            sample_weight_list = sample_weight.tolist()
        else:
            # No conversion needed if it's already a list
            sample_weight_list = sample_weight

        # Create a dictionary containing model parameters, training data, and sample weights
        client_update = {
            "pod_name": get_pod_name(),
            "coords": coords_encoded,
            "model_params": model_params_encoded,
            "sample_weight": sample_weight_list,  # Use the converted list
        }

        # URL of the server endpoint
        SERVER_URL = f"http://{peer_ip}:{port}/receive_data"

        # Send the data to the server
        response = requests.post(SERVER_URL, json=client_update)

        if response.status_code == 200:
            print("Model update sent to server successfully")
            # Parse the response if needed
            response_data = response.json()
            print("Response from server:", response_data)
        else:
            print(f"Failed to send model update to server. Status code: {response.status_code}")

    except Exception as e:
        print(f"Error sending model update to server: {e}")


def receive_data_from_server(peer_ip="server-service", port=3000):
    SERVER_URL = f"http://{peer_ip}:{port}/send_data"
    try:
        # Make a POST request to the server's endpoint
        response = requests.post(SERVER_URL)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Extract the data from the response
            data_received = response.json()
            print("Data received from server:", data_received)
            return data_received["data"]  # Assuming the server returns the data in the "data" field
        else:
            # Print an error message if the request was not successful
            print(f"Failed to receive data from server. Status code: {response.status_code}")
            return None

    except Exception as e:
        # Print an error message if an exception occurs
        print(f"Error receiving data from server: {e}")
        return None
    

# TODO: Remove this after replace ConfigMap by Docker Volume
# wait_for_job_completion()

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

    # Send model update to the server
    send_model_update_to_server(coords, local_model, Xtrain, ytrain, sample_weight)
else:
    print("Error: ConfigMap data not found.")


data_received = False

while not data_received:
    received_data = receive_data_from_server()
    if received_data is not None:
        data_received = True
        print(received_data)
    else:
        time.sleep(60)
