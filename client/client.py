import os
import pickle
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import base64
import time
from kubernetes import client, config
import sys
import requests
from flask import Flask, request, jsonify

def load_partitioned_data(data_dir='/pod-data'):
    # Get the pod's name from the environment variable
    pod_name = os.environ.get('MY_POD_NAME', 'unknown_pod')
    
    # Extract the pod index from the pod's name
    entries = os.listdir(data_dir)
    files = [entry for entry in entries if os.path.isfile(os.path.join(data_dir, entry))]
    data_file = files[0]
    
    # Construct the path to the data partition file
    partition_file = os.path.join(data_dir, data_file)
    
    # Check if the partition file exists
    if not os.path.exists(partition_file):
        raise FileNotFoundError(f"Partition file not found: {partition_file}")
    
    # Load the data from the partition file
    with open(partition_file, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Successfully loaded data for {pod_name} from {partition_file}")
    return data
    

def get_pod_name():
    # Get the pod name from the environment variable
    pod_name = os.environ.get('MY_POD_NAME')

    if pod_name:
        print("Pod name:", pod_name)
    else:
        print("Pod name not found.")
    return pod_name


def train_local_model(Xtrain, ytrain, max_depth=4):
    model = DecisionTreeRegressor(max_depth=max_depth)
    model.fit(Xtrain, ytrain)
    return model


def send_model_update_to_server(coords, model_params, Xtrain, ytrain, sample_weight, peer_ip="server-service", port=3000):
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
    return response  # Return the response object


def receive_data_from_server( peer_ip="server-service", port=3000):
    client_id = get_pod_name()
    SERVER_URL = f"http://{peer_ip}:{port}/send_data"
    try:
        headers = {'Content-Type': 'application/json', 'Client-ID': client_id}
        response = requests.post(SERVER_URL, headers=headers)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Extract the data from the response
            data_received = response.json()
            print("Data received from server:", data_received)
            return data_received
        else:
            # Print an error message if the request was not successful
            print(f"Failed to receive data from server. Status code: {response.status_code}")
            return None

    except Exception as e:
        # Print an error message if an exception occurs
        print(f"Error receiving data from server: {e}")
        return None


def FedRelaxClient(server_predictions, Xtrain, ytrain, sample_weight, regparam=0, maxiter=2):
    # Repeat the local updates (simultaneously at all nodes) for maxiter iterations
    for iter_GD in range(maxiter):
        for client, predictions in server_predictions.items():
            print(f"Client: {client}")
            for prediction in predictions:
                neighbourpred = np.array(prediction['neighbourpred'])
                Xtest = np.array(prediction['Xtest'])
                testsize = prediction['testsize']
                weight = prediction['weight']
                print("Neighbour predictions:", neighbourpred, "Xtest :", Xtest)

                # Augment local dataset by a new dataset obtained from the features of the test set
                neighbourpred = np.tile(neighbourpred, (1, len(ytrain[0])))  # Tile to match num features in ytrain
                ytrain = np.vstack((ytrain, neighbourpred))
                Xtrain = np.vstack((Xtrain, Xtest))

                # Set sample weights of added local dataset according to edge weight and GTV regularization parameter
                sampleweightaug = (regparam * len(ytrain) / testsize)

                # Reshape sample_weight to ensure compatible dimensions for stacking
                sample_weight_reshaped = sample_weight.reshape(-1, 1)

                sample_weight = np.vstack((sample_weight_reshaped, sampleweightaug * weight * np.ones((len(neighbourpred), 1))))
            
            # Fit the local model with the augmented dataset and sample weights
            local_model.fit(Xtrain, ytrain, sample_weight=sample_weight.reshape(-1))
            print("Local model has been trained with augmented dataset and sample weights")
    
    return local_model
        
data = load_partitioned_data()
Xtrain = data["Xtrain"]
ytrain = data["ytrain"]
coords = data["coords"]

# Train a local model (replace with your actual model training)
local_model = train_local_model(Xtrain, ytrain)

# Get sample weights (e.g., set equal weights for all data points)
sample_weight = np.ones(len(Xtrain))

# Retry sending model update to the server until success
while True:
    response = send_model_update_to_server(coords, local_model, Xtrain, ytrain, sample_weight)
    if response.status_code == 200:
        print("Model update sent successfully.")
        break  # Exit the loop if sending is successful
    else:
        print(f"Failed to send model update to server. Status code: {response.status_code}")
        print("Retrying...")
        time.sleep(10)  # Wait for a while before retrying


data_received = False

while not data_received:
    time.sleep(30)
    server_predictions = receive_data_from_server()
    if server_predictions is not None:
        data_received = True
        print(server_predictions)
    else:
        time.sleep(60)

FedRelaxClient(server_predictions, Xtrain, ytrain, sample_weight, regparam=0)
