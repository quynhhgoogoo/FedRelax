import os
import pickle
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import base64
import time
from kubernetes import client, config
import sys
import requests
from sklearn.metrics import mean_squared_error

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
    print("Model update is sent to server", response)
    return response  # Return the response object


def send_evaluations_to_server(trained_local_model, train_features, train_labels, val_features, val_labels, peer_ip="server-service", port=3000):
    trainerr = mean_squared_error(train_labels, trained_local_model.predict(train_features))
    valerr = mean_squared_error(val_labels, trained_local_model.predict(val_features))

    # Send evaluation parameters to server
    trained_local_model_encoded = base64.b64encode(pickle.dumps(trained_local_model)).decode('utf-8')
    val_features_encoded = base64.b64encode(pickle.dumps(val_features)).decode('utf-8')
    val_labels_encoded = base64.b64encode(pickle.dumps(val_labels)).decode('utf-8')
    trainerr_encoded = base64.b64encode(pickle.dumps(trainerr)).decode('utf-8')
    valerr_encoded = base64.b64encode(pickle.dumps(valerr)).decode('utf-8')

    # Create a dictionary containing evaluation data
    model_update = {
        "pod_name": get_pod_name(),
        "model": trained_local_model_encoded,
        "val_features": val_features_encoded,
        "val_labels": val_labels_encoded,
        "trainerr": trainerr_encoded,
        "valerr": valerr_encoded
    }

    # URL of the server endpoint
    SERVER_URL = f"http://{peer_ip}:{port}/receive_model"

    # Send the data to the server
    response = requests.post(SERVER_URL, json=model_update)
    print("Final model is sent to server", response)
    return response


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
Xval = data["Xval"]
yval = data["yval"]
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

final_model = FedRelaxClient(server_predictions, Xtrain, ytrain, sample_weight, regparam=0)

# Retry sending local model evaluation to the server until success
while True:
    response = send_evaluations_to_server(final_model, Xtrain, ytrain, Xval, yval)
    if response.status_code == 200:
        print("Final model sent successfully.")
        break 
    else:
        print(f"Failed to send final model to server. Status code: {response.status_code}")
        print("Retrying...")
        time.sleep(10) 

# Keep pods alive after procedure
while True:
    time.sleep(120)