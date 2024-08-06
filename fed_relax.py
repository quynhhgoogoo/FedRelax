import os
import pickle
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import base64
import time
import requests
import threading
from sklearn.metrics import mean_squared_error
import logging
from flask import Flask, request, jsonify, Response
from kubernetes import client, config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Initialize empty dictionary to store coords of nodes
all_client_coords = {}
# Initialize empty dictionary to store model updates of neighbours
neighbours_models = {}
desired_num_pods = 3
service_name = "processor-service"
namespace = "fed-relax"
port = 4000

# Initialize Kubernetes client
config.load_incluster_config()
v1 = client.CoreV1Api()

# Get the list of pod URLs
def get_pod_list(service_name, namespace, port, desired_num_pods=desired_num_pods):
    """Get a list of all pods in the StatefulSet."""
    pod_list = []
    for i in range(desired_num_pods):  # Adjust as needed
        pod_name = f"processor-{i}"
        pod_list.append(f"http://{pod_name}.{service_name}.{namespace}.svc.cluster.local:{port}/")
    return pod_list

pod_urls = get_pod_list(service_name, namespace, port)

# Remove the URL of the current pod from the list
my_pod_name = os.getenv('MY_POD_NAME')
my_pod_url = f"http://{my_pod_name}.{service_name}.{namespace}.svc.cluster.local:{port}/receive_coords"
pod_urls = [url for url in pod_urls if url != my_pod_url]

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

def train_local_model(Xtrain, ytrain, max_depth=4):
    model = DecisionTreeRegressor(max_depth=max_depth)
    model.fit(Xtrain, ytrain)
    return model

def decode_and_unpickle(encoded_data):
    decoded_data = base64.b64decode(encoded_data)
    unpickled_data = pickle.loads(decoded_data)
    return unpickled_data

def process_all_coords(coords_received):
    global all_client_coords
    try:
        pod_name = coords_received['pod_name']
        coords = decode_and_unpickle(coords_received['coords'])
        print(f"Received update from pod: {pod_name} with {coords}")

        pod_attributes = {"coords": coords}
        all_client_coords[pod_name] = pod_attributes

    except Exception as e:
        app.logger.error("Error processing client attributes: %s", e)
        return jsonify({"error": "An error occurred while processing client attributes"}), 500

def send_coordinates(coords, pod_urls):
    """Send coordinates to a list of pod URLs."""
    print("Broadcasting coordinates to other pods")
    coords_encoded = base64.b64encode(pickle.dumps(coords)).decode('utf-8')
    for url in pod_urls:
        try:
            url += "/receive_coords"
            # Create a dictionary containing model parameters, training data, and sample weights
            client_update = {
                "pod_name": os.environ.get('MY_POD_NAME'),
                "coords": coords_encoded
            }
            print("Broadcasting coordinates to ", url)
            # Send the data to the server
            response = requests.post(url, json=client_update)
            if response.status_code == 200:
                print(f"Coordinates sent successfully to {url}")
            else:
                print(f"Failed to send coordinates to {url}. Status code: {response.status_code}")
            return response  # Return the response object

        except Exception as e:
            logger.error("Error sending coordinates to server: %s", e)
            return None

def are_all_pods_ready():
    """Check if all pods in the StatefulSet are ready."""
    try:
        # Fetch the list of pods with the specified label
        pods = v1.list_namespaced_pod(namespace=namespace, label_selector="app=fedrelax-client")

        # Check if each pod is in the 'Running' phase and is ready
        ready_pods = [pod for pod in pods.items if pod.status.phase == "Running"]
        
        # Log the ready pods
        logger.debug(f"Ready pods: {[pod.metadata.name for pod in ready_pods]}")

        # Check if the number of ready pods meets the desired count
        return len(ready_pods) == desired_num_pods
    except Exception as e:
        logger.error("Error checking pod readiness: %s", e)
        return False

    
@app.route('/receive_coords', methods=['POST'])
def receive_all_coords():
    try:
        # Receive data from the client
        data = request.get_json()
        app.logger.debug("Received client attributes %s", data)

        # Process the received JSON data
        process_all_coords(data)   
        
        # Check if all pods have sent their attributes
        if len(all_client_coords) == desired_num_pods - 1:
            print("Received all coords of nodes across graph: ", all_client_coords)

        # Send the response
        return jsonify({"message": "Data processed successfully."}), 200

    except ValueError as ve:
        logging.error(f"ValueError: {ve}")
        return jsonify({"error": str(ve)}), 400

    except Exception as e:
        print("UnexpectedError:", e)
        app.logger.error("UnexpectedError: %s", e)
        return jsonify({"error": "An unexpected error occurred"}), 500
    

def main():
    # Wait until all pods are ready
    while not are_all_pods_ready():
        print("Waiting for all pods to be ready...")
        time.sleep(10)

    # Load data and train local model
    data = load_partitioned_data()
    Xtrain = data["Xtrain"]
    ytrain = data["ytrain"]
    Xval = data["Xval"]
    yval = data["yval"]
    coords = data["coords"]

    # Send coordinates to all other pods
    send_coordinates(coords, pod_urls)

if __name__ == '__main__':
    # Start Flask server in a separate thread
    threading.Thread(target=app.run, kwargs={'host': '0.0.0.0', 'port': 4000, 'threaded': True}).start()

    # Run the main logic
    main()
