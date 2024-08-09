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
import networkx as nx
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)
attributes_lock = threading.Lock()

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Initialize empty dictionary to store coords of nodes
all_client_coords = {}
# Initialize empty dictionary to store model updates of neighbours
neighbours_models = {}
# Keep track on neighbour nodes
neighbour_lists = []
local_train_errors, local_val_errors = [], []
all_train_errors, all_val_errors = [], []
desired_num_pods = 5
service_name = "processor-service"
namespace = "fed-relax"
port = 4000
current_iteration = 0

# Initialize Kubernetes client
config.load_incluster_config()
v1 = client.CoreV1Api()

# Get the list of pod URLs
def get_pod_list(service_name, namespace, port, desired_num_pods=desired_num_pods):
    """Get a list of all pods in the StatefulSet."""
    pod_list = []
    for i in range(desired_num_pods):  # Adjust as needed
        pod_name = f"processor-{i}"
        pod_list.append(f"http://{pod_name}.{service_name}.{namespace}.svc.cluster.local:{port}/receive_coords")
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

# Load data and train local model
data = load_partitioned_data()
Xtrain = data["Xtrain"]
ytrain = data["ytrain"]
Xval = data["Xval"]
yval = data["yval"]
coords = data["coords"]
Xtest = np.arange(0.0, 1, 0.1).reshape(-1, 1)

# Train a local model (replace with your actual model training)
local_model = train_local_model(Xtrain, ytrain)

# Get sample weights (e.g., set equal weights for all data points)
sample_weight = np.ones(len(Xtrain))

def decode_and_unpickle(encoded_data):
    try:
        decoded_data = base64.b64decode(encoded_data)
        unpickled_data = pickle.loads(decoded_data)
        return unpickled_data
    except Exception as e:
        app.logger.error("Error decoding and unpickling data: %s", e)
        raise

def process_all_coords(coords_received):
    global all_client_coords
    print("Decoding data..")
    try:
        pod_name = coords_received['pod_name']
        coords = decode_and_unpickle(coords_received['coords'])
        print(f"Received update from pod: {pod_name} with {coords}")

        pod_attributes = {"coords": coords}
        with attributes_lock:
            # Add the attributes to the global dictionary
            all_client_coords[pod_name] = pod_attributes

    except Exception as e:
        app.logger.error("Error processing client attributes: %s", e)
        return jsonify({"error": "An error occurred while processing client attributes"}), 500


def send_data(client_update, pod_urls, max_retries=100000, retry_delay=20):
    """Send coordinates to a list of pod URLs with retry logic."""
    responses = []  # Collect all responses
    for url in pod_urls:
        success = False
        attempt = 0
        
        while not success:
            try:
                print(f"Attempt {attempt + 1} to send coordinates to {url}")
                response = requests.post(url, json=client_update)
                
                if response.status_code == 200:
                    print(f"Coordinates sent successfully to {url}")
                    success = True
                else:
                    print(f"Failed to send coordinates to {url}. Status code: {response.status_code}")
                    attempt += 1
                    if attempt < max_retries:
                        print(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        logger.error(f"Max retries reached for {url}. Giving up.")
                        
            except Exception as e:
                logger.error("Error sending coordinates to server: %s", e)
                attempt += 1
                if attempt < max_retries:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Max retries reached for {url}. Giving up.")
        
        responses.append(response if success else None) 

    return responses


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


def add_edges_k8s(clients_attributes, nrneighbors=1):
    """
    Add edges to the graph based on pod attributes retrieved from Kubernetes config maps
    using k-nearest neighbors approach.
    """
    # Initialize empty graph
    graph = nx.Graph()

    # Add nodes to the graph with pod attributes
    for pod_name, attributes in clients_attributes.items():
        graph.add_node(pod_name, **attributes)

    # Build a numpy array containing node positions
    node_positions = np.array([attributes["coords"] for attributes in clients_attributes.values()], dtype=float)

    # Ensure there are enough samples to calculate neighbors
    if len(node_positions) <= nrneighbors:
        raise ValueError(f"Number of samples ({len(node_positions)}) is less than or equal to number of neighbors ({nrneighbors}).")

    # Calculate k-nearest neighbors graph
    A = kneighbors_graph(node_positions, n_neighbors=nrneighbors, mode='connectivity', include_self=False)

    # Iterate over the k-nearest neighbors graph and add edges with weights
    for i in range(len(clients_attributes)):
        for j in range(len(clients_attributes)):
            if A[i, j] > 0:
                pod_name_i = list(clients_attributes.keys())[i]
                pod_name_j = list(clients_attributes.keys())[j]

                # Calculate the Euclidean distance between pods based on their positions
                distance = np.linalg.norm(node_positions[i] - node_positions[j])

                # Add edge with weight based on distance
                graph.add_edge(pod_name_i, pod_name_j, weight=distance)

    return graph

    
# Function to visualize the graph and save the image
def visualize_and_save_graph(graph, output_path):
    pos = nx.spring_layout(graph)  # Compute layout for visualization
    plt.figure(figsize=(10, 10))
    nx.draw(graph, pos, with_labels=True, node_size=3000, node_color='skyblue', font_size=10, font_weight='bold')
    plt.title("Graph Visualization")
    plt.savefig(output_path)  # Save the image to a file
    print(f"Image is successfully saved in {output_path}")
    plt.show()  # Display the graph


def FedRelax(G, regparam=0, maxiter=5):
    # Determine the number of data points in the test set
    global neighbours_models, local_model, Xtrain, ytrain, Xtest, sample_weight, current_iteration
    testsize = Xtest.shape[0]
    for iter_GD in range(maxiter):
        current_iteration = iter_GD
        # Send local model and receive model from neighbours
        local_model_encoded = base64.b64encode(pickle.dumps(local_model)).decode('utf-8')
        iter_GD_encoded = base64.b64encode(pickle.dumps(iter_GD)).decode('utf-8')
        model_update = {
            "pod_name": os.environ.get('MY_POD_NAME'),
            "model": local_model_encoded,
            "iter": iter_GD_encoded
        }
        print("Send model to neighbour pods")
        send_data(model_update, neighbour_lists)

        # Wait until all the models from neighbours are received
        while len(neighbours_models) < len(neighbour_lists):
            time.sleep(90)
        if len(neighbours_models) == len(neighbour_lists):
            print("Received all models from neighbours: %s", neighbours_models)

        # Update local model
        for neighbour, attributes in neighbours_models.items():
            nmodel = attributes["model"]
            neighbourpred = nmodel.predict(Xtest).reshape(-1, 1)
            neighbourpred = np.tile(neighbourpred, (1, len(ytrain[0])))
            ytrain = np.vstack((ytrain, neighbourpred))
            Xtrain = np.vstack((Xtrain, Xtest))

            # Set sample weights of added local dataset according to edge weight and GTV regularization parameter
            sampleweightaug = (regparam * len(ytrain) / testsize)
            sample_weight_reshaped = sample_weight.reshape(-1, 1)
            sample_weight = np.vstack((sample_weight_reshaped, sampleweightaug * G.edges[(my_pod_name, neighbour)]["weight"] * np.ones((len(neighbourpred), 1))))

        # Fit the local model with the augmented dataset and sample weights
        local_model.fit(Xtrain, ytrain, sample_weight=sample_weight.reshape(-1))
        print("Local model has been trained with augmented dataset and sample weights, current counter: ", iter_GD)
        neighbours_models.clear()
    app.logger.debug("The FedRelax process is completed")
    return local_model


@app.route('/receive_coords', methods=['POST'])
def receive_all_coords():
    try:
        # Receive data from the client
        coords_received = request.get_json()
        app.logger.debug("Received client attributes: %s", coords_received)
        
        if not coords_received:
            app.logger.error("No data received in the request.")
            return jsonify({"error": "No data received."}), 400

        print("Decoding data...")  # Ensure this line is reached
        app.logger.debug("Decoding data...")

        pod_name = coords_received.get('pod_name')
        if not pod_name:
            app.logger.error("No 'pod_name' in received data.")
            return jsonify({"error": "No 'pod_name' in received data."}), 400

        coords_encoded = coords_received.get('coords')
        if not coords_encoded:
            app.logger.error("No 'coords' in received data.")
            return jsonify({"error": "No 'coords' in received data."}), 400

        coords = decode_and_unpickle(coords_encoded)
        print(f"Received update from pod: {pod_name} with coordinates: {coords}")
        app.logger.debug(f"Received update from pod: {pod_name} with coordinates: {coords}")

        pod_attributes = {"coords": coords}
        with attributes_lock:
            # Add the attributes to the global dictionary
            all_client_coords[pod_name] = pod_attributes
            app.logger.debug(f"Updated all_client_coords: {all_client_coords}")

        print("Current all_client_coords:", all_client_coords)
        app.logger.debug("Current all_client_coords: %s", all_client_coords)
        
        return jsonify({"message": "Data processed successfully."}), 200

    except ValueError as ve:
        app.logger.error(f"ValueError: {ve}")
        return jsonify({"error": str(ve)}), 400

    except Exception as e:
        app.logger.error("UnexpectedError: %s", e)
        return jsonify({"error": "An unexpected error occurred"}), 500
    

@app.route('/receive_models', methods=['POST'])
def receive_models():
    try:
        models_received = request.get_json()
        app.logger.debug("Received client attributes: %s", models_received)
        
        if not models_received:
            app.logger.error("No data received in the request.")
            return jsonify({"error": "No data received."}), 400

        print("Decoding data...")  # Ensure this line is reached
        app.logger.debug("Decoding data...")

        pod_name = models_received.get('pod_name')
        if not pod_name:
            app.logger.error("No 'pod_name' in received data.")
            return jsonify({"error": "No 'pod_name' in received data."}), 400

        models_encoded = models_received.get('model')
        if not models_encoded:
            app.logger.error("No 'model' in received data.")
            return jsonify({"error": "No 'model' in received data."}), 400
        
        iter_encoded = models_received.get('iter')
        if not iter_encoded:
            app.logger.error("No 'iter' in received data.")
            return jsonify({"error": "No 'iter' in received data."}), 400
        
        iter = decode_and_unpickle(iter_encoded)
        if iter != current_iteration:
            app.logger.error("Current iteration is not the same as received iteration: received=%s, current=%s", iter, current_iteration)
            return jsonify({"Current iteration is not the same as received iteration."}), 400

        model = decode_and_unpickle(models_encoded)
        
        print(f"Received update from pod: {pod_name} with coordinates: {model}")
        app.logger.debug(f"Received update from pod: {pod_name} with coordinates: {model}")

        pod_attributes = {"model": model}
        with attributes_lock:
            # Add the attributes to the global dictionary
            neighbours_models[pod_name] = pod_attributes
            app.logger.debug(f"Updated neighbours_models: {neighbours_models}")

        print("Current neighbours_models:", neighbours_models)
        app.logger.debug("Current neighbours_models: %s", neighbours_models)            
        
        return jsonify({"message": "Data processed successfully."}), 200

    except ValueError as ve:
        app.logger.error(f"ValueError: {ve}")
        return jsonify({"error": str(ve)}), 400

    except Exception as e:
        app.logger.error("UnexpectedError: %s", e)
        return jsonify({"error": "An unexpected error occurred"}), 500
    

def model_evaluation(model, iteration=None, output_path=None):
    trainerr = mean_squared_error(ytrain, model.predict(Xtrain))
    valerr = mean_squared_error(yval, model.predict(Xval))
    local_train_errors.append(trainerr)
    local_val_errors.append(valerr)

    if iteration == None:
        print(f"Final iteration: Training error: {trainerr}. Validation error: {valerr}")
        # Plot the errors across iterations
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(local_train_errors) + 1), local_train_errors, label='Training Error', marker='o')
        plt.plot(range(1, len(local_val_errors) + 1), local_val_errors, label='Validation Error', marker='o')
        
        # Add title and labels
        plt.title(f'Training vs Validation Error over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Mean Squared Error')
        plt.legend()
        plt.savefig(output_path)
        print(f"Image is successfully saved in {output_path}")
    else:
        print(f"Iteration {iteration}: Training error: {trainerr}. Validation error: {valerr}")


def main():
    # Wait until all pods are ready
    while not are_all_pods_ready():
        print("Waiting for all pods to be ready...")
        time.sleep(10)

    # Send coordinates to all other pods
    print("Broadcasting coordinates to other pods")
    coords_encoded = base64.b64encode(pickle.dumps(coords)).decode('utf-8')
    coord_update = {
        "pod_name": os.environ.get('MY_POD_NAME'),
        "coords": coords_encoded
    }
    send_data(coord_update, pod_urls)

    while len(all_client_coords) < desired_num_pods-1:
        time.sleep(90)

    if len(all_client_coords) == desired_num_pods-1:
        pod_attributes = {"coords": coords}
        all_client_coords[my_pod_name] = pod_attributes
        print("Graph after being fully updated", all_client_coords)
        knn_graph = add_edges_k8s(all_client_coords)
        visualize_and_save_graph(knn_graph, '/app/knn_graph_{}.png'.format(my_pod_name))

    # Get neighbour list
    global neighbour_lists
    for node_j in knn_graph.neighbors(my_pod_name):
        neighbour_lists.append(f"http://{node_j}.{service_name}.{namespace}.svc.cluster.local:{port}/receive_models")
    print("The neighbour pods for current local pod are: ", neighbour_lists)

    print("Running FedRelax algorithm...")
    final_model = FedRelax(knn_graph, regparam=0)
    print("Evaluating final model")
    model_evaluation(final_model, None, '/app/local_mse_{}.png'.format(my_pod_name))

if __name__ == '__main__':
    # Start Flask server in a separate thread
    threading.Thread(target=app.run, kwargs={'host': '0.0.0.0', 'port': 4000, 'threaded': True}).start()

    # Run the main logic
    main()
