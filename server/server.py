import socket
import pickle
import json
from collections import defaultdict
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import datetime
from kubernetes import client, config
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph
import base64
import pickle

# Define a class to represent a node in the FedRelax graph
class Node:
    def __init__(self, pod_name, model, weight, coords):
        self.name = pod_name
        self.model = None  # Initialize local model
        self.weight = None  # Store received client update
        self.coords = None


# Function to receive data from a client socket
def receive_data(client_socket):
    try:
        message_header = client_socket.recv(2)
        if not len(message_header):
            return False
        message_length = int.from_bytes(message_header, byteorder='big')
        data = client_socket.recv(message_length)
        return data.decode()
    except:
        return False


# Function to send global model froms server back to client
def send_global_model_to_client(global_model, client_socket):
    try:
        # Encode global model using base64
        global_model_encoded = base64.b64encode(pickle.dumps(global_model)).decode('utf-8')

        # Send global model to client
        client_socket.sendall(global_model_encoded.encode())
        ack = client_socket.recv(1024)
        if ack.decode('utf-8') == "ACK":
            print("Global model acknowledgment received from client")
        else:
            print("Error: Failed to receive acknowledgment from client")
    except Exception as e:
        print(f"Error sending global model to client: {e}")


def add_edges_k8s(coords,namespace="fed-relax", nrneighbors=3, pos='coords', refdistance=1):
    """
    Add edges to the graph based on pod attributes retrieved from Kubernetes config maps
    using k-nearest neighbors approach.
    """
    
    # Initialize empty graph
    graph = nx.Graph()
    
    # Build a numpy array containing node positions
    node_positions = np.array([coords], dtype=float)
    
    # Calculate k-nearest neighbors graph
    A = kneighbors_graph(node_positions, n_neighbors=nrneighbors, mode='connectivity', include_self=False)
    
    # Iterate over the k-nearest neighbors graph and add edges with weights
    for i in range(len(pod_info)):
        for j in range(len(pod_info)):
            if A[i, j] > 0:
                pod_name_i = list(pod_info.keys())[i]
                pod_name_j = list(pod_info.keys())[j]
                
                # Calculate the Euclidean distance between pods based on their positions
                distance = np.linalg.norm(node_positions[i] - node_positions[j])
                
                # Add edge with weight based on distance
                graph.add_edge(pod_name_i, pod_name_j, weight=distance)
    
    return graph


# Function to aggregate model updates from clients
def aggregate_updates(Xtest, client_updates):
    # Initialize lists to store weights and predictions
    all_weights = []
    all_predictions = []
    
    # Extract weights and predictions from client updates
    for update in client_updates:
        all_weights.append(update['sample_weight'])
        model = decode_and_unpickle(update['model_params'])
        prediction = model.predict(Xtest)
        all_predictions.append(prediction)
    
    # Concatenate lists of arrays into single NumPy arrays
    all_weights_concatenated = np.concatenate(all_weights)
    all_predictions_concatenated = np.concatenate(all_predictions)
    
    # Check shapes of concatenated arrays
    print("Sample weights shape:", all_weights_concatenated.shape)
    print("Predictions shape:", all_predictions_concatenated.shape)
    
    # Aggregate model updates
    average_model = DecisionTreeRegressor().fit(Xtest, np.average(all_predictions_concatenated, weights=all_weights_concatenated, axis=0))
    return average_model


# Function to visualize the graph and save the image
def visualize_and_save_graph(graph, output_path):
    pos = nx.spring_layout(graph)  # Compute layout for visualization
    plt.figure(figsize=(10, 10))
    nx.draw(graph, pos, with_labels=True, node_size=3000, node_color='skyblue', font_size=10, font_weight='bold')
    plt.title("Graph Visualization")
    plt.savefig(output_path)  # Save the image to a file
    print(f"Image is successfully saved in {output_path}")
    plt.show()  # Display the graph


def decode_and_unpickle(encoded_data):
    decoded_data = base64.b64decode(encoded_data)
    unpickled_data = pickle.loads(decoded_data)
    return unpickled_data

# Server-side script for FedRelax on Kubernetes
def main():
    # Set up socket server
    HOST = socket.gethostbyname(socket.gethostname())  # Get server's IP address
    PORT = 3000
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))
    server_socket.listen()
    print(f'Listening for connections on {HOST}:{PORT}...')

    # Initialize empty graph
    graph = defaultdict(Node)

    # Global model and test data
    global_model = None
    Xtest = np.arange(0.0, 1, 0.1).reshape(-1, 1) 
    # TODO: Replace by the number of nodes inside graph
    desired_num_clients = 1

    while True:
        client_socket, _ = server_socket.accept()
        data = receive_data(client_socket)
        if data:
            try:
                # Receive client update
                client_update = json.loads(data)
                pod_name = client_update['pod_name']

                # Decode model parameters, sample weights, coords
                model_params = decode_and_unpickle(client_update['model_params'])
                sample_weight = client_update['sample_weight']  # No need to decode sample_weight
                coords = decode_and_unpickle(client_update['coords'])
                print(f"Received update from pod: {pod_name} with {model_params}, {sample_weight}, {coords}")

                # Decode client updates serialized data
                client_update['model_params'], client_update['coords'] = model_params, coords

                # Update node information in the graph
                if pod_name not in graph:
                    graph[pod_name] = Node(pod_name, model_params, sample_weight, coords)
                graph[pod_name].update = client_update

                # Send a simple acknowledgment message to the client
                ack_message = json.dumps({"message": "Update received"}).encode()
                client_socket.sendall(ack_message)

                # Log information about received update with timestamp
                with open("server_logs.txt", "a") as log_file:
                    log_file.write(f"Received update from pod: {pod_name} at {datetime.datetime.now()}\n")

                # Trigger global model update after receiving updates from all clients
                if len(graph) == desired_num_clients:
                    client_updates = [node.update for node in graph.values() if node.update is not None]
                    print(client_updates)
                    knn_graph = add_edges_k8s(coords)
                    visualize_and_save_graph(knn_graph, '/app/knn_graph.png')
                    # TODO: Modify aggregate_updates by using FedRelax
                    # global_model = aggregate_updates(Xtest, client_updates)

                    print("Sending global model back to client")
                    send_global_model_to_client(global_model, client_socket)

                    # Reset client updates for the next round
                    for node in graph.values():
                        node.update = None

            except Exception as e:
                print(f"Error processing client update: {e}")
        else:
            client_socket.close()

if __name__ == "__main__":
    main()
