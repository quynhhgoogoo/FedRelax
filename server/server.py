import socket
import pickle
import json
from collections import defaultdict
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import datetime

# Define a class to represent a node in the FedRelax graph
class Node:
    def __init__(self, pod_name, attributes):
        self.name = pod_name
        self.attributes = attributes
        self.model = None  # Initialize local model
        self.update = None  # Store received client update


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


# Function to aggregate model updates from clients
def aggregate_updates(Xtest, client_updates):
    # TODO: Implement aggregation logic
    all_weights = np.concatenate([update['sample_weight'] for update in client_updates])
    all_predictions = np.concatenate([update['model'].predict(Xtest) for update in client_updates])
    average_model = DecisionTreeRegressor().fit(Xtest, np.average(all_predictions, weights=all_weights, axis=0))
    return average_model


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
    desired_num_clients = 3

    while True:
        client_socket, _ = server_socket.accept()
        data = receive_data(client_socket)
        if data:
            try:
                # Receive client update: pod info, model parameters
                client_update = json.loads(data)
                pod_name = client_update['pod_name']
                print(f"Received update from pod: {pod_name}")

                # Update node information in the graph
                if pod_name not in graph:
                    graph[pod_name] = Node(pod_name, client_update['attributes'])
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
                    global_model = aggregate_updates(Xtest, client_updates)

                    # Send updated global model to clients (implementation not shown here)
                    # Reset client updates for the next round
                    for node in graph.values():
                        node.update = None

            except Exception as e:
                print(f"Error processing client update: {e}")
        else:
            client_socket.close()

if __name__ == "__main__":
    main()
