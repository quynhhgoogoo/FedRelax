import requests
import pickle
import json
import numpy as np
from sklearn.neighbors import kneighbors_graph
import base64
from flask import Flask, request, jsonify
import networkx as nx
import matplotlib.pyplot as plt
import datetime

app = Flask(__name__)

# Initialize empty dictionary to store client attributes
client_attributes = {}

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

    # Calculate k-nearest neighbors graph
    A = kneighbors_graph(node_positions, n_neighbors=nrneighbors, mode='connectivity', include_self=False)

    # Iterate over the k-nearest neighbors graph and add edges with weights
    for i in range(len(clients_attributes)):
        for j in range(len(clients_attributes)):
            if A[i, j] > 0:
                pod_name_i = list(clients_attributes.keys())[i]
                pod_name_j = list(clients_attributes.keys())[j]  # Fix index issue

                # Calculate the Euclidean distance between pods based on their positions
                distance = np.linalg.norm(node_positions[i] - node_positions[j])

                # Add edge with weight based on distance
                graph.add_edge(pod_name_i, pod_name_j, weight=distance)

    return graph


# Function to decode and unpickle data
def decode_and_unpickle(encoded_data):
    decoded_data = base64.b64decode(encoded_data)
    unpickled_data = pickle.loads(decoded_data)
    return unpickled_data


# Server-side script for FedRelax on Kubernetes
def FedRelax(Xtest, knn_graph, client_attributes, namespace="fed-relax", regparam=0, maxiter=100):
    # Determine the number of data points in the test set
    testsize = Xtest.shape[0]
    G = knn_graph

    # Attach a DecisionTreeRegressor as the local model to each node in G
    for node_i in G.nodes(data=False):
        G.nodes[node_i]["model"] = client_attributes[node_i]["model"]
        G.nodes[node_i]["sample_weight"] = client_attributes[node_i]["sample_weight"]  # Initialize sample weights

    # Iterate over all nodes in the graph
    for node_i in G.nodes(data=False):
        # Share predictions with neighbors
        for node_j in G.neighbors(node_i):
            # Add the predictions of the current hypothesis at node j as labels
            neighbourpred = G.nodes[node_j]["model"].predict(Xtest).reshape(-1, 1)

            # Prepare data to be sent back to clients
            data_to_send = {
                "neighbourpred": neighbourpred.tolist(),
                "Xtest": Xtest.tolist(),
                "testsize": testsize
            }
            print("Sending data to client", data_to_send)

            # Send the data back to clients
            data_to_send_encoded = json.dumps(data_to_send).encode()

    return data_to_send_encoded


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

def process_client_attributes(client_update):
    client_attributes = {}

    # Receive client update
    # client_update = json.loads(data)
    pod_name = client_update['pod_name']

    # Decode model parameters, sample weights, coords
    model_params = decode_and_unpickle(client_update['model_params'])
    sample_weight = client_update['sample_weight']  # No need to decode sample_weight
    coords = decode_and_unpickle(client_update['coords'])
    print(f"Received update from pod: {pod_name} with {model_params}, {sample_weight}, {coords}")

    # Decode client updates serialized data
    try:
        client_update['model_params'], client_update['coords'] = model_params, coords
    except Exception as e:
        print(f"Error decoding client updates: {e}")
        print(f"Received data: {client_update}")

    # Update node information in the graph
    pod_attributes = {
        "coords": coords,
        "model": model_params,
        "sample_weight": sample_weight
    }
    client_attributes[pod_name] = pod_attributes
    print(client_attributes)

    # Log information about received update with timestamp
    with open("server_logs.txt", "a") as log_file:
        log_file.write(f"Received update from pod: {pod_name} at {datetime.datetime.now()}\n")
            
    return client_attributes


# Server-side script for FedRelax on Kubernetes
def main(client_attributes):
    # TODO: Modify this value to the number of client pods
    desired_num_clients = 2
    Xtest = np.arange(0.0, 1, 0.1).reshape(-1, 1)

    # Trigger global model update after receiving updates from all clients
    if len(client_attributes) == desired_num_clients:
        print("Graph after being fully updated", client_attributes)
        knn_graph = add_edges_k8s(client_attributes)
        visualize_and_save_graph(knn_graph, '/app/knn_graph.png')
        # TODO: Modify aggregate_updates by using FedRelax
        final_graph = FedRelax(Xtest, knn_graph, client_attributes)
        visualize_and_save_graph(final_graph, '/app/fin_graph.png')


@app.route('/send_data', methods=['POST'])
def send_global_model_to_client():
    try:
        data = request.get_json()
        #main(data)
        return jsonify({"message": "Data processed successfully."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/receive_data', methods=['POST'])
def receive_model_update():
    try:
        # Receive data from the client
        data = request.get_json()
        print("Received client attributes", data)

        # Process the received JSON data
        client_attributes = process_client_attributes(data)
        print("Processed attributes:", client_attributes)

        # Send the processed data back to the client
        return jsonify({"message": "Data processed successfully."}), 200
    except Exception as e:
        # Handle exceptions
        print("Error:", e)
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
