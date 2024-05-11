from http.client import responses
import pickle
import json
import numpy as np
from sklearn.neighbors import kneighbors_graph
import base64
from flask import Flask, request, jsonify, Response
import networkx as nx
import matplotlib.pyplot as plt
import datetime
import time

app = Flask(__name__)

# Initialize empty dictionary to store client attributes
all_client_attributes = {}
# Initialize empty dictionary to store all neighbour predictions's attributes
data_to_sends = dict()
desired_num_pods = 2

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


def FedRelax(Xtest, knn_graph, client_attributes, namespace="fed-relax", regparam=0, maxiter=100):
    # Determine the number of data points in the test set
    testsize = Xtest.shape[0]
    G = knn_graph

    # Attach models and weights to nodes
    for node_i in G.nodes(data=False):
        G.nodes[node_i]["model"] = client_attributes[node_i]["model"]
        G.nodes[node_i]["sample_weight"] = client_attributes[node_i]["sample_weight"]

    # Iterate and share predictions
    for node_i in G.nodes(data=False):
        neighbour_lists = []
        for node_j in G.neighbors(node_i):
            # Share predictions with neighbors
            neighbourpred = G.nodes[node_j]["model"].predict(Xtest).reshape(-1, 1)
            # Prepare data to send to client
            data_to_send = {
                "neighbourpred": neighbourpred.tolist(),
                "Xtest": Xtest.tolist(),
                "testsize": testsize,
                "weight": G.edges[(node_i, node_j)]["weight"]
            }

            neighbour_lists.append(data_to_send)  # Append data_to_send to the list
        data_to_sends[node_i] = neighbour_lists

    return data_to_sends, G


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
    global all_client_attributes
    
    # Receive client update
    pod_name = client_update['pod_name']

    # Decode model parameters, sample weights, coords
    model_params = decode_and_unpickle(client_update['model_params'])
    sample_weight = client_update['sample_weight']  # No need to decode sample_weight
    coords = decode_and_unpickle(client_update['coords'])
    print(f"Received update from pod: {pod_name} with {model_params}, {sample_weight}, {coords}")

    # Update node information in the graph
    pod_attributes = {
        "coords": coords,
        "model": model_params,
        "sample_weight": sample_weight
    }
    
    # Add the attributes to the global dictionary
    all_client_attributes[pod_name] = pod_attributes

    # Log information about received update with timestamp
    with open("server_logs.txt", "a") as log_file:
        log_file.write(f"Received update from pod: {pod_name} at {datetime.datetime.now()}\n")
            

# Trigger global model update after receiving updates from all clients
def runFedRelax(client_attributes):
    print("Running FedRelax")
    Xtest = np.arange(0.0, 1, 0.1).reshape(-1, 1)

    print("Graph after being fully updated", client_attributes)
    knn_graph = add_edges_k8s(client_attributes)
    visualize_and_save_graph(knn_graph, '/app/knn_graph.png')
    # TODO: Modify aggregate_updates by using FedRelax
    data,final_graph = FedRelax(Xtest, knn_graph, client_attributes)
    visualize_and_save_graph(final_graph, '/app/fin_graph.png')


@app.route('/send_data', methods=['POST'])
def send_data_to_client():
    global data_to_sends
    print("Sending neighbour updates to client", data_to_sends)

    while len(data_to_sends) < desired_num_pods:
        # Wait until data_to_sends has enough data
        time.sleep(30)
        
    try:
        # Add Content-Type header to the response
        response_data = json.dumps(data_to_sends)
        response = Response(response_data, status=200, mimetype='application/json')

        print("Sending response:", response)
        return response

    except Exception as e:
        print("Error sending data to client:", e)
        return jsonify({"error": str(e)}), 400


@app.route('/receive_data', methods=['POST'])
def receive_model_update():
    try:
        # Receive data from the client
        data = request.get_json()
        print("Received client attributes", data)

        # Process the received JSON data
        process_client_attributes(data)   
        
        # Check if all pods have sent their attributes
        if len(all_client_attributes) == desired_num_pods:
            runFedRelax(all_client_attributes)
            all_client_attributes.clear()

        # Send the response
        return jsonify({"message": "Data processed successfully."}), 200

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
