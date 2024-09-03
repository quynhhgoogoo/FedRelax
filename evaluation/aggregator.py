import base64
from flask import Flask, request, jsonify, Response
import logging
import matplotlib.pyplot as plt
import numpy as np
import pickle
import threading
import time

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Initialize empty dictionary to store client attributes for model evaluation
all_client_attributes = {}
attributes_lock = threading.Lock()
desired_num_pods = 20

def decode_and_unpickle(encoded_data):
    try:
        decoded_data = base64.b64decode(encoded_data)
        unpickled_data = pickle.loads(decoded_data)
        return unpickled_data
    except Exception as e:
        app.logger.error("Error decoding and unpickling data: %s", e)
        raise


def validation_graph():
    global all_client_attributes
    X_val = all_client_attributes.get("processor-0")["Xval"]
    y_1 = all_client_attributes.get("processor-11")["model"].predict(X_val)
    y_2 = all_client_attributes.get("processor-5")["model"].predict(X_val)

    # Plot the results
    plt.figure()
    plt.plot(X_val, y_1, color="orange", label="validation data cluster 0", linewidth=2)
    plt.plot(X_val, y_2, color="green", label="validation data cluster 0", linewidth=2)
    plt.plot(all_client_attributes.get("processor-7")["Xval"], all_client_attributes.get("processor-0")["yval"], color="blue", label="validation data cluster 0", linewidth=2)
    plt.plot(all_client_attributes.get("processor-15")["Xval"], all_client_attributes.get("processor-11")["yval"], color="red", label="val data second cluster", linewidth=2)
    plt.title("Comparison of Model Predictions and Actual Validation Data", fontsize=14)
    plt.xlabel("Validation Feature Data (X_val)", fontsize=12)
    plt.ylabel("Predicted/Actual Values", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.savefig('/app/validation.png')
    print(f"Validation graph is successfully saved in /app/validation.png")

def PlotGraph(graphin, pos='coords', annotate='name'):
    # the numpy array x will hold the horizontal coord of markers for each node in emp. graph graphin
    x = np.zeros(len(graphin.nodes))
    # vertical coords of markers 
    y = np.zeros(len(graphin.nodes))

    # Create a mapping between node names and their corresponding indices
    name_to_index = {name: i for i, name in enumerate(graphin.nodes)}

    for edge_dmy in graphin.edges:
        node_i = name_to_index[edge_dmy[0]]  # Extract node indices from the edge tuple
        node_j = name_to_index[edge_dmy[1]]  # Extract node indices from the edge tuple

        x[node_i] = graphin.nodes[edge_dmy[0]][pos][0]
        x[node_j] = graphin.nodes[edge_dmy[1]][pos][0]

        y[node_i] = graphin.nodes[edge_dmy[0]][pos][1]
        y[node_j] = graphin.nodes[edge_dmy[1]][pos][1]

    # standareize the coordinates of node markers 
    x = (x - np.min(x, axis=0)) / np.std(x, axis=0) + 1 
    y = (y - np.min(y, axis=0)) / np.std(y, axis=0) + 1 

    # create a figure with prescribed dimensions 
    fig, ax = plt.subplots(figsize=(10, 10))

    # generate a scatter plot with each marker representing a node in graphin
    ax.scatter(x, y, 300, marker='o', color='Black')

    # draw links between two nodes if they are connected by an edge 
    # in the empirical graph. use the "weight" of the edge to determine the line thickness
    for edge_dmy in graphin.edges:
        node_i = name_to_index[edge_dmy[0]]  # Extract node indices from the edge tuple
        node_j = name_to_index[edge_dmy[1]]  # Extract node indices from the edge tuple

        ax.plot([x[node_i], x[node_j]], [y[node_i], y[node_j]], c='black', lw=4 * graphin.edges[edge_dmy]["weight"])

    # annotate each marker by the node attribute whose name is stored in the input parameter "annotate"
    for iter_node in graphin.nodes:
        i = name_to_index[iter_node]
        if annotate in graphin.nodes[iter_node]:
            ax.annotate(str(graphin.nodes[iter_node][annotate]), (x[i] + 0.2, 0.995 * y[i]), c="red")
        else:
            ax.annotate(str(iter_node), (x[i] + 0.2, 0.995 * y[i]), c="red")

    ax.set_ylim(0.9 * np.min(y), 1.1 * np.max(y))
    ax.set_xlim(0.9 * np.min(x), 1.1 * np.max(x))

    plt.savefig('/app/final.png')  # Save the image to a file
    print(f"Final graph is successfully saved in /app/final.png")


@app.route('/receive_attributes', methods=['POST'])
def receive_attributes():
    try:
        # Receive data from the client
        received_data = request.get_json()
        app.logger.debug("Received client attributes: %s", received_data)
        
        if not received_data:
            app.logger.error("No data received in the request.")
            return jsonify({"error": "No data received."}), 400
        
        app.logger.debug("Decoding data...")

        pod_name = received_data.get('pod_name')
        if not pod_name:
            app.logger.error("No 'pod_name' in received data.")
            return jsonify({"error": "No 'pod_name' in received data."}), 400

        coords_encoded = received_data.get('coords')
        if not coords_encoded:
            app.logger.error("No 'coords' in received data.")
            return jsonify({"error": "No 'coords' in received data."}), 400
        
        model_encoded = received_data.get('model')
        if not model_encoded:
            app.logger.error("No 'model' in received data.")
            return jsonify({"error": "No 'model' in received data."}), 400

        Xval_encoded = received_data.get('Xval')
        if not Xval_encoded:
            app.logger.error("No 'Xval' in received data.")
            return jsonify({"error": "No 'Xval' in received data."}), 400
        
        yval_encoded = received_data.get('yval')
        if not yval_encoded:
            app.logger.error("No 'yval' in received data.")
            return jsonify({"error": "No 'yval' in received data."}), 400
        
        graph_encoded = received_data.get('graph')
        if not graph_encoded:
            app.logger.error("No 'graph' in received data.")
            return jsonify({"error": "No 'graph' in received data."}), 400

        coords = decode_and_unpickle(coords_encoded)
        model = decode_and_unpickle(model_encoded)
        Xval = decode_and_unpickle(Xval_encoded)
        yval = decode_and_unpickle(yval_encoded)
        graph = decode_and_unpickle(graph_encoded)

        app.logger.debug(f"Received update from pod: {pod_name} with coordinates {coords}, model {model}, Xval {Xval}, yval {yval}")

        pod_attributes = {"coords": coords, "model": model, "Xval": Xval, "yval": yval, "graph": graph}
        with attributes_lock:
            # Add the attributes to the global dictionary
            all_client_attributes[pod_name] = pod_attributes
            app.logger.debug(f"Updated all_client_coords: {all_client_attributes}")

        print("Current all_client_coords:", all_client_attributes)
        app.logger.debug("Current all_client_coords: %s", all_client_attributes)
        
        return jsonify({"message": "Data processed successfully."}), 200

    except ValueError as ve:
        app.logger.error(f"ValueError: {ve}")
        return jsonify({"error": str(ve)}), 400

    except Exception as e:
        app.logger.error("UnexpectedError: %s", e)
        return jsonify({"error": "An unexpected error occurred"}), 500

def main():
    global all_client_attributes
    while len(all_client_attributes) < desired_num_pods:
        logger.debug("all_client_attributes: %d. Desired_num_pods %d", len(all_client_attributes), desired_num_pods)
        time.sleep(20)
    graphin = all_client_attributes.get("processor-0")["graph"]
    print("Plotting the graph")
    PlotGraph(graphin)
    validation_graph()

if __name__ == '__main__':
    threading.Thread(target=app.run, kwargs={'host': '0.0.0.0', 'port': 3000, 'threaded': True}).start()
    main()