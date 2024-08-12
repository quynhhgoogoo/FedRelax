import base64
from flask import Flask, request, jsonify, Response
import logging
import matplotlib.pyplot as plt
import numpy as np
import pickle
import threading

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Initialize empty dictionary to store client attributes for model evaluation
all_client_attributes = {}
attributes_lock = threading.Lock()
desired_num_pods = 10

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
    plt.plot(all_client_attributes.get("processor-8")["Xval"], all_client_attributes.get("processor-2")["yval"], color="blue", label="validation data cluster 0", linewidth=2)
    plt.plot(all_client_attributes.get("processor-15")["Xval"], all_client_attributes.get("processor-11")["yval"], color="red", label="val data second cluster", linewidth=2)
    plt.savefig('/app/validation.png')
    print(f"Validation graph is successfully saved in /app/validation.png")

def PlotGraph(graphin, output_path="/app/final.png", annotate="name"):     
    # the numpy array x will hold the horizontal coord of markers for each node in emp. graph graphin
    x = np.zeros(len(desired_num_pods))
    # vertical coords of markers 
    y = np.zeros(len(desired_num_pods))
    
    iter_node = 0
    for pod, attributes in all_client_attributes.items():
        x[iter_node] = attributes['coords']
        y[iter_node] = attributes['coords']
        iter_node += 1

    # standareize the coordinates of node markers 
    x = (x - np.min(x, axis=0))/np.std(x, axis=0) + 1 
    y = (y - np.min(y, axis=0))/np.std(y, axis=0) + 1 

    # create a figure with prescribed dimensions 
    fig, ax = plt.subplots(figsize=(10,10))
    
    # generate a scatter plot with each marker representing a node in graphin
    ax.scatter(x, y, 300, marker='o', color='Black')
    
    # draw links between two nodes if they are connected by an edge 
    # in the empirical graph. use the "weight" of the edge to determine the line thickness
    for edge_dmy in graphin.edges:
        ax.plot([x[edge_dmy[0]],x[edge_dmy[1]]],[y[edge_dmy[0]],y[edge_dmy[1]]],c='black',lw=4*graphin.edges[edge_dmy]["weight"])

    # annotate each marker by the node attribute whose name is stored in the input parameter "annotate"
    for iter_node in graphin.nodes : 
        ax.annotate(str(graphin.nodes[iter_node][annotate]),(x[iter_node]+0.2, 0.995*y[iter_node]),c="red" )
    ax.set_ylim(0.9*np.min(y),1.1*np.max(y))
    ax.set_xlim(0.9*np.min(x),1.1*np.max(x))

    fig.savefig(output_path, format='png') 
    plt.close(fig)


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
    graphin = all_client_attributes.get("processor-0")["model"]
    PlotGraph(graphin)
    validation_graph()

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    app.run(host='0.0.0.0', port=3000, threaded=True)
    main()