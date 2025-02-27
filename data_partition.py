import os
import pickle
import numpy as np

def partition_graph_data(graph_file, output_dir):
    # Load the graph object from the pickle file
    with open(graph_file, 'rb') as f:
        G = pickle.load(f)

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate over each node in the graph and save its data to a separate file
    for node in G.nodes():
        node_data = G.nodes[node]
        
        # Extract node features
        nodefeatures = np.array([np.mean(node_data["Xtrain"]), np.mean(node_data["ytrain"])])
        node_data['coords'] = nodefeatures
        node_data["name"] = str(node)
        
        # Save the node data to a file
        node_file = os.path.join(output_dir, f'node_{node}_data.pickle')
        with open(node_file, 'wb') as nf:
            pickle.dump(node_data, nf)
        
        print(f'Successfully saved data for node {node} to {node_file}')

if __name__ == "__main__":
    # Define the path to the input graph file and the output directory
    current_dir = os.path.dirname(os.path.realpath(__file__))

    # Construct the path to the pickle file relative to the current directory
    graph_data = os.path.join(current_dir, 'algorithm', 'data', 'QuizGraphLearning.pickle')

    # Construct the path to the output directory
    partition_dir = os.path.join(current_dir, 'algorithm', 'data', 'input-data-on-laptop')

    # Partition the graph data into separate files
    partition_graph_data(graph_data, partition_dir)
