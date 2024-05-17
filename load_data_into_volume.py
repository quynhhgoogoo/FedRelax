import os
import pickle

# Get the current working directory
current_dir = os.path.dirname(os.path.realpath(__file__))

# Construct the path to the pickle file relative to the current directory
pickle_file_path = os.path.join(current_dir, 'algorithm', 'data', 'QuizGraphLearning.pickle')

# Load the pickle file
with open(pickle_file_path, 'rb') as f:
    G = pickle.load(f)

# Iterate through each node in the graph
for node_index, node_data in enumerate(G.nodes()):
    # Determine the volume name
    volume_name = f"volume{node_index + 1}"
    
    # Load data associated with the node
    node_data = G.nodes[node_data]
    
    # Assuming data is stored in a variable named 'data'
    data = {
        "Xtrain": node_data["Xtrain"],
        "ytrain": node_data["ytrain"],
        "Xval": node_data["Xval"],
        "yval": node_data["yval"]
    }
    
    # Write data to the corresponding Docker volume
    volume_path = f'/var/lib/docker/volumes/{volume_name}/_data'
    os.makedirs(volume_path, exist_ok=True)
    
    with open(os.path.join(volume_path, 'data.pickle'), 'wb') as f:
        pickle.dump(data, f)
