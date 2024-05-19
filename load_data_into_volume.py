import os
import pickle
import json

# Get the current working directory
current_dir = os.path.dirname(os.path.realpath(__file__))

# Construct the path to the pickle file relative to the current directory
pickle_file_path = os.path.join(current_dir, 'algorithm', 'data', 'QuizGraphLearning.pickle')

# Load the pickle file
with open(pickle_file_path, 'rb') as f:
    G = pickle.load(f)

# Iterate through each node in the graph
for node_index, node_data in enumerate(G.nodes()):
    # Determine the volume name, matching the Docker volume naming convention
    volume_name = f"thesis-code_volume{node_index + 1}"
    
    # Load data associated with the node
    node_data = G.nodes[node_data]
    
    # Assuming data is stored in a variable named 'data'
    data = {
        "Xtrain": node_data["Xtrain"].tolist(),  # Convert numpy arrays to lists
        "ytrain": node_data["ytrain"].tolist(),
        "Xval": node_data["Xval"].tolist(),
        "yval": node_data["yval"].tolist()
    }
    
    # Write data to the corresponding Docker volume
    volume_path = f'/var/lib/docker/volumes/{volume_name}/_data'
    os.makedirs(volume_path, exist_ok=True)
    
    # Serialize and save the data to a file within the volume
    data_path = os.path.join(volume_path, 'data.json')
    with open(data_path, 'w') as f:
        json.dump(data, f)
    
    print(f"Data is loaded successfully to {volume_name}")
