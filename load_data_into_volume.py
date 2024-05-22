import os
import pickle
import json
import numpy as np
import subprocess

def load_and_process_graph(pickle_file_path, num_replicas):
    with open(pickle_file_path, 'rb') as f:
        G = pickle.load(f)

    for iter_node in G.nodes():
        nodefeatures = np.array([np.mean(G.nodes[iter_node]["Xtrain"]), np.mean(G.nodes[iter_node]["ytrain"])])
        G.nodes[iter_node]['coords'] = nodefeatures
        G.nodes[iter_node]["name"] = str(iter_node)

    return G

def save_node_data_to_volume(G, num_replicas):
    for node_index, node_data in enumerate(G.nodes(data=True)):
        if node_index >= num_replicas:
            break

        volume_name = f"volume{node_index + 1}"
        volume_path = f'/var/lib/docker/volumes/{volume_name}/_data'
        os.makedirs(volume_path, exist_ok=True)
        
        data = {
            "Xtrain": node_data[1]["Xtrain"].tolist(),
            "ytrain": node_data[1]["ytrain"].tolist(),
            "Xval": node_data[1]["Xval"].tolist(),
            "yval": node_data[1]["yval"].tolist(),
            "coords": node_data[1]["coords"].tolist(),
            "name": node_data[1]["name"]
        }

        data_path = os.path.join(volume_path, 'data.json')
        with open(data_path, 'w') as f:
            json.dump(data, f)

        print(f"Data is loaded successfully to {volume_name}")

def create_docker_volumes(num_replicas):
    for i in range(1, num_replicas + 1):
        volume_name = f"volume{i}"
        subprocess.run(["docker", "volume", "create", volume_name])

def generate_docker_compose(num_replicas):
    services = {}
    volumes = {}
    
    for i in range(1, num_replicas + 1):
        service_name = f"client{i}"
        volume_name = f"volume{i}"
        
        services[service_name] = {
            "image": "quynhhgoogoo/fed-relax-client:latest",
            "volumes": [f"{volume_name}:/app/data"]
        }
        
        volumes[volume_name] = {
            "driver": "local"
        }
    
    docker_compose = {
        "version": "3",
        "services": services,
        "volumes": volumes
    }
    
    with open('docker-compose.yaml', 'w') as f:
        json.dump(docker_compose, f, indent=2)
    
    print("docker-compose.yaml is generated successfully")

def main():
    current_dir = os.path.dirname(os.path.realpath(__file__))

    # Construct the path to the pickle file relative to the current directory
    pickle_file_path = os.path.join(current_dir, 'algorithm', 'data', 'QuizGraphLearning.pickle')
    num_replicas = 5
    
    # Load and process the graph
    G = load_and_process_graph(pickle_file_path, num_replicas)
    
    # Create Docker volumes
    create_docker_volumes(num_replicas)
    
    # Save node data to Docker volumes
    save_node_data_to_volume(G, num_replicas)
    
    # Generate docker-compose.yaml
    generate_docker_compose(num_replicas)

if __name__ == "__main__":
    main()
