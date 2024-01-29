from kubernetes import client, config
import socket

def send_to_pod(pod_index, pod_ip, model_params):
    # Allow pods to send parameters to other pods in the same ns
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((pod_ip, peer_port))
    client_socket.send(str(model_params).encode())
    print(f"Send to pod {pod_index} received: {str(model_params).encode()}")


def receive_from_pod(pod_index, pod_ip):
    # Allow pods to receive parameters from other pods in same ns
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((pod_ip, peer_port))
    data = client_socket.recv(1024)
    print(f"Pod {pod_index} received: {data.decode()}")

    return data

# Get all pod's ip in namespace
def get_pod_ip_addresses():
    config.load_incluster_config()  # Load in-cluster Kubernetes configuration

    v1 = client.CoreV1Api()

    pod_ip_addresses = {}
    pods = v1.list_namespaced_pod(namespace="fed-relax")

    for pod in pods.items:
        pod_ip_addresses[pod.metadata.name] = pod.status.pod_ip

    return pod_ip_addresses

peer_ips = get_pod_ip_addresses() 
peer_port = 8000


# Initialize global model weights as the local model's weights
global_model_weights = "test"

# Number of pods
num_pods = 2

# Number of iterations for FedRelax
num_iterations = 10

for iteration in range(num_iterations):
    # Share and receive global model weights from other pods
    for i, pod_id in enumerate(peer_ips):
        # Send your global model weights to another pods
        send_to_pod(i, pod_id, global_model_weights)
        # Receive the global model weights from another pod
        received_weights = receive_from_pod(i, pod_id)
        # Update your global model weights based on received_weights
        global_model_weights = (global_model_weights + received_weights) / 2  # You can adjust this aggregation method

    # Update your local model with the global weights
    local_model.coef_ = global_model_weights
    local_model.fit(X1, y1)