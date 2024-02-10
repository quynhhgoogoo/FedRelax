from kubernetes import client, config
import socket
import json
import time
import subprocess

def send_to_pod(pod_index, pod_ip, model_params):
    # Allow pods to send parameters to other pods in the same ns
    try:
        with socket.create_connection((pod_ip, peer_port)) as client_socket:
            serialized_params = json.dumps(model_params).encode()
            client_socket.sendall(serialized_params)
            print(f"Send to pod {pod_index} received: {serialized_params}")
    except Exception as e:
        print(f"Error sending data to pod {pod_index}: {e}")


def receive_from_pod(pod_index, pod_ip):
    # Allow pods to receive parameters from other pods in same ns
    try:
        with socket.create_connection((pod_ip, peer_port)) as client_socket:
            data = client_socket.recv(1024)
            received_params = json.loads(data.decode())
            print(f"Pod {pod_index} received: {received_params}")
            return received_params
    except Exception as e:
        print(f"Error receiving data from pod {pod_index}: {e}")
        return None


def connect_with_retry(pod_ip):
    max_retries = 5
    retry_delay = 5

    for retry in range(max_retries):
        try:
            print(f"Attempting connection to {pod_ip} (Retry {retry + 1}/{max_retries})")
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect((pod_ip, peer_port))
            return client_socket
        except Exception as e:
            print(f"Retry {retry + 1}/{max_retries}: {e}")
            time.sleep(retry_delay)

    raise Exception("Unable to establish connection after multiple retries.")


def is_pod_ready(pod_name):
    v1 = client.CoreV1Api()
    pod_status = v1.read_namespaced_pod_status(name=pod_name, namespace="fed-relax")
    return pod_status.status.phase == "Running" and all(c.ready for c in pod_status.status.container_statuses)

def wait_for_pods_ready():
    v1 = client.CoreV1Api()
    while True:
        ready_pods = [pod.metadata.name for pod in v1.list_namespaced_pod(namespace="fed-relax").items if is_pod_ready(pod.metadata.name)]
        if len(ready_pods) == 2:
            print("All pods are ready.")
            break
        else:
            print(f"Waiting for pods to be ready. Ready pods: {ready_pods}")
            time.sleep(5)

# Get all pod's ip in namespace
def get_pod_ip_addresses():
    config.load_incluster_config()  # Load in-cluster Kubernetes configuration

    v1 = client.CoreV1Api()

    pod_ip_addresses = {}
    pods = v1.list_namespaced_pod(namespace="fed-relax")

    for pod in pods.items:
        pod_ip_addresses[pod.metadata.name] = pod.status.pod_ip

    return pod_ip_addresses

# Get resolved IP addresses
def print_resolved_ips(hostname):
    ip_addresses = socket.gethostbyname_ex(hostname)[-1]
    print(f"Resolved IPs for {hostname}: {ip_addresses}")


print("Debugging: Getting Peer IP address")
peer_ips = get_pod_ip_addresses() 
peer_port = 8000
print("Peers IP addresses", peer_ips)

# Print the default kube-config file path
print(config.KUBE_CONFIG_DEFAULT_LOCATION)

# Try loading kube-config and print the configuration
try:
    config.load_kube_config()
    print(config.list_kube_config_contexts())
except Exception as e:
    print(f"Error loading kube-config: {e}")


# Wait for all pods ready before starting the communication
print("Wait for pods to ready")
wait_for_pods_ready()

# Enable log to confirm context
# config.load_incluster_config()  # Load in-cluster Kubernetes configuration
# current_context = config.list_kube_config_contexts()[1]['context']['cluster']
# print(f"Current context: {current_context}")

# Check DNS resolution within the pod
# dns_output = subprocess.check_output(["cat", "/etc/resolv.conf"])
# print(f"DNS Resolution:\n{dns_output.decode()}")

# for pod_id, pod_ip in peer_ips.items():
#    print(pod_id)
#    print_resolved_ips(pod_id)

# Initialize global model weights as the local model's weights
global_model_weights = 10

# Number of pods
num_pods = 2

# Number of iterations for FedRelax
num_iterations = 10

time.sleep(120)

for iteration in range(num_iterations):
    # Share and receive global model weights from other pods
    for i, pod_id in enumerate(peer_ips):
        print(i, pod_id)
        if is_pod_ready(pod_id):
            client_socket = connect_with_retry(peer_ips[pod_id])
            # Send your global model weights to another pods
            print("Sending data to:", peer_ips[pod_id])
            send_to_pod(i, peer_ips[pod_id], global_model_weights)
            # Receive the global model weights from another pod
            received_weights = receive_from_pod(i, peer_ips[pod_id])
            # Update your global model weights based on received_weights
            global_model_weights = (global_model_weights + received_weights) / 2  # You can adjust this aggregation method
        
        else:
            print(f"Pod {pod_id} is not ready. Skipping.")

    # Update your local model with the global weights
    # local_model.coef_ = global_model_weights
    # local_model.fit(X1, y1)

while True:
    time.sleep(1)