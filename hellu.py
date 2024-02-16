from kubernetes import client, config
import socket
import json
import time

def send_to_pod(pod_index, pod_ip, model_params):
    try:
        with socket.create_connection((pod_ip, peer_port), timeout=60) as client_socket:
            serialized_params = json.dumps(model_params).encode()
            client_socket.sendall(serialized_params)
            print(f"Pod {pod_index} sent data: {serialized_params}")
    except Exception as e:
        print(f"Error sending data from Pod {pod_index}: {e}")

def receive_from_pod(pod_index, pod_ip):
    try:
        with socket.create_connection((pod_ip, peer_port), timeout=60) as client_socket:
            data = client_socket.recv(1024)
            received_params = json.loads(data.decode())
            print(f"Pod {pod_index} received data: {received_params}")
            return received_params
    except Exception as e:
        print(f"Error receiving data in Pod {pod_index}: {e}")
        return None

def get_pod_info(namespace):
    config.load_incluster_config()
    v1 = client.CoreV1Api()

    pod_info = {}
    pods = v1.list_namespaced_pod(namespace=namespace)

    for pod in pods.items:
        pod_info[pod.metadata.name] = pod.status.pod_ip

    return pod_info

def run_bidirectional_communication(pod_index, peer_pods):
    while True:
        # Simulate sending and receiving data between pods
        for peer_pod_index, peer_pod_ip in peer_pods.items():
            if peer_pod_index != pod_index:
                # Send data to the peer pod
                send_to_pod(pod_index, peer_pod_ip, {"message": f"Hello from Pod {pod_index}"})
                
                # Receive data from the peer pod
                received_data = receive_from_pod(pod_index, peer_pod_ip)
                
                # Process the received data (you can customize this part)
                if received_data:
                    print(f"Pod {pod_index} received: {received_data}")

        # Add a delay to simulate a continuous communication loop
        time.sleep(5)

if __name__ == "__main__":
    namespace = 'fed-relax'
    peer_port = 3000
    
    pod_info = get_pod_info(namespace)
    print(pod_info)

    for pod_index, pod_ip in pod_info.items():
        run_bidirectional_communication(pod_index, pod_info)
