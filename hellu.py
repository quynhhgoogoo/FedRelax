from kubernetes import client, config
import socket
import json
import time
import threading
from queue import Queue


def get_pod_info(namespace):
    config.load_incluster_config()
    v1 = client.CoreV1Api()

    pod_info = {}
    pods = v1.list_namespaced_pod(namespace=namespace)

    for pod in pods.items:
        pod_info[pod.metadata.name] = pod.status.pod_ip

    return pod_info

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

def send_to_pod(pod_ip, model_params, send_queue):
    print(f"Sending data to: {pod_ip}")
    try:
        with socket.create_connection((pod_ip, peer_port), timeout=60) as client_socket:
            serialized_params = json.dumps(model_params).encode()
            client_socket.sendall(serialized_params)
            print(f"Sent data: {serialized_params}")
    except Exception as e:
        print(f"Error sending data: {e}")
    finally:
        # Signal the completion of the send operation
        send_queue.put(None)

def receive_from_pod(pod_ip, receive_queue):
    try:
        with socket.create_connection((pod_ip, peer_port), timeout=60) as client_socket:
            data = client_socket.recv(1024)
            received_params = json.loads(data.decode())
            print(f"Received data: {received_params}")
            return received_params
    except Exception as e:
        print(f"Error receiving data: {e}")
        return None
    finally:
        # Signal the completion of the receive operation
        receive_queue.put(None)


def run_bidirectional_communication(pod_index, peer_pods):
    while True:
        send_receive_threads = []

        # Use queues to signal completion of send and receive operations
        send_queue = Queue()
        receive_queue = Queue()

        # Simulate sending and receiving data between pods
        for peer_pod_index, peer_pod_ip in peer_pods.items():
            if peer_pod_index != pod_index:
                # Create threads for sending and receiving
                send_thread = threading.Thread(target=send_to_pod, args=(peer_pod_ip, {"message": f"Hello from Pod {pod_index}"}, send_queue))
                receive_thread = threading.Thread(target=receive_from_pod, args=(peer_pod_ip, receive_queue))

                send_receive_threads.extend([send_thread, receive_thread])

                # Start the threads
                send_thread.start()
                receive_thread.start()

        # Wait for all send threads to complete
        for send_thread in send_receive_threads:
            if send_thread.is_alive():
                send_thread.join()

        # Wait for all receive threads to complete
        for receive_thread in send_receive_threads:
            if receive_thread.is_alive():
                receive_thread.join()

        # Process the received data 
        for _ in range(len(peer_pods) - 1):
            received_data = receive_queue.get()
            if received_data:
                print(f"Pod {pod_index} received: {received_data}")

        # Add a delay to simulate a continuous communication loop
        time.sleep(5)

if __name__ == "__main__":
    namespace = 'fed-relax'
    peer_port = 3000
    
    # wait_for_pods_ready()

    pod_info = get_pod_info(namespace)
    print(pod_info)

    for pod_index, pod_ip in pod_info.items():
        run_bidirectional_communication(pod_index, pod_info)
