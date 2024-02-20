import socket
import sys
import os
from kubernetes import client, config

PORT = 3000

def get_pod_ip():
    # Get the hostname of the pod
    hostname = socket.gethostname()
    # Get the IP address corresponding to the hostname
    ip_address = socket.gethostbyname(hostname)
    return ip_address

def get_pod_info(namespace):
    config.load_incluster_config()
    v1 = client.CoreV1Api()
    pod_info = {}
    pods = v1.list_namespaced_pod(namespace=namespace)
    for pod in pods.items:
        pod_info[pod.metadata.name] = pod.status.pod_ip
    return pod_info

pod_index = get_pod_ip()
peer_pods = get_pod_info("fed-relax")

for peer_pod_index, peer_pod_ip in peer_pods.items():
    if peer_pod_ip != pod_index:
        print(peer_pod_ip, pod_index)
        # Need to rewrite this. Will be different case if the number of pods are > 2
        SRV = peer_pod_ip


sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = (SRV, PORT)
print('Starting up on {} port {}'.format(*server_address))
sock.bind(server_address)
sock.listen()

while True:
    print('\nWaiting for a connection')
    connection, client_address = sock.accept()
    try:
        print('Connection from', client_address)
        while True:
            data = connection.recv(64)
            print('Received {!r}'.format(data))
            if data:
                print('Sending data back to the client')
                connection.sendall(data)
            else:
                print('No data from', client_address)
                break
    finally:
        connection.close()
