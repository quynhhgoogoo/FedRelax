import socket
import os
import sys
import time
from kubernetes import client, config

counter = 0

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
        # Need to rewrite this. Will be different case if the number of pods are > 2
        SRV = peer_pod_ip

PORT = 3000

while 1:
    if counter != 0:
        time.sleep(5)
    counter += 1
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = (SRV, PORT)
    print("Connection #{}".format(counter))
    print('Connecting to {} port {}'.format(*server_address))
    try:
        sock.connect(server_address)
    except Exception as e:
        print("Cannot connect to the server,", e)
        continue
    try:
        message = b'This is the message. It will be repeated.'
        print('Sending:  {!r}'.format(message))
        sock.sendall(message)
        amount_received = 0
        amount_expected = len(message)
        while amount_received < amount_expected:
            data = sock.recv(64)
            amount_received += len(data)
            print('Received: {!r}'.format(data))
    finally:
        print('Closing socket\n')
        sock.close()