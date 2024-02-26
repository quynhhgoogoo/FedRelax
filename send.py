import socket
import os
import sys
import time
import select
import errno
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

SRV = list(peer_pods.values())[0]
PORT = 3000

client_socket = socket.socket()
client_socket.connect((SRV, PORT))
client_socket.setblocking(False)

while True:
    message = message.encode('utf-8')
    client_socket.send( message)
    try:
        while True:
            message = client_socket.recv(1024).decode('utf-8')
            print(f'{pod_index} > {message}')
    except IOError as e:
        if e.errno != errno.EAGAIN and e.errno != errno.EWOULDBLOCK:
            print('Reading error: {}'.format(str(e)))
            sys.exit()
        continue
    except Exception as e:
        print('Reading error: '.format(str(e)))
        sys.exit()
   