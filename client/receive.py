import socket
import os
import sys
import time
import select
import errno
from kubernetes import client, config

def get_pod_ip():
    hostname = socket.gethostname()
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

SRV = list(peer_pods.values())[-1]
#SRV = "service1"
PORT = 3000
print(SRV, PORT)

client_socket = socket.socket()
client_socket.connect((SRV, PORT))
client_socket.setblocking(False)

while True:
    message = "10"
    message = message.encode('utf-8')

    while True:
        try:
            client_socket.send(message)
            break
        except IOError as e:
            if e.errno == errno.EAGAIN or e.errno == errno.EWOULDBLOCK:
                # If the operation would block, wait and try again
                time.sleep(0.1)
            else:
                print('Sending error: {}'.format(str(e)))
                sys.exit()

    try:
        # Wait for the socket to become readable
        ready_to_read, _, _ = select.select([client_socket], [], [], 1)
        if ready_to_read:
            data = client_socket.recv(1024).decode('utf-8')
            print(f'{pod_index} > {data}')
    except IOError as e:
        if e.errno != errno.EAGAIN and e.errno != errno.EWOULDBLOCK:
            print('Reading error: {}'.format(str(e)))
            sys.exit()
    except Exception as e:
        print('Reading error: {}'.format(str(e)))
        sys.exit()
