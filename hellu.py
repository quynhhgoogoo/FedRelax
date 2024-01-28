from ipaddress import ip_address
from kubernetes import client, config

# Get all pod's ip in namespace
def get_pod_ip_addresses():
    config.load_kube_config()  # Load your Kubernetes configuration (e.g., ~/.kube/config)

    v1 = client.CoreV1Api()

    pod_ip_addresses = {}
    pods = v1.list_namespaced_pod(namespace="fed-relax")

    for pod in pods.items:
        pod_ip_addresses[pod.metadata.name] = pod.status.pod_ip

    return pod_ip_addresses

ip_addresses =  get_pod_ip_addresses()
print(ip_addresses)