from kubernetes import client, config

config.load_incluster_config()
v1 = client.CoreV1Api()

namespace = "fed-relax"
pods = v1.list_namespaced_pod(namespace=namespace)

for pod in pods.items:
    print(f"Pod Name: {pod.metadata.name}, Labels: {pod.metadata.labels}")
    