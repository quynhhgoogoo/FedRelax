from kubernetes import client, config
from flask import Flask, request, jsonify
import pickle
import numpy as np
import base64

# Initialize variables to store received predictions and node attributes
all_predictions = []
node_attributes = {}

app = Flask(__name__)

def get_node_attributes(pod_selector="app=fedrelax-client", namespace="fed-relax"):
    config.load_incluster_config()
    v1 = client.CoreV1Api()
    attributes = {}
    pods = v1.list_namespaced_pod(namespace=namespace, label_selector=pod_selector)

    for pod in pods.items:
        # Get the ConfigMap name associated with the pod
        configmap_name = pod.metadata.labels.get("configmap-name")

        if not configmap_name:
            print(f"Warning: Pod {pod.metadata.name} doesn't have a configmap-name label.")
            continue

        # Get the ConfigMap data
        try:
            configmap = v1.read_namespaced_config_map(name=configmap_name, namespace=namespace)
        except client.rest.ApiException as e:
            print(f"Error retrieving ConfigMap {configmap_name} for pod {pod.metadata.name}: {e}")
            continue

        # Extract relevant attributes from the ConfigMap data
        node_attributes[pod.metadata.name] = {
            "coords": pickle.loads(base64.b64decode(configmap.data["coords"])),
        }

    return node_attributes

@app.route("/predictions", methods=["POST"])
def receive_predictions():
    data = request.get_json()
    predictions = np.array(data["predictions"])
