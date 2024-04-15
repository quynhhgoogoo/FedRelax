from kubernetes import client, config
import pickle
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import requests

def get_pod_name():
    config.load_incluster_config()
    v1 = client.CoreV1Api()
    pod_name = v1.read_namespaced_pod(name="", namespace="fed-relax").metadata.name
    return pod_name

def get_configmap_data(pod_name, configmap_name, namespace="fed-relax"):
    v1 = client.CoreV1Api()
    try:
        configmap = v1.read_namespaced_config_map(name=configmap_name, namespace=namespace)
    except client.rest.ApiException as e:
        print(f"Error retrieving ConfigMap {configmap_name} for pod {pod_name}: {e}")
        return None
    return {key: pickle.loads(base64.b64decode(value)) for key, value in configmap.data.items()}

def train_local_model(Xtrain, ytrain, max_depth=4):
    model = DecisionTreeRegressor(max_depth=max_depth)
    model.fit(Xtrain, ytrain)
    return model

def send_predictions_to_server(predictions, service_name="fedrelax-server", namespace="fed-relax"):
    # Use service discovery to find the server pod IP
    v1 = client.AppsV1Api()
    try:
        service = v1.read_namespaced_service(name=service_name, namespace=namespace)
        server_ip = service.spec.cluster_ip  # Get cluster IP of the service
    except client.rest.ApiException as e:
        print(f"Error retrieving service {service_name}: {e}")
        return

    # Send predictions to the server using a library like requests for a REST API
    url = f"http://{server_ip}:3000/predictions"  
    data = {"predictions": predictions.tolist()} 
    response = requests.post(url, json=data)

    if response.status_code == 200:
        print("Predictions sent successfully!")
    else:
        print(f"Error sending predictions: {response.text}")

# Load data and attributes from ConfigMap
pod_name = get_pod_name()
configmap_name = f"node-configmap-{pod_name.split('-')[-1]}"  # Extract node ID from pod name
configmap_data = get_configmap_data(pod_name, configmap_name)

if configmap_data:
    Xtrain = configmap_data["Xtrain"]
    ytrain = configmap_data["ytrain"]

    # Train local model
    local_model = train_local_model(Xtrain, ytrain)

    # Generate predictions on a test set (replace with actual test set generation)
    Xtest = np.arange(0.0, 1, 0.1).reshape(-1, 1)
    predictions = local_model.predict(Xtest)

    # Send predictions to the server for aggregation
    send_predictions_to_server(predictions)
else:
    print("Error: ConfigMap data not found.")
