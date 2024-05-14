import concurrent.futures
from kubernetes import client, config
import subprocess
import signal
import sys

config.load_kube_config()

def execute_command_in_pod(pod_name, namespace, command):
    exec_command = [
        "/bin/sh",
        "-c",
        f"kubectl exec -n {namespace} {pod_name} -- {command}"
    ]
    return subprocess.run(exec_command, capture_output=True, text=True)

# Get the list of pod names
v1 = client.CoreV1Api()
namespace = "fed-relax"
label_selector = "app=fedrelax-client"
pod_list = v1.list_namespaced_pod(namespace, label_selector=label_selector)
pod_names = [pod.metadata.name for pod in pod_list.items]

# Command to execute in each pod
command = "python3 client/client.py"

# Function to handle termination and terminate pod executions
def handle_termination(signum, frame):
    print("Terminating script execution...")
    sys.exit(1)

# Register signal handler for termination
signal.signal(signal.SIGINT, handle_termination)

# Execute the command in each pod in parallel
with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = [executor.submit(execute_command_in_pod, pod_name, namespace, command) for pod_name in pod_names]
    try:
        for future, pod_name in zip(concurrent.futures.as_completed(futures), pod_names):
            result = future.result()
            if result.returncode == 0:
                print(f"Command executed successfully in pod {pod_name}")
            else:
                print(f"Error executing command in pod {pod_name}: {result.stderr}")
    except KeyboardInterrupt:
        print("Keyboard interrupt received. Terminating pod executions...")
        for pod_name in pod_names:
            subprocess.run(["kubectl", "delete", "pod", pod_name, "-n", namespace])
