import os
import subprocess

NUM_REPLICAS = 5

def create_docker_volumes(num_replicas):
    for i in range(1, num_replicas + 1):
        volume_name = f"volume{i}"
        subprocess.run(["docker", "volume", "create", volume_name])

def generate_kubernetes_manifests(num_replicas):
    manifests = []
    for i in range(1, num_replicas + 1):
        pod_name = f"client{i}"
        volume_name = f"volume{i}"
        volume_path = f"/var/lib/docker/volumes/{volume_name}/_data"
        
        manifest = f"""
apiVersion: v1
kind: Pod
metadata:
  name: {pod_name}
spec:
  containers:
  - name: {pod_name}-container
    image: quynhhgoogoo/fed-relax-client:latest
    volumeMounts:
    - mountPath: /app/data
      name: {volume_name}
  volumes:
  - name: {volume_name}
    hostPath:
      path: {volume_path}
      type: Directory
"""
        manifests.append(manifest)
    
    return manifests

def write_manifests_to_files(manifests):
    for i, manifest in enumerate(manifests):
        file_name = f"client{i+1}-pod.yaml"
        with open(file_name, 'w') as f:
            f.write(manifest)

def apply_manifests(num_replicas):
    for i in range(1, num_replicas + 1):
        file_name = f"client{i}-pod.yaml"
        subprocess.run(["kubectl", "apply", "-f", file_name])

def main():
    create_docker_volumes(NUM_REPLICAS)
    manifests = generate_kubernetes_manifests(NUM_REPLICAS)
    write_manifests_to_files(manifests)
    apply_manifests(NUM_REPLICAS)

if __name__ == "__main__":
    main()
