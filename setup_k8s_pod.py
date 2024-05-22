import os
import subprocess

def generate_pv_pvc_yaml(volume_index):
    storage_class_name = "standard"
    return f"""
apiVersion: v1
kind: PersistentVolume
metadata:
  name: pv-volume{volume_index}
spec:
  capacity:
    storage: 1Gi
  accessModes:
    - ReadWriteOnce
  hostPath:
    path: "/var/lib/docker/volumes/volume{volume_index}/_data"
  storageClassName: {storage_class_name}
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: pvc-volume{volume_index}
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 1Gi
  volumeName: pv-volume{volume_index}
  storageClassName: {storage_class_name}
"""

def generate_pod_yaml(volume_index):
    return f"""
apiVersion: v1
kind: Pod
metadata:
  name: client{volume_index}
spec:
  containers:
  - name: fedrelax-container-client
    image: quynhhgoogoo/fed-relax-client:latest
    volumeMounts:
    - mountPath: /app/data
      name: volume{volume_index}
  volumes:
  - name: volume{volume_index}
    persistentVolumeClaim:
      claimName: pvc-volume{volume_index}
"""

def main():
    num_volumes = 5  # Adjust based on the number of volumes needed

    for i in range(1, num_volumes + 1):
        # Generate PV and PVC YAML
        pv_pvc_yaml = generate_pv_pvc_yaml(i)
        pv_pvc_filename = f"pv-pvc-{i}.yaml"
        with open(pv_pvc_filename, "w") as f:
            f.write(pv_pvc_yaml)
        
        # Apply PV and PVC YAML
        subprocess.run(["kubectl", "apply", "-f", pv_pvc_filename])

        # Generate Pod YAML
        pod_yaml = generate_pod_yaml(i)
        pod_filename = f"client{i}-pod.yaml"
        with open(pod_filename, "w") as f:
            f.write(pod_yaml)

        # Apply Pod YAML
        subprocess.run(["kubectl", "apply", "-f", pod_filename])

if __name__ == "__main__":
    main()
