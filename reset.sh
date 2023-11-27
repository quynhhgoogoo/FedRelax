kubectl delete ns fed-relax
kubectl create ns fed-relax
kubectl apply -f service.yaml
kubectl apply -f pods.yaml 
kubectl get pods -n fed-relax