# Restart minikube
# minikube stop
# minikube start

# Amateur clean
kubectl delete ns fed-relax
kubectl create ns fed-relax
docker images

# Clean latest image
docker rmi quynhhgoogoo/fed-relax

# Build the images
sudo docker build -t fed-relax .
docker tag fed-relax quynhhgoogoo/fed-relax:latest
docker push quynhhgoogoo/fed-relax:latest
docker run --rm -it quynhhgoogoo/fed-relax:latest ls /app

# Provide credentials
#read -p 'Username: ' uservar
#read -sp 'Password: ' passvar
#kubectl create secret docker-registry regcred \
#    --docker-server='https://hub.docker.com'\
#    --docker-username= $uservar\
#    --docker-password= $passvar\
#    --docker-email='luongdiemquynh1998@gmail.com'

# Apply rbac with necessary permissions
kubectl apply -f role.yaml
kubectl apply -f role_binding.yaml

# Deployment scripts to simplify the process
kubectl apply -f service.yaml
#kubectl apply -f pods.yaml
kubectl apply -f deployment.yaml 
kubectl get pods -n fed-relax
kubectl config set-context --current --namespace=fed-relax

# kubectl get pod pod1 -n fed-relax -o=jsonpath='{.status.containerStatuses[*].state.terminated.exitCode}'
# sudo docker build -t fed-relax .
# docker tag fed-relax quynhhgoogoo/fed-relax:latest
# docker push quynhhgoogoo/fed-relax:latest
# kubectl apply -f deployment.yaml

# Troubleshoot CoreDNS logs
# kubectl get pods -n kube-system -l k8s-app=kube-dns
# kubectl logs -n kube-system -l k8s-app=kube-dns

# Exec to DNS pod
# kubectl config set-context --current --namespace=kube-system
# kubectl debug -it coredns-5d78c9869d-b75zr --image=busybox:1.28 --target=coredns

# kubectl exec -it -n kube-system <COREDNS_POD_NAME> -- /bin/sh
# nslookup fedrelax-deployment-ccc754d4f-vmmqh.fed-relax.svc.cluster.local


