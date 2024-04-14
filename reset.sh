# Restart minikube
# minikube stop
# minikube start

# Amateur clean
# docker system prune -a 
# sudo docker image pull python:3.9  
kubectl delete ns fed-relax
kubectl create ns fed-relax
docker images

# Clean latest image
docker rmi quynhhgoogoo/fed-relax-client
docker rmi quynhhgoogoo/fed-relax-server

# Build the images
# Seperated containers
sudo docker build -t fed-relax-server -f server/Dockerfile .
docker tag fed-relax-server quynhhgoogoo/fed-relax-server:latest
docker push quynhhgoogoo/fed-relax-server:latest

sudo docker build -t fed-relax-client -f client/Dockerfile .
docker tag fed-relax-client quynhhgoogoo/fed-relax-client:latest
docker push quynhhgoogoo/fed-relax-client:latest

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
kubectl apply -f service_account.yaml

# Deployment scripts to simplify the process
kubectl apply -f service.yaml
#kubectl apply -f pods.yaml
kubectl apply -f deployment.yaml 
kubectl get pods -n fed-relax
kubectl config set-context --current --namespace=fed-relax

# Add metric server
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
minikube addons enable metrics-server
# minikube dashboard
# kubectl top node

# kubectl get pod pod1 -n fed-relax -o=jsonpath='{.status.containerStatuses[*].state.terminated.exitCode}'

# Troubleshoot CoreDNS logs
# kubectl get pods -n kube-system -l k8s-app=kube-dns
# kubectl logs -n kube-system -l k8s-app=kube-dns

# Exec to DNS pod
# kubectl config set-context --current --namespace=kube-system
# kubectl debug -it coredns-5d78c9869d-b75zr --image=busybox:1.28 --target=coredns

# kubectl exec -it -n kube-system <COREDNS_POD_NAME> -- /bin/sh
# nslookup fedrelax-deployment-ccc754d4f-vmmqh.fed-relax.svc.cluster.local

# sudo iptables -A INPUT -p tcp --dport <port_number> -j ACCEPT

# Check list of k8s rules
# kubectl api-resources --sort-by name -o wide

# Check configmaps
# kubectl get configmaps node-configmap-0 -o yaml
# Get pods's annotation
# Check if pod is binding to configmap
# kubectl get pod <pod_name> -o yaml | grep configmap
# kubectl get pods -o=jsonpath='{range .items[*]}Pod: {.metadata.name}{"\nAnnotations:\n"}{range .metadata.annotations}{.key}: {.value}{"\n"}{end}{"\n\n"}{end}'

# Download image to local machine
# kubectl cp server-6b7cd85fd6-6tq6b:/app/init_graph.png ./output/init_graph.png