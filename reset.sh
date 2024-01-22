# Amateur clean
kubectl delete ns fed-relax
kubectl create ns fed-relax

# Build the images
docker build -t fed-relax .
docker tag fed-relax quynhhgoogoo/fed-relax
docker push quynhhgoogoo/fed-relax

# Provide credentials
#read -p 'Username: ' uservar
#read -sp 'Password: ' passvar
#kubectl create secret docker-registry regcred \
#    --docker-server='https://hub.docker.com'\
#    --docker-username= $uservar\
#    --docker-password= $passvar\
#    --docker-email='luongdiemquynh1998@gmail.com'

# Deployment scripts to simplify the process
kubectl apply -f service.yaml
kubectl apply -f pods.yaml 
kubectl get pods -n fed-relax
kubectl config set-context --current --namespace=fed-relax
