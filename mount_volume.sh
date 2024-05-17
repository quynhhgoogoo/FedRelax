#!/bin/bash

# TODO: Replace with the real number of replicas/pods
NUM_REPLICAS=5

# Generate Docker Compose file dynamically
echo "version: '3'" > docker-compose.yml
echo "services:" >> docker-compose.yml
for ((i=1; i<=$NUM_REPLICAS; i++)); do
    echo "  client$i:" >> docker-compose.yml
    echo "    image: quynhhgoogoo/fed-relax-client:latest" >> docker-compose.yml
    echo "    volumes:" >> docker-compose.yml
    echo "      - volume$i:/app/data" >> docker-compose.yml
done

echo "volumes:" >> docker-compose.yml
for ((i=1; i<=$NUM_REPLICAS; i++)); do
    echo "  volume$i:" >> docker-compose.yml
    echo "    driver: local" >> docker-compose.yml
done

# Run Docker Compose
docker-compose up -d
