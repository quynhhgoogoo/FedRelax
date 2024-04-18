FROM python:3.9

WORKDIR /app

#COPY requirements.txt .
COPY . /app

# RUN pip install --no-cache-dir -r requirements.txt
RUN pip install Flask numpy scikit-learn matplotlib kubernetes networkx Flask
RUN apt-get update && apt-get install -y iputils-ping \ 
    && apt-get install -y net-tools \
    && apt-get install -y dnsutils  \
    && apt-get install -y vim \
    && apt-get install -y iptables && rm -rf /var/lib/apt/lists/*


CMD ["python", "/client/test.py"]
