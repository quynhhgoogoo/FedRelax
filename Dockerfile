FROM python:3.9

WORKDIR /app

#COPY requirements.txt .
COPY . /app

# RUN pip install --no-cache-dir -r requirements.txt
RUN pip install Flask numpy scikit-learn matplotlib kubernetes
RUN apt-get update && apt-get install -y iputils-ping
RUN apt-get update && apt-get install -y net-tools

# Make port 8000 available to the world outside this container
EXPOSE 8000


CMD ["python", "hellu.py"]
#ENTRYPOINT ["python", "-u", "/app/hello.py"]

# Health check
# HEALTHCHECK --interval=5s \
#            --timeout=3s \
#            CMD ["python", "healthcheck.py"]
