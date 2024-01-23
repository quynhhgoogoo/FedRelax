FROM python:3.9

WORKDIR /app

#COPY requirements.txt .

COPY . /app

# RUN pip install --no-cache-dir -r requirements.txt
RUN pip install numpy scikit-learn matplotlib kubernetes

#COPY . /app

CMD ["python", "fedrelax_pod.py"]