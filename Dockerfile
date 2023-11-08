FROM python:3.8

WORKDIR /app

COPY requirements.txt .

# RUN pip install --no-cache-dir -r requirements.txt
RUN pip install numpy scikit-learn matplotlib kubernetes

COPY . /app

CMD ["python", "fedrelax_pod.py"]