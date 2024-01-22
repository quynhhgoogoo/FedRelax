FROM python:3.9

WORKDIR /app

#COPY requirements.txt .

COPY fedrelax_pod.py /app/
COPY data/ /app/data/
# Test, will be removed later
COPY hello.py /app/

# RUN pip install --no-cache-dir -r requirements.txt
RUN pip install numpy scikit-learn matplotlib kubernetes

#COPY . /app

CMD ["python", "fedrelax_pod.py"]