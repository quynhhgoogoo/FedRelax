from flask import Flask, jsonify
from kubernetes import client, config
import socket
import time
import subprocess
from threading import Thread

#while True:
#    print("Hello")
#    time.sleep(1)

app = Flask(__name__)

print("hello there!")

# Flag to indicate the health status
healthy = True

@app.route('/')
def hello():
    return 'Hello World!'

@app.route('/health')
def health_check():
    global healthy
    # Perform health check logic here
    if healthy:
        return jsonify(status='OK', details='Application is healthy'), 200
    else:
        return jsonify(status='ERROR', details='Application is not healthy'), 500

def run_flask_app():
    app.run(host='0.0.0.0', port=8000)

if __name__ == '__main__':
    # Your existing initialization code

    # Start Flask app in a separate thread
    flask_thread = Thread(target=run_flask_app)
    flask_thread.start()

    # Your existing main loop
    while True:
        # Your existing logic for communication with other pods

        # Example health check condition, modify as needed
        if some_condition:
            healthy = True
        else:
            healthy = False

        # Adjust sleep time based on your requirements
        time.sleep(1)
