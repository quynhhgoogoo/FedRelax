from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/send_data', methods=['POST'])
def send_data():
    try:
        # Receive data from the client
        data = request.json.get('data')

        # Check if the received data is 1
        if data == 1:
            # If the received data is 1, send 2 back to the client
            return jsonify({"message": "Data received. Sending 2 to client.", "data": 2}), 200
        else:
            # If the received data is not 1, send 0 back to the client
            return jsonify({"message": "Invalid data received. Sending 0 to client.", "data": 0}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/receive_data', methods=['POST'])
def receive_data():
    try:
        # Receive data from the client
        data = request.json.get('data')

        # Process the received data (add 1 to it)
        final_response = data + 1

        # Send the processed data back to the client
        return jsonify({"message": "Data processed successfully.", "final_response": final_response}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
