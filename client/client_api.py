import requests

# URL of the server endpoint
SERVER_URL = "http://server-service:3000"

# Send 1 to the server
response = requests.post(f"{SERVER_URL}/send_data", json={"data": 1})

# Print the response received from the server
print("Response from server:", response.text)

# Check if the response contains JSON data
if response.headers.get('content-type') == 'application/json':
    try:
        # Try to parse the JSON response
        response_json = response.json()
        # Print the message from the server
        print("Final response from server:", response_json.get("message"))
    except:
        # If parsing fails, print an error message
        print("Error: Failed to parse JSON response")
else:
    # If the response does not contain JSON data, print a warning
    print("Warning: Response does not contain JSON data")
