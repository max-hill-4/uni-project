from flask import Flask, render_template, request
app = Flask(__name__)
import socketio
import io
sio = socketio.Client()
from werkzeug.datastructures import FileStorage
import base64
import json
sio.connect("http://localhost:5005", transports=["websocket"])

@sio.on("connect")
def on_connect():
    print("Connected to server!")

def send_and_wait_for_response(data):
    
    if not sio.connected:
        print("SocketIO client not connected. Connecting now...")
        sio.connect('http://localhost:5005')  # Use your actual server address

    try:
        response = sio.call("send_data", data, timeout=10)
        print(f"Received response: {response}")
    except socketio.exceptions.TimeoutError:
        print("Timed out waiting for response from server.")
    except socketio.exceptions.BadNamespaceError:
        print("Not connected to the correct namespace.")


@app.route("/")
@app.route("/index")
def index():
	return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    print('upload detected!') 
    if "file" not in request.files:
        return "No file uploaded"
    
    file = request.files["file"]
    
    if file.filename == "":
        return "No file selected"

    data = io.BytesIO(file.read()).getvalue()

    # Encode the file data in base64
    encoded_data = base64.b64encode(data).decode('utf-8')

    send_and_wait_for_response(json.dumps({'yay': 'yay'}))
    return f"File {file.filename} received successfully!"


if __name__ == '__main__':
	app.run(debug=True, port=5008)