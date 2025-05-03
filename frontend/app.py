from flask import Flask, render_template, request
app = Flask(__name__)
import socketio
import io
sio = socketio.Client()

@sio.on("connect")
def on_connect():
    print("Connected to server!")

def send_and_wait_for_response(data):
    
    sio.connect("http://localhost:5005")
    response = sio.call("send_data", data)  # Sends and waits for reply
    print(f"Received response: {response}")


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

    # Process the file without saving
    file_bytes = io.BytesIO(file.read()).getvalue()  # Convert FileStorage to bytes
    print(file.filename)
    send_and_wait_for_response(file_bytes)
    return f"File {file.filename} received successfully!"


if __name__ == '__main__':
	app.run(debug=True, port=5008)