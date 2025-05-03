from flask import Flask, render_template, request
app = Flask(__name__)
import socket
import io

def send_file_over_socket(mat_file):
    HOST = "localhost"  # Change to your destination server
    PORT = 5001  # Your socket server port

    # Open socket connection
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))

        # Convert .mat file content into byte stream
        file_bytes = io.BytesIO(mat_file.read()).getvalue()
        
        # Send data
        s.sendall(file_bytes)
        print("File sent successfully!")


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
    print(file.filename)
    send_file_over_socket(file)
    return f"File {file.filename} received successfully!"


if __name__ == '__main__':
	app.run(debug=True)