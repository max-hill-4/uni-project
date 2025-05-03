from flask import Flask, render_template, request
from werkzeug.datastructures import FileStorage
import requests

app = Flask(__name__)

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

    response = requests.post('http://localhost:5005/receive', files={'file' : file.read()})
    print(response.text)
    return f"File {response.text} received successfully!"


if __name__ == '__main__':
	app.run(debug=True, port=5008)