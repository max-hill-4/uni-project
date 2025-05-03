from flask import Flask, render_template, request
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

    # Process the file without saving
    print(file.filename)
    return f"File {file.filename} received successfully!"


if __name__ == '__main__':
	app.run(debug=True)