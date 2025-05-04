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
    # why the fuck does this even exist ? arent i just fetching the data in results anyways ? IDEKEKEKEKDKEK
    file = request.files["file"]
    hormone = request.form.get('hormone') 
    feature = request.form.get('feature') 
    print(f'feature is of type {type(feature)}')

    if file.filename == "":
        return "No file selected"
    files = {'file' : file.read()}
    data = {'hormone' : hormone, 'feature' : feature}
    response = requests.post('http://localhost:6000/receive', files=files, data=data)
    print(response.text)
    return f"{response.text}"

@app.route("/results")
def show_results():
    return render_template("results.html")

if __name__ == '__main__':
	app.run(debug=True, port=5008)