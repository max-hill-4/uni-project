import scipy.io
import torch
from analysis.models import EEGCNN
from features_extract import FeatureExtractor

from flask import Flask
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

@socketio.on("send_data")
def handle_send_data(data):
    print(f"Received: {data}")
    
    try:
        f = FeatureExtractor([{'coh' : 'alpha'}]) # could get freqs from embdedded byte header?
        mat_data = scipy.io.loadmat(data)  # Load directly from memory
        m = EEGCNN(num_classes=1)
        m.load_state_dict(torch.load("model.pt"))
        m.eval()
        input_tensor = torch.tensor(f.get(mat_data), dtype=torch.float32)
        input_tensor = input_tensor.unsqueeze(1)
        print(input_tensor.shape)
        
        print("got model, predicting")
        with torch.no_grad():  # Disable gradient computation for inference
            prediction = m(input_tensor)  # Forward pass
        print(f"Predicted value: {prediction.item()}")
    except Exception as e:
        print(f"Error loading .mat file: {e}")
        # Process data and respond
        response = f'{e}'
    return response  

if __name__ == '__main__':
	app.run(debug=True, port=5005)