import scipy.io
import torch
from analysis.models import EEGCNN
from werkzeug.datastructures import FileStorage
from features_extract import FeatureExtractor
from flask import Flask, request
from app import compute_saliency_map
import matplotlib.pyplot as plt
import io 
app = Flask(__name__)

@app.route('/receive', methods=['POST'])
def handle_send_data():
    data = request.files['file']
    print(f"Received: {data}")
    
    try:
        f = FeatureExtractor([{'coh' : 'alpha'}]) # could get freqs from embdedded byte header?
        mat_data = scipy.io.loadmat(data.stream)  # Load directly from memory
        m = EEGCNN(num_classes=1)
        m.load_state_dict(torch.load("model.pt"))
        m.eval()
        input_tensor = torch.tensor(f.get(mat_data), dtype=torch.float32).unsqueeze(1)
        
        with torch.no_grad():  # Disable gradient computation for inference
            prediction = m(input_tensor)  # Forward pass


        response = {'prediction' : str(prediction.item())}
    

    except Exception as e:
        print(f"Error loading .mat file: {e}")
        # Process data and respond
        response = f'{e}'
    return response



@app.route('/sm', methods=['POST'])
def get_sm():
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Defined here
    # input_tensor = input_tensor.to(device) 
    # sm = compute_saliency_map(m, input_tensor, device)
    # buffered = io.BytesIO()
    # plt.imsave(buffered, sm, cmap="jet")
    # buffered.seek(-1)
    # sm_fs = FileStorage(stream=buffered, filename="saliency_map.png")
    return 0
    
    
if __name__ == '__main__':
	app.run(debug=True, port=5005)