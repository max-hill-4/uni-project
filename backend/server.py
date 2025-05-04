import scipy.io
import torch
from analysis.models import EEGCNN
from features_extract import FeatureExtractor
from flask import Flask, request
from app import compute_saliency_map
import matplotlib.pyplot as plt
import io 
import base64
import json
server = Flask(__name__)

@server.route('/receive', methods=['POST'])
def handle_send_data():
    data = request.files['file']
    feature = request.form.get('feature')
    hormone = request.form.get('hormone')
    feature = json.loads(feature)
    print(feature)
    print(f"Received: {data}")
    hormone_map = {
        "TAC mmol/L": "BDC1",
        "ADA U/L": "BDC1.1",
        "ADA2 U/L": "BDC1.2",
        "%ADA2": "BDC1.3",
        "GLU mg/Dl": "BDC1.4",
        "PHOS mg/Dl": "BDC1.5",
        "CA mg/Dl": "BDC1.6",
        "CHOL mg/Dl": "BDC1.7",
        "TRI mg/Dl": "BDC1.8",
        "HDL mg/dL": "BDC1.9",
        "LDL-C mg/Dl": "BDC1.10",
        "CPK U/L": "BDC1.11",
    }

    try:
        f = FeatureExtractor([dict(feature)]) # could get freqs from embdedded byte header?
        print(data.stream)
        mat_data = scipy.io.loadmat(data.stream)  # Load directly from memory
        m = EEGCNN(num_classes=1)
        print(f'trying to load {hormone_map[hormone]}.pt')
        m.load_state_dict(torch.load(f"trained_models/{[dict(feature)]},{hormone}.pt"))
        m.eval()
        input_tensor = torch.tensor(f.get(mat_data), dtype=torch.float32).unsqueeze(1)
        
        with torch.no_grad():  # Disable gradient computation for inference
            prediction = m(input_tensor)  # Forward pass

        device = torch.device('cpu')
        sm = compute_saliency_map(m, input_tensor, device)
        buffered = io.BytesIO()
        plt.imsave(buffered, sm, cmap="jet", format="png")
        saliency_map_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        response = {'prediction' : str(prediction.item()), 'sm' : f"data:image/png;base64,{saliency_map_base64}"}
    
    except Exception as e:
        print(f"Error loading .mat file: {e}")
        # Process data and respond
        response = f'{e}'
    return response


    
if __name__ == '__main__':
	server.run(debug=True, port=6000)