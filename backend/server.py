import scipy.io
import torch
from analysis.models import EEGCNN
from features_extract import FeatureExtractor
from flask import Flask, request
import data_io
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
    print(f'my feature before jsonloda is {feature}')
    feature = json.loads(f'[{feature}]') if feature else []
    print(f'type of feature is {type(feature)}')
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
        f = FeatureExtractor(feature) # could get freqs from embdedded byte header?
        print(data.stream)
        mat_data = scipy.io.loadmat(data.stream)  # Load directly from memory
        print('loadedmat')
        m = EEGCNN()
        print(f'trying to load {hormone_map[hormone]}.pt')
        m.load_state_dict(torch.load(f"/models/{feature} + ['{hormone_map[hormone]}'].pth"))
        m.eval()
        input_tensor = torch.tensor(f.get(mat_data), dtype=torch.float32).unsqueeze(1)
        print(f"creating tensor of {feature}")
        with torch.no_grad():  # Disable gradient computation for inference
            prediction = m(input_tensor)  # Forward pass
        prediction = torch.argmax(prediction, dim=1)
        sm, saliency = data_io.saliencymap.compute_saliency_map(m, input_tensor)
        print('sm computed') 
        topmap = data_io.saliencymap.compute_topo_map(saliency)
        
        print('tm computed')

        sm_buffer = io.BytesIO()
        tm_buffer = io.BytesIO() 
        
        topmap.savefig(tm_buffer, format="png")
        sm.savefig(sm_buffer, format="png")
        print('buffers loaded')
        sm= base64.b64encode(sm_buffer.getvalue()).decode('utf-8')
        topmap = base64.b64encode(tm_buffer.getvalue()).decode('utf-8')
        print(prediction)
        response = {'prediction' : str(prediction.item()), 'sm' : f"data:image/png;base64,{sm}", 'tm' : f"data:image/png;base64,{topmap}"}
    
    except Exception as e:
        print(f"Error loading .mat file: {e}")
        # Process data and respond
        response = f'{e}'
    return response


    
if __name__ == '__main__':
	server.run(debug=True, port=8888)