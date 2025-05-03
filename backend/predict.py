import socket
from io import BytesIO
import scipy.io
import torch
from analysis.models import EEGCNN
from features_extract import FeatureExtractor
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(("localhost", 5001))
server.listen(1)

print("Server listening on port 5001...")

conn, addr = server.accept()
buffer = BytesIO()

while True:
    data = conn.recv(1024)  # Receive chunks of data
    if not data:
        break
    buffer.write(data)  # Append received bytes to buffer

buffer.seek(0)  # Reset buffer position

try:
    f = FeatureExtractor([{'coh' : 'alpha'}]) # could get freqs from embdedded byte header?
    mat_data = scipy.io.loadmat(buffer)  # Load directly from memory
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

conn.close()
server.close()