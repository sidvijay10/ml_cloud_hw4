# infer_mnist.py
from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io
from flask_cors import CORS

# Define the Flask application
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Define the Classifier class (as defined in your training script)
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # Your model architecture (same as the training)
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        # Flatten the image
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1)
        return x

# Initialize the model and load the saved weights
model = Classifier()
model_path = '/mnt/mnist_model/mnist.pth'  # Ensure this is the path where your model is saved
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Define a transform to normalize the data for inference
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # MNIST is grayscale
    transforms.Resize((28, 28)),  # Resize the image to 28x28 as expected by the model
    transforms.ToTensor(),  # Convert the image to a torch tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize the image
])

from flask import Flask, request, jsonify, render_template, send_from_directory
import os

app = Flask(__name__, static_folder='static', template_folder='templates')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('static', path)


# Define the route for inference
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided.'}), 400
        image_file = request.files['image']
        
        # Convert the image file to a PIL image
        img = Image.open(io.BytesIO(image_file.read())).convert('L')  # Convert to grayscale
        img = transform(img)  # Apply the transformation
        img = img.unsqueeze(0)  # Add a batch dimension
        
        # Perform inference
        with torch.no_grad():
            output = model(img)
            _, predicted = torch.max(output, 1)
            prediction = predicted.item()
        
        # Return the prediction as JSON
        return jsonify({'prediction': prediction})

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

