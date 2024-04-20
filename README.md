
# MNIST Model Deployment Project

This project involves training and deploying a neural network model to classify handwritten digits from the MNIST dataset using Google Kubernetes Engine (GKE). The project is divided into two main parts: training and inference.

## Project Structure

The project directory is structured as follows:

```
/
|- /training
|  |- Dockerfile
|  |- train_mnist.py
|  |- mnist-training-job.yaml
|  |- mnist-pvc.yaml
|
|- /inference
   |- Dockerfile
   |- infer_mnist.py
   |- mnist-inference-deployment.yaml
   |- mnist-inference-service.yaml
```

### Training Folder

This folder contains all files necessary to train the MNIST model.

- **Dockerfile**: Defines the Docker image for training the model. It includes the environment and dependencies needed to run `train_mnist.py`.
- **train_mnist.py**: The Python script that trains the MNIST model using PyTorch. It saves the trained model to a Persistent Volume.
- **mnist-training-job.yaml**: Kubernetes Job YAML configuration for running the training job in GKE.
- **mnist-pvc.yaml**: YAML configuration for the Persistent Volume Claim to store the trained model persistently.

### Inference Folder

This folder contains all files necessary for deploying the model to perform inference.

- **Dockerfile**: Builds the Docker image for the inference server, which hosts the Flask app.
- **infer_mnist.py**: Flask application that loads the trained model and predicts the digit from a POSTed image.
- **mnist-inference-deployment.yaml**: Kubernetes Deployment YAML configuration for deploying the inference server in GKE.
- **mnist-inference-service.yaml**: Kubernetes Service YAML configuration for exposing the inference server externally via a LoadBalancer.

## Testing the Deployment

To test the deployed model, you can send an image of a handwritten digit to the inference service using `curl`. This command should be executed from a terminal with access to the internet and where the image file is locally available.

```bash
curl -X POST -F 'image=@path_to_your_image.png' http://34.29.4.121/predict
```

Replace `path_to_your_image.png` with the path to your handwritten digit image file. The service will return the predicted digit.

### Example:

```bash
curl -X POST -F 'image=@six.png' http://34.29.4.121/predict
```

This command sends an image named `six.png` (which should be a digit image) located in the current directory to the inference service. The service will return the digit recognized in the image.

---

This README should help new users or contributors understand the structure of your project and how to interact with the deployed model. It's also good practice to include additional sections as needed, such as requirements for running the project locally, versioning information, and authorship or licensing details.
