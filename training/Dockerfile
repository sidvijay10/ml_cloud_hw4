FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime

# Install any needed packages
RUN pip install torchvision

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Run training script when the container launches
CMD ["python", "train_mnist.py"]
