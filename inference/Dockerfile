FROM python:3.8-slim

RUN pip install flask torch torchvision pillow --no-cache-dir

COPY infer_mnist.py /app/infer_mnist.py

EXPOSE 5000

CMD ["python", "/app/infer_mnist.py"]
