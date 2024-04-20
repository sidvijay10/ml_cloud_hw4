document.getElementById('uploadForm').addEventListener('submit', function(event) {
    event.preventDefault();

    var formData = new FormData();
    formData.append('image', document.getElementById('imageInput').files[0]);

    fetch('http://34.29.4.121/predict', {
        method: 'POST',
        body: formData,
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('predictionResult').textContent = data.prediction;
    })
    .catch(error => console.error('Error:', error));
});

