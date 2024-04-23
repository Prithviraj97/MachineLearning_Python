const video = document.getElementById('videoElement');
const canvas = document.getElementById('canvasElement');
const ctx = canvas.getContext('2d');

navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        video.srcObject = stream;
    })
    .catch(error => {
        console.error('Error accessing the camera: ', error);
    });

function captureFrame() {
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const imageData = canvas.toDataURL('image/jpeg');

    fetch('/detect', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ imageData: imageData }),
    })
    .then(response => response.json())
    .then(results => {
        displayDetectionResults(results);
    })
    .catch(error => {
        console.error('Error detecting objects: ', error);
    });
}

function displayDetectionResults(results) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    results.forEach(result => {
        ctx.beginPath();
        ctx.rect(result.x, result.y, result.width, result.height);
        ctx.strokeStyle = 'red';
        ctx.lineWidth = 2;
        ctx.stroke();
    });
}

video.addEventListener('play', () => {
    setInterval(captureFrame, 1000); // Capture a frame every second
});
