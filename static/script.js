let mediaRecorder;
let audioChunks = [];
let recordingStartTime;

// Define the base URL (update this with your actual Heroku app URL)
const BASE_URL = "https://speech-emotion-llm-e32536e9edb2.herokuapp.com/"; // Replace with your actual app URL

document.getElementById('startRecording').addEventListener('click', startRecording);
document.getElementById('stopRecording').addEventListener('click', stopRecording);

async function startRecording() {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);
    audioChunks = [];
    recordingStartTime = Date.now();

    mediaRecorder.ondataavailable = event => {
        audioChunks.push(event.data);
    };

    mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunks);
        const audioUrl = URL.createObjectURL(audioBlob);
        document.getElementById('audioPlayback').src = audioUrl;

        // Check the duration
        const recordingDuration = (Date.now() - recordingStartTime) / 1000; // in seconds
        if (recordingDuration < 3 || recordingDuration > 30) {
            document.getElementById("status").innerText = "Recording must be between 3 and 30 seconds.";
            return;
        }

        const formData = new FormData();
        formData.append("audio", audioBlob, "recording.wav");

        try {
            const response = await fetch(`${BASE_URL}/process_audio`, { // Use the full URL here
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error("Network response was not ok");
            }

            const result = await response.json();
            displayResults(result);
            document.getElementById("status").innerText = "Prediction fetched successfully.";
        } catch (error) {
            document.getElementById('response').innerText = "Error: " + error.message;
        }
    };

    mediaRecorder.start();
    document.getElementById('startRecording').disabled = true;
    document.getElementById('stopRecording').disabled = false;
    document.getElementById("status").innerText = "Recording...";
}

function stopRecording() {
    mediaRecorder.stop();
    document.getElementById('startRecording').disabled = false;
    document.getElementById('stopRecording').disabled = true;
}
