let mediaRecorder;
let audioChunks = [];
let recordingStartTime;

const BASE_URL = "https://speech-emotion-llm-e32536e9edb2.herokuapp.com/";

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
        console.log(`Recorded audio MIME type: ${audioBlob.type}`);
        const audioUrl = URL.createObjectURL(audioBlob);
        document.getElementById('audioPlayback').src = audioUrl;

        const recordingDuration = (Date.now() - recordingStartTime) / 1000;
        if (recordingDuration < 3 || recordingDuration > 30) {
            document.getElementById("status").innerText = "Recording must be between 3 and 30 seconds.";
            return;
        }

        const formData = new FormData();
        formData.append("audio", audioBlob, "recording.wav");

        try {
            const response = await fetch(`${BASE_URL}/process_audio`, {
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

function displayResults(result) {
    const responseDiv = document.getElementById('response');
    responseDiv.innerHTML = '';

    // Display Emotion Probabilities
    const emotionsDiv = document.createElement('div');
    emotionsDiv.innerHTML = '<h3>Emotion Probabilities:</h3>';
    for (const [emotion, probability] of Object.entries(result["Emotion Probabilities"])) {
        emotionsDiv.innerHTML += `<p>${emotion}: ${probability}</p>`;
    }
    responseDiv.appendChild(emotionsDiv);

    // Display Transcription
    const transcriptionDiv = document.createElement('div');
    transcriptionDiv.innerHTML = `<h3>Transcription:</h3><p>${result.Transcription}</p>`;
    responseDiv.appendChild(transcriptionDiv);

    // Display LLM Interpretation
    const interpretationDiv = document.createElement('div');
    interpretationDiv.innerHTML = `<h3>AI Interpretation:</h3><p>${result["LLM Interpretation"]}</p>`;
    responseDiv.appendChild(interpretationDiv);
}
