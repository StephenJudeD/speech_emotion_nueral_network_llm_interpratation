let mediaRecorder;
let audioChunks = [];

document.getElementById('recordButton').addEventListener('click', startRecording);
document.getElementById('stopButton').addEventListener('click', stopRecording);

async function startRecording() {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

    mediaRecorder = new MediaRecorder(stream);
    audioChunks = [];
    mediaRecorder.addEventListener("dataavailable", event => {
        audioChunks.push(event.data);
    });

    mediaRecorder.start();
    document.getElementById("status").innerText = "Recording...";
    document.getElementById('recordButton').disabled = true;
    document.getElementById('stopButton').disabled = false;
}

function stopRecording() {
    mediaRecorder.stop();
    document.getElementById("status").innerText = "Recording stopped.";
    document.getElementById('recordButton').disabled = false;
    document.getElementById('stopButton').disabled = true;

    mediaRecorder.addEventListener("stop", async () => {
        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
        const audioUrl = URL.createObjectURL(audioBlob);
        document.getElementById('audioPlayback').src = audioUrl;

        // Send audio blob to the server
        const formData = new FormData();
        formData.append("audio", audioBlob, "audio.wav");

        try {
            const response = await fetch('/process_audio', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error("Network response was not ok");
            }

            const result = await response.json();
            displayResults(result);
        } catch (error) {
            document.getElementById('response').innerText = "Error: " + error.message;
        }
    });
}

function displayResults(result) {
    const responseDiv = document.getElementById('response');
    responseDiv.innerHTML = `
        <h2>Predictions:</h2>
        <p><strong>Emotion Probabilities:</strong> ${JSON.stringify(result["Emotion Probabilities"], null, 2)}</p>
        <p><strong>Transcription:</strong> ${result["Transcription"]}</p>
        <p><strong>LLM Interpretation:</strong> ${result["LLM Interpretation"]}</p>
    `;
}