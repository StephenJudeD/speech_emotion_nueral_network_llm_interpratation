document.getElementById('uploadButton').addEventListener('click', uploadAudio);

async function uploadAudio() {
    const audioFileInput = document.getElementById('audioInput');
    const audioFile = audioFileInput.files[0];

    if (!audioFile) {
        document.getElementById("status").innerText = "Please select an audio file.";
        return;
    }

    // Display the audio playback for the uploaded file
    const audioUrl = URL.createObjectURL(audioFile);
    document.getElementById('audioPlayback').src = audioUrl;

    const formData = new FormData();
    formData.append("audio", audioFile);

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
        document.getElementById("status").innerText = "Processing complete.";
    } catch (error) {
        document.getElementById('response').innerText = "Error: " + error.message;
    }
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
