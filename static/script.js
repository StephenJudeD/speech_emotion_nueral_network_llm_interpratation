let mediaRecorder;
let audioChunks = [];
let recordingStartTime;

const BASE_URL = "https://speech-emotion-llm-e32536e9edb2.herokuapp.com/";

document.getElementById('startRecording').addEventListener('click', startRecording);
document.getElementById('stopRecording').addEventListener('click', stopRecording);

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
            audio: {
                channelCount: 1,        // Mono audio
                sampleRate: 16000,      // 16 kHz sample rate
                sampleSize: 16          // 16-bit
            } 
        });
        
        mediaRecorder = new MediaRecorder(stream, {
            mimeType: 'audio/webm;codecs=opus'  // Using WebM format with Opus codec
        });
        
        audioChunks = [];
        recordingStartTime = Date.now();

        mediaRecorder.ondataavailable = event => {
            audioChunks.push(event.data);
        };

        mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
            const audioUrl = URL.createObjectURL(audioBlob);
            document.getElementById('audioPlayback').src = audioUrl;

            // Check recording duration
            const recordingDuration = (Date.now() - recordingStartTime) / 1000;
            if (recordingDuration < 3 || recordingDuration > 30) {
                document.getElementById("status").innerText = "Recording must be between 3 and 30 seconds.";
                return;
            }

            // Convert to WAV format before sending
            const formData = new FormData();
            
            // Add a timestamp to prevent caching
            const timestamp = new Date().getTime();
            formData.append("audio", audioBlob, `recording_${timestamp}.webm`);

            try {
                document.getElementById("status").innerText = "Processing audio...";
                
                const response = await fetch(`${BASE_URL}/process_audio`, {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const result = await response.json();
                displayResults(result);
                document.getElementById("status").innerText = "Analysis complete!";
            } catch (error) {
                console.error('Error:', error);
                document.getElementById('response').innerText = "Error: " + error.message;
                document.getElementById("status").innerText = "Error processing audio.";
            }
        };

        mediaRecorder.start(1000); // Collect data every second
        document.getElementById('startRecording').disabled = true;
        document.getElementById('stopRecording').disabled = false;
        document.getElementById("status").innerText = "Recording...";
        
    } catch (error) {
        console.error('Error accessing microphone:', error);
        document.getElementById("status").innerText = "Error accessing microphone. Please ensure microphone permissions are granted.";
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
        mediaRecorder.stream.getTracks().forEach(track => track.stop());
        document.getElementById('startRecording').disabled = false;
        document.getElementById('stopRecording').disabled = true;
    }
}

function displayResults(result) {
    const responseDiv = document.getElementById('response');
    responseDiv.innerHTML = '';

    if (result.error) {
        responseDiv.innerHTML = `<div class="error">Error: ${result.error}</div>`;
        return;
    }

    // Display Emotion Probabilities
    const emotionsDiv = document.createElement('div');
    emotionsDiv.innerHTML = '<h3>Emotion Probabilities:</h3>';
    for (const [emotion, probability] of Object.entries(result["Emotion Probabilities"] || {})) {
        emotionsDiv.innerHTML += `<p>${emotion}: ${probability}</p>`;
    }
    responseDiv.appendChild(emotionsDiv);

    // Display Transcription
    if (result.Transcription) {
        const transcriptionDiv = document.createElement('div');
        transcriptionDiv.innerHTML = `<h3>Transcription:</h3><p>${result.Transcription}</p>`;
        responseDiv.appendChild(transcriptionDiv);
    }

    // Display LLM Interpretation
    if (result["LLM Interpretation"]) {
        const interpretationDiv = document.createElement('div');
        interpretationDiv.innerHTML = `<h3>AI Interpretation:</h3><p>${result["LLM Interpretation"]}</p>`;
        responseDiv.appendChild(interpretationDiv);
    }
}
