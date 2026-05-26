/**
 * Sign Language Recognition - Gemini UI
 * Integrated with MediaPipe Hands and Python SVM Backend.
 */

// UI Elements
const videoElement = document.getElementById('webcam');
const canvasElement = document.getElementById('output_canvas');
const canvasCtx = canvasElement.getContext('2d');
const glossContainer = document.getElementById('glossContainer');
const finalSentenceElement = document.getElementById('finalSentence');
const currentTimeElement = document.getElementById('currentTime');
const progressFill = document.getElementById('progressFill');
const statusTitle = document.getElementById('statusTitle');
const statusDesc = document.getElementById('statusDesc');
const clearBtn = document.getElementById('clearBtn');
const dockItems = document.querySelectorAll('.dock-item');

// Processing Canvas (Offscreen)
const procCanvas = document.createElement('canvas');
const procCtx = procCanvas.getContext('2d');

let lastInferenceTime = 0;
const INFERENCE_INTERVAL = 100; // ms

// --- 1. MediaPipe Setup ---

const hands = new Hands({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
});

hands.setOptions({
    maxNumHands: 2,
    modelComplexity: 1,
    minDetectionConfidence: 0.7,
    minTrackingConfidence: 0.7
});

hands.onResults(onResults);

const camera = new Camera(videoElement, {
    onFrame: async () => {
        await hands.send({ image: videoElement });
    },
    width: 1280,
    height: 720
});

// --- 2. Inference & UI Sync ---

async function onResults(results) {
    // Sync canvas size
    if (canvasElement.width !== videoElement.clientWidth || canvasElement.height !== videoElement.clientHeight) {
        canvasElement.width = videoElement.clientWidth;
        canvasElement.height = videoElement.clientHeight;
    }

    // Draw Feedback
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    
    // Note: We draw on the mirrored canvas to match the user view
    // CSS handle the mirroring for video, but for canvas we might need it too if not handled by CSS
    // Actually our CSS mirrors BOTH video and canvas.
    // So landmarks from MediaPipe (which saw flipped image) will be "correctly flipped" when drawn on flipped canvas?
    // Let's draw carefully.
    
    if (results.multiHandLandmarks) {
        setPipelineStage(0); // 1. Hand Tracking Active
        for (const landmarks of results.multiHandLandmarks) {
            drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, { color: '#ffffff', lineWidth: 2 });
            drawLandmarks(canvasCtx, landmarks, { color: '#4ade80', lineWidth: 1, radius: 2 });
        }
    } else {
        setPipelineStage(-1);
    }
    canvasCtx.restore();

    // Throttled Inference
    const now = Date.now();
    if (now - lastInferenceTime > INFERENCE_INTERVAL) {
        lastInferenceTime = now;
        await performInference(results);
    }
}

async function performInference(results) {
    const handsData = [];
    if (results.multiHandLandmarks) {
        results.multiHandLandmarks.forEach((landmarks, index) => {
            const rawLabel = results.multiHandedness[index].label;
            // Mirror logic for handedness (since we flipped the image for MediaPipe)
            const handedness = rawLabel === 'Right' ? 'Left' : 'Right';
            handsData.push({
                landmarks: landmarks,
                handedness: handedness
            });
        });
    }

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ hands: handsData })
        });
        const data = await response.json();
        updateUI(data);
    } catch (err) {
        console.error("Inference Error:", err);
    }
}

function updateUI(data) {
    // 1. Gloss Tags
    if (data.glosses) {
        glossContainer.innerHTML = '';
        data.glosses.forEach((g, i) => {
            const tag = document.createElement('span');
            tag.className = `gloss-tag active green`;
            tag.innerText = g;
            glossContainer.appendChild(tag);
        });
    }

    // 2. Center Sentence
    if (data.text) {
        finalSentenceElement.innerText = data.text;
        finalSentenceElement.classList.remove('hidden');
    } else {
        finalSentenceElement.classList.add('hidden');
    }

    // 3. Status Card (Real-time classification)
    if (data.mode === 'INPUT') {
        setPipelineStage(1); // 2. Classification
        updateStatusCard(data.prediction, data.confidence, data.stable_text);
    } else if (data.mode === 'IDLE') {
        updateStatusCard('START 대기 중', 0, data.message);
    } else if (data.mode === 'RESULT') {
        setPipelineStage(3); // 4. Text Output
        updateStatusCard('완료', 100, data.message);
    }

    // Update Dock based on mode
    if (data.mode === 'INPUT') {
        // If we are composing, maybe stage 3
        if (data.glosses && data.glosses.length > 0) {
            setPipelineStage(2); // 3. Assembly
        }
    }
}

function updateStatusCard(title, percent, desc) {
    statusTitle.innerText = title || '...';
    statusDesc.innerText = desc || '';
    progressFill.style.width = `${percent || 0}%`;
}

function setPipelineStage(index) {
    dockItems.forEach((item, i) => {
        if (i === index) {
            item.classList.add('active');
            if (!item.querySelector('.active-indicator')) {
                const indicator = document.createElement('div');
                indicator.className = 'active-indicator';
                item.appendChild(indicator);
            }
        } else {
            // Keep active if it's a previous stage? 
            // Or just single active. Let's do single active for clarity.
            if (i > index) {
                item.classList.remove('active');
                const ind = item.querySelector('.active-indicator');
                if (ind) ind.remove();
            }
        }
    });
}

// --- 3. Utilities ---

// Empty for now

// --- 4. Event Listeners & Initialization ---

clearBtn.addEventListener('click', async () => {
    await fetch('/api/reset', { method: 'POST' });
});

window.addEventListener('DOMContentLoaded', () => {
    camera.start();
});
