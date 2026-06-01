const videoElement = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");

// UI Elements
const appContainer = document.getElementById("appContainer");
const nicknameInput = document.getElementById("nicknameInput");
const roomCodeInput = document.getElementById("roomCodeInput");
const createRoomBtn = document.getElementById("createRoomBtn");
const joinRoomBtn = document.getElementById("joinRoomBtn");
const rejoinRoomBtn = document.getElementById("rejoinRoomBtn");
const roleInputs = Array.from(document.querySelectorAll("input[name='userRole']"));
const entryMessage = document.getElementById("entryMessage");
const roomCodeBadge = document.getElementById("roomCodeBadge");
const connectionStatus = document.getElementById("connectionStatus");
const peerConnectionBadge = document.getElementById("peerConnectionBadge");
const peerAvatar = document.getElementById("peerAvatar");
const peerStatusTitle = document.getElementById("peerStatusTitle");
const peerStatusText = document.getElementById("peerStatusText");
const localUserName = document.getElementById("localUserName");
const modeBadge = document.getElementById("modeBadge");
const tokenContainer = document.getElementById("tokenContainer");
const resultText = document.getElementById("resultText");
const recognitionCard = document.getElementById("recognitionCard");
const statusTitle = document.getElementById("statusTitle");
const confidenceText = document.getElementById("confidenceText");
const progressFill = document.getElementById("progressFill");
const debugStatus = document.getElementById("debugStatus");
const topPredictions = document.getElementById("topPredictions");
const clearBtn = document.getElementById("clearBtn");
const leaveBtn = document.getElementById("leaveBtn");
const micBtn = document.getElementById("micBtn");
const cameraBtn = document.getElementById("cameraBtn");
const captionStatus = document.getElementById("captionStatus");
const captionLog = document.getElementById("captionLog");

const INFERENCE_INTERVAL = 100;
let lastInferenceTime = 0;
let tokenConfidenceHistory = [];
let previousTokens = [];
let cameraStarted = false;
let lastCaptionText = "";
let socket = null;
let recognition = null;
let recognizing = false;

const appState = {
    screen: "entry",
    nickname: "",
    roomCode: "",
    connected: false,
    role: "sign",
    micEnabled: true,
    cameraEnabled: true,
    lastNickname: "",
    lastRoomCode: "",
    lastRole: "sign",
    captions: [],
};

const roleMeta = {
    sign: {
        label: "수어 사용자",
        shortLabel: "수어",
        title: "START를 입력하세요",
        captionIdle: "수어 인식 대기",
        resultEmpty: "준비됨",
    },
    voice: {
        label: "비수어 사용자",
        shortLabel: "STT",
        title: "음성 인식 대기",
        captionIdle: "STT 대기",
        resultEmpty: "말하면 자막으로 표시됩니다",
    },
};

const modeMeta = {
    IDLE: {
        label: "대기 중",
        className: "idle",
        title: "START를 입력하세요",
    },
    INPUT: {
        label: "입력 중",
        className: "input",
        title: "인식 중...",
    },
    RESULT: {
        label: "완료",
        className: "result",
        title: "입력 완료",
    },
    ERROR: {
        label: "연결 오류",
        className: "error",
        title: "서버 연결 실패",
    },
};

const hands = new Hands({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
});

hands.setOptions({
    maxNumHands: 2,
    modelComplexity: 1,
    minDetectionConfidence: 0.7,
    minTrackingConfidence: 0.7,
});

hands.onResults(onResults);

const camera = new Camera(videoElement, {
    onFrame: async () => {
        await hands.send({ image: videoElement });
    },
    width: 1280,
    height: 720,
});

function normalizeRoomCode(value) {
    return value.trim().toUpperCase().replace(/[^A-Z0-9]/g, "").slice(0, 6);
}

function generateRoomCode() {
    const alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789";
    let code = "";
    for (let index = 0; index < 4; index += 1) {
        code += alphabet[Math.floor(Math.random() * alphabet.length)];
    }
    return code;
}

function setEntryMessage(message, isError = false) {
    entryMessage.textContent = message;
    entryMessage.style.color = isError ? "var(--accent-red)" : "var(--text-secondary)";
}

function setEntryBusy(isBusy) {
    createRoomBtn.disabled = isBusy;
    joinRoomBtn.disabled = isBusy;
    rejoinRoomBtn.disabled = isBusy;
}

function selectedRole() {
    const checked = roleInputs.find((input) => input.checked);
    return checked ? checked.value : "sign";
}

function setSelectedRole(role) {
    roleInputs.forEach((input) => {
        input.checked = input.value === role;
    });
}

function isSignRole() {
    return appState.role === "sign";
}

function sendSocketMessage(message) {
    if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify(message));
    }
}

function sendProfile() {
    sendSocketMessage({
        type: "profile",
        user: appState.nickname,
        role: appState.role,
    });
}

async function requestCameraPermission() {
    if (!window.isSecureContext) {
        throw new Error("insecure_context");
    }
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error("media_devices_unavailable");
    }
    const permissionStream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false,
    });
    permissionStream.getTracks().forEach((track) => track.stop());
}

function cameraErrorMessage(error) {
    if (!window.isSecureContext || error.message === "insecure_context") {
        return "카메라는 HTTPS 또는 localhost 접속에서만 사용할 수 있습니다. 다른 기기는 HTTPS 주소로 접속하세요.";
    }
    if (error.name === "NotAllowedError" || error.name === "PermissionDeniedError") {
        return "브라우저 카메라 권한이 거부되었습니다. 주소창의 카메라 권한을 허용하세요.";
    }
    if (error.name === "NotFoundError" || error.name === "DevicesNotFoundError") {
        return "사용 가능한 카메라를 찾지 못했습니다. 카메라 장치를 확인하세요.";
    }
    if (error.message === "media_devices_unavailable") {
        return "이 브라우저 또는 접속 주소에서는 카메라 API를 사용할 수 없습니다.";
    }
    return "카메라를 시작하지 못했습니다. 브라우저 권한과 접속 주소를 확인하세요.";
}

async function enterRoom(roomCode) {
    const nickname = nicknameInput.value.trim();
    const normalizedRoomCode = normalizeRoomCode(roomCode);
    const role = selectedRole();

    if (!nickname) {
        setEntryMessage("닉네임을 입력하세요.", true);
        nicknameInput.focus();
        return;
    }
    if (!normalizedRoomCode) {
        setEntryMessage("방 코드를 입력하거나 방을 생성하세요.", true);
        roomCodeInput.focus();
        return;
    }

    roomCodeInput.value = normalizedRoomCode;
    setEntryBusy(true);
    if (role === "sign") {
        setEntryMessage("카메라 권한을 요청하는 중입니다.");
        try {
            await requestCameraPermission();
        } catch (error) {
            console.error("Camera permission error:", error);
            const message = cameraErrorMessage(error);
            alert(message);
            setEntryMessage(message, true);
            setEntryBusy(false);
        }
    } else {
        setEntryMessage("STT 화면으로 입장합니다.");
    }

    appState.screen = "call";
    appState.nickname = nickname;
    appState.roomCode = normalizedRoomCode;
    appState.connected = true;
    appState.role = role;
    appState.lastNickname = nickname;
    appState.lastRoomCode = normalizedRoomCode;
    appState.lastRole = role;
    appState.captions = [];
    lastCaptionText = "";

    appContainer.classList.toggle("role-sign", role === "sign");
    appContainer.classList.toggle("role-voice", role === "voice");
    roomCodeBadge.textContent = `방 ${normalizedRoomCode}`;
    connectionStatus.textContent = `${nickname} · ${roleMeta[role].shortLabel}`;
    peerConnectionBadge.textContent = "대기";
    peerAvatar.textContent = "상대";
    peerStatusTitle.textContent = "상대방 대기 중";
    peerStatusText.textContent = "같은 방 사용자가 들어오면 역할과 최근 자막이 표시됩니다.";
    localUserName.textContent = `${nickname} · ${roleMeta[role].shortLabel}`;
    captionStatus.textContent = roleMeta[role].captionIdle;
    modeBadge.textContent = role === "sign" ? "대기 중" : "STT";
    statusTitle.textContent = roleMeta[role].title;
    confidenceText.textContent = role === "sign" ? "0%" : "ON";
    progressFill.style.width = "0%";
    updateResult("");
    updateTokens([], {});
    topPredictions.innerHTML = "";
    debugStatus.textContent = role === "sign" ? "손 감지 대기" : "수어 인식 꺼짐 | STT 모드";
    renderCaptionLog();
    appContainer.classList.remove("entry-active");
    rejoinRoomBtn.hidden = false;
    setEntryBusy(false);

    if (role === "sign") {
        await startCamera();
        stopSpeechRecognition();
    } else {
        stopCamera();
        startSpeechRecognition();
    }
    connectWebSocket();
}

async function startCamera() {
    if (cameraStarted) return;
    try {
        await camera.start();
        cameraStarted = true;
    } catch (error) {
        console.error("Camera error:", error);
        statusTitle.textContent = "카메라 오류";
        connectionStatus.textContent = cameraErrorMessage(error);
    }
}

function stopCamera() {
    if (typeof camera.stop === "function") {
        camera.stop();
    }
    const stream = videoElement.srcObject;
    if (stream) {
        stream.getTracks().forEach((track) => track.stop());
        videoElement.srcObject = null;
    }
    cameraStarted = false;
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
}

function leaveRoom() {
    if (socket) {
        socket.close();
        socket = null;
    }
    stopCamera();
    appState.screen = "entry";
    appState.connected = false;
    appState.cameraEnabled = true;
    stopSpeechRecognition();
    appContainer.classList.add("entry-active");
    appContainer.classList.remove("role-sign", "role-voice");
    nicknameInput.value = appState.lastNickname || appState.nickname;
    roomCodeInput.value = appState.lastRoomCode || appState.roomCode;
    setSelectedRole(appState.lastRole || appState.role || "sign");
    rejoinRoomBtn.hidden = !appState.lastRoomCode;
    roomCodeBadge.textContent = "방 없음";
    connectionStatus.textContent = "입장 대기";
    peerConnectionBadge.textContent = "대기";
    localUserName.textContent = "입장 전";
    captionStatus.textContent = "실시간 대기";
    appState.captions = [];
    lastCaptionText = "";
    renderCaptionLog();
    setMediaButtonState(cameraBtn, appState.cameraEnabled);
    setEntryMessage(appState.lastRoomCode ? `방 ${appState.lastRoomCode}에 다시 입장할 수 있습니다.` : "방 생성 시 임시 코드가 자동으로 만들어집니다.");
}

function setMediaButtonState(button, enabled) {
    button.classList.toggle("low", !enabled);
    button.style.opacity = enabled ? "1" : "0.55";
}

function toggleCameraTrack() {
    appState.cameraEnabled = !appState.cameraEnabled;
    const stream = videoElement.srcObject;
    if (stream) {
        stream.getVideoTracks().forEach((track) => {
            track.enabled = appState.cameraEnabled;
        });
    }
    setMediaButtonState(cameraBtn, appState.cameraEnabled);
}

function toggleMicState() {
    appState.micEnabled = !appState.micEnabled;
    setMediaButtonState(micBtn, appState.micEnabled);
    if (!recognition) {
        captionStatus.textContent = "STT 미지원 브라우저";
        return;
    }
    if (appState.micEnabled) {
        startSpeechRecognition();
    } else {
        stopSpeechRecognition();
    }
}

function startSpeechRecognition() {
    if (!recognition) {
        captionStatus.textContent = "STT 미지원 브라우저";
        statusTitle.textContent = "STT 미지원";
        return;
    }
    if (recognizing) {
        captionStatus.textContent = "음성 인식 중";
        return;
    }
    try {
        recognition.start();
        captionStatus.textContent = "음성 인식 중";
        statusTitle.textContent = "음성을 듣는 중...";
        modeBadge.textContent = "STT";
        modeBadge.className = "mode-badge input";
    } catch (error) {
        console.error("STT start error:", error);
        captionStatus.textContent = "STT 시작 실패";
        statusTitle.textContent = "마이크 권한 확인 필요";
    }
}

function stopSpeechRecognition() {
    if (!recognition || !recognizing) {
        if (appState.role === "voice") {
            captionStatus.textContent = "마이크 꺼짐";
        }
        return;
    }
    recognition.stop();
    captionStatus.textContent = "마이크 꺼짐";
    if (appState.role === "voice") {
        statusTitle.textContent = "음성 인식 대기";
        modeBadge.textContent = "STT";
        modeBadge.className = "mode-badge idle";
    }
}

function syncCanvasSize() {
    if (
        canvasElement.width !== videoElement.clientWidth ||
        canvasElement.height !== videoElement.clientHeight
    ) {
        canvasElement.width = videoElement.clientWidth;
        canvasElement.height = videoElement.clientHeight;
    }
}

async function onResults(results) {
    if (appState.screen !== "call") return;
    syncCanvasSize();
    if (!isSignRole()) {
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
        return;
    }
    drawHands(results);

    const now = Date.now();
    if (now - lastInferenceTime > INFERENCE_INTERVAL) {
        lastInferenceTime = now;
        await performInference(results);
    }
}

function drawHands(results) {
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

    if (results.multiHandLandmarks) {
        for (const landmarks of results.multiHandLandmarks) {
            drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, { color: "rgba(255, 255, 255, 0.4)", lineWidth: 2 });
            drawLandmarks(canvasCtx, landmarks, { color: "#ffffff", lineWidth: 1, radius: 2 });
        }
    }

    canvasCtx.restore();
}

async function performInference(results) {
    if (!isSignRole()) return;
    const handsData = [];
    if (results.multiHandLandmarks) {
        results.multiHandLandmarks.forEach((landmarks, index) => {
            const rawLabel = results.multiHandedness[index].label;
            const handedness = rawLabel === "Right" ? "Left" : "Right";
            const mirroredLandmarks = landmarks.map((point) => ({
                x: 1 - point.x,
                y: point.y,
                z: point.z,
            }));
            handsData.push({ landmarks: mirroredLandmarks, handedness });
        });
    }

    try {
        const response = await fetch("/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ hands: handsData, timestamp: Date.now() }),
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        updateUI(await response.json());
    } catch (error) {
        console.error("Inference error:", error);
        updateErrorUI();
    }
}

function updateUI(data) {
    if (!isSignRole()) return;
    const meta = modeMeta[data.mode] || modeMeta.IDLE;
    const confidence = Math.round(data.confidence || 0);
    
    setMode(meta, confidence);
    updateResult(data.text);
    updateTokens(data.glosses || [], data);
    
    // Update recognition card details
    const prediction = data.prediction || meta.title;
    statusTitle.textContent = prediction;
    
    confidenceText.textContent = `${confidence}%`;
    progressFill.style.width = `${confidence}%`;
    updateCaptionPreview(data);
    updateDebug(data);
}

function updateErrorUI() {
    const meta = modeMeta.ERROR;
    setMode(meta, null);
    statusTitle.textContent = meta.title;
    confidenceText.textContent = "0%";
    progressFill.style.width = "0%";
    debugStatus.textContent = "백엔드 연결 실패";
    captionStatus.textContent = "서버 연결 실패";
    topPredictions.innerHTML = "";
}

function setMode(meta, confidence = null) {
    modeBadge.textContent = meta.label;
    modeBadge.className = `mode-badge ${meta.className}`;
    const confidenceState = Number.isFinite(confidence) && confidence > 0
        ? ` ${confidenceClass(confidence)}`
        : "";
    recognitionCard.className = `glass-card recognition-card ${meta.className}${confidenceState}`;
}

function updateResult(text) {
    if (text) {
        resultText.textContent = text;
        resultText.classList.remove("empty");
    } else {
        resultText.textContent = roleMeta[appState.role].resultEmpty;
        resultText.classList.add("empty");
    }
}

function updateCaptionPreview(data) {
    if (data.text) {
        captionStatus.textContent = data.mode === "RESULT" ? "입력 완료" : "수어 입력 중";
        if (data.text && data.text !== lastCaptionText) {
            appendCaption({
                type: "sign_result",
                user: appState.nickname || "나",
                text: data.text,
                confidence: data.confidence,
            });

            if (socket && socket.readyState === WebSocket.OPEN) {
                console.log("수어 결과 websocket 전송:", data.text);
                sendSocketMessage({
                    type: "sign_result",
                    user: appState.nickname,
                    role: appState.role,
                    text: data.text,
                    confidence: data.confidence,
                });
            }

            lastCaptionText = data.text;
        }
    } else if (data.prediction) {
        captionStatus.textContent = "수어 인식 중";
    } else {
        captionStatus.textContent = "수어 인식 대기";
    }
}

function captionKind(messageType) {
    if (messageType === "stt_result") return "stt";
    if (messageType === "user_status") return "status";
    return "sign";
}

function captionLabel(messageType) {
    if (messageType === "stt_result") return "음성";
    if (messageType === "user_status") return "상태";
    return "수어";
}

function confidencePercent(value) {
    const confidence = Number(value);
    if (!Number.isFinite(confidence)) return null;
    return confidence <= 1 ? Math.round(confidence * 100) : Math.round(confidence);
}

function appendCaption(message) {
    if (!message || !message.text) return;
    appState.captions.push({
        type: message.type || "sign_result",
        user: message.user || appState.nickname || "사용자",
        text: message.text,
        confidence: message.confidence,
        timestamp: message.timestamp || Date.now(),
    });
    appState.captions = appState.captions.slice(-8);
    renderCaptionLog();
}

function updatePeerPanel(message) {
    if (!message || !message.user || message.user === appState.nickname) return;
    const role = message.role || "unknown";
    const roleLabel = roleMeta[role] ? roleMeta[role].label : "상대방";
    peerConnectionBadge.textContent = "접속";
    peerAvatar.textContent = role === "voice" ? "STT" : "수어";
    peerStatusTitle.textContent = `${message.user} · ${roleLabel}`;
    if (message.text) {
        peerStatusText.textContent = `최근 자막: ${message.text}`;
    } else {
        peerStatusText.textContent = "같은 방에 접속했습니다. 자막을 기다리는 중입니다.";
    }
}

function renderCaptionLog() {
    captionLog.innerHTML = "";
    if (appState.captions.length === 0) {
        const empty = document.createElement("div");
        empty.className = "caption-empty";
        empty.textContent = "수어와 음성 자막이 여기에 표시됩니다.";
        captionLog.appendChild(empty);
        return;
    }

    appState.captions.forEach((message) => {
        const item = document.createElement("div");
        item.className = `caption-item ${captionKind(message.type)}`;

        const type = document.createElement("span");
        type.className = "caption-type";
        type.textContent = captionLabel(message.type);

        const body = document.createElement("div");
        const text = document.createElement("strong");
        text.textContent = message.text;

        const meta = document.createElement("span");
        meta.className = "caption-meta";
        const confidence = confidencePercent(message.confidence);
        const confidenceText = confidence !== null
            ? ` · ${confidence}%`
            : "";
        meta.textContent = `${message.user}${confidenceText}`;

        body.appendChild(text);
        body.appendChild(meta);
        item.appendChild(type);
        item.appendChild(body);
        captionLog.appendChild(item);
    });

    captionLog.scrollTop = captionLog.scrollHeight;
}

function handleRealtimeMessage(message) {
    if (!message || !message.type) return;
    if (message.type === "user_status") {
        if (message.user === appState.nickname) {
            return;
        }
        peerConnectionBadge.textContent = message.status === "joined" ? "접속" : "대기";
        if (message.status === "joined") {
            peerStatusTitle.textContent = `${message.user} 접속`;
            peerStatusText.textContent = "상대방 역할 정보를 기다리는 중입니다.";
            sendProfile();
        }
        if (message.status === "left") {
            peerAvatar.textContent = "상대";
            peerStatusTitle.textContent = "상대방 대기 중";
            peerStatusText.textContent = "같은 방 사용자가 들어오면 역할과 최근 자막이 표시됩니다.";
        }
        appendCaption({
            type: "user_status",
            user: message.user || "상대방",
            text: message.status === "joined" ? "상대방이 입장했습니다." : "상대방이 나갔습니다.",
        });
        return;
    }
    if (message.type === "profile") {
        updatePeerPanel(message);
        return;
    }
    if (message.type === "sign_result" || message.type === "stt_result") {
        updatePeerPanel(message);
        appendCaption(message);
    }
}

window.handleRealtimeMessage = handleRealtimeMessage;

function updateTokens(tokens, data = {}) {
    const tokenSignature = tokens.join("\u0001");
    const previousSignature = previousTokens.join("\u0001");
    if (tokenSignature === previousSignature && tokenContainer.children.length === tokens.length) {
        return;
    }

    if (tokens.length < tokenConfidenceHistory.length) {
        tokenConfidenceHistory = tokenConfidenceHistory.slice(0, tokens.length);
    }
    if (tokens.length === 0) {
        tokenConfidenceHistory = [];
    }
    while (tokenConfidenceHistory.length < tokens.length) {
        tokenConfidenceHistory.push(currentStableConfidence(data));
    }
    previousTokens = [...tokens];

    tokenContainer.innerHTML = "";
    tokens.forEach((token, index) => {
        const confidence = tokenConfidenceHistory[index];
        const capsule = document.createElement("div");
        capsule.className = `token-capsule ${confidenceClass(confidence)}`;
        
        const dot = document.createElement("div");
        dot.className = "status-dot";
        
        const text = document.createElement("span");
        text.textContent = token;

        const confidenceText = document.createElement("span");
        confidenceText.className = "token-confidence";
        confidenceText.textContent = Number.isFinite(confidence) ? `${confidence}%` : "--";
        
        capsule.appendChild(dot);
        capsule.appendChild(text);
        capsule.appendChild(confidenceText);
        tokenContainer.appendChild(capsule);
    });
}

function currentStableConfidence(data) {
    const stableConfidence = Number(data.stable && data.stable.confidence);
    if (Number.isFinite(stableConfidence)) {
        return Math.round(stableConfidence * 100);
    }
    const confidence = Number(data.confidence);
    return Number.isFinite(confidence) ? Math.round(confidence) : null;
}

function updateDebug(data) {
    const debug = data.debug || {};
    const stable = data.stable || {};
    const hands = Array.isArray(debug.handedness) ? debug.handedness.join(", ") : "-";
    const stableRatio = Number.isFinite(Number(stable.ratio))
        ? `${Math.round(Number(stable.ratio) * 100)}%`
        : "-";
    const label = data.prediction_label
        ? `${data.prediction_label.id}:${data.prediction_label.code}/${data.prediction_label.group || "-"}`
        : "-";
    const reason = debug.rejected_reason || "none";

    debugStatus.textContent =
        `mode=${data.mode || "-"} | hands=${debug.hands_count ?? 0} (${hands || "-"}) | ` +
        `label=${label} | stable=${stableRatio} | reason=${reason}`;

    renderTopPredictions(data.top_predictions || []);
}

function renderTopPredictions(predictions) {
    topPredictions.innerHTML = "";
    predictions.forEach((prediction, index) => {
        const confidence = Number(prediction.confidence_percent);
        const chip = document.createElement("span");
        chip.className = `prediction-chip ${confidenceClass(confidence)}`;

        const label = document.createElement("span");
        label.textContent = `${index + 1}. ${prediction.code}`;

        const meta = document.createElement("span");
        meta.className = "chip-meta";
        meta.textContent = `${prediction.id} · ${prediction.group || "-"} · ${prediction.confidence_percent}%`;

        chip.appendChild(label);
        chip.appendChild(meta);
        topPredictions.appendChild(chip);
    });
}

function confidenceClass(confidence) {
    if (!Number.isFinite(confidence)) return "medium";
    if (confidence >= 80) return "high";
    if (confidence >= 50) return "medium";
    return "low";
}

clearBtn.addEventListener("click", async () => {
    try {
        const response = await fetch("/api/reset", { method: "POST" });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        updateUI(await response.json());
    } catch (error) {
        console.error("Reset error:", error);
        updateErrorUI();
    }
});

createRoomBtn.addEventListener("click", () => {
    enterRoom(generateRoomCode());
});

joinRoomBtn.addEventListener("click", () => {
    enterRoom(roomCodeInput.value);
});

rejoinRoomBtn.addEventListener("click", () => {
    if (appState.lastNickname) {
        nicknameInput.value = appState.lastNickname;
    }
    if (appState.lastRoomCode) {
        roomCodeInput.value = appState.lastRoomCode;
        enterRoom(appState.lastRoomCode);
    }
});

roomCodeInput.addEventListener("input", () => {
    roomCodeInput.value = normalizeRoomCode(roomCodeInput.value);
});

roomCodeInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
        enterRoom(roomCodeInput.value);
    }
});

nicknameInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
        if (roomCodeInput.value) {
            enterRoom(roomCodeInput.value);
        } else {
            enterRoom(generateRoomCode());
        }
    }
});

leaveBtn.addEventListener("click", leaveRoom);
cameraBtn.addEventListener("click", toggleCameraTrack);
micBtn.addEventListener("click", toggleMicState);

window.addEventListener("DOMContentLoaded", () => {
    setMediaButtonState(micBtn, appState.micEnabled);
    setMediaButtonState(cameraBtn, appState.cameraEnabled);
    setupSpeechRecognition();
});

function setupSpeechRecognition() {

    const SpeechRecognition =
        window.SpeechRecognition ||
        window.webkitSpeechRecognition;

    if (!SpeechRecognition) {
        console.log("STT 미지원 브라우저");
        return;
    }

    recognition = new SpeechRecognition();

    recognition.lang = "ko-KR";
    recognition.continuous = true;
    recognition.interimResults = false;

    recognition.onstart = () => {
        recognizing = true;
        console.log("STT 시작");
    };

    recognition.onend = () => {
        recognizing = false;
        console.log("STT 종료");
    };

    recognition.onresult = (event) => {

        let transcript = "";

        for (let i = event.resultIndex; i < event.results.length; i++) {
            if (!event.results[i].isFinal) {
                continue;
            }
            transcript += event.results[i][0].transcript;
        }

        transcript = transcript.trim();

        if (!transcript) return;

        console.log("STT 결과:", transcript);

        appendCaption({
            type: "stt_result",
            user: appState.nickname || "나",
            role: appState.role,
            text: transcript,
        });
        if (appState.role === "voice") {
            updateResult(transcript);
            statusTitle.textContent = "음성 자막 전송";
            captionStatus.textContent = "음성 인식 중";
        }

        sendSocketMessage({
            type: "stt_result",
            user: appState.nickname,
            role: appState.role,
            text: transcript,
        });
    };
}

function connectWebSocket() {
    if (socket) {
        socket.close();
    }

    const protocol = location.protocol === "https:" ? "wss" : "ws";
    const encodedRoomCode = encodeURIComponent(appState.roomCode);
    const encodedNickname = encodeURIComponent(appState.nickname);
    socket = new WebSocket(
        `${protocol}://${location.host}/ws/${encodedRoomCode}/${encodedNickname}`
    );

    socket.onopen = () => {
        console.log("WebSocket connected");
        connectionStatus.textContent = `${appState.nickname} 접속`;
        sendProfile();
    };
    
    socket.onmessage = (event) => {
        try {
            const message = JSON.parse(event.data);
            handleRealtimeMessage(message);
        } catch (error) {
            console.error("WebSocket message parse error:", error);
        }
    };

    socket.onclose = () => {
        console.log("WebSocket disconnected");
        if (appState.screen === "call") {
            connectionStatus.textContent = "연결 끊김";
            peerConnectionBadge.textContent = "대기";
        }
    };
}
