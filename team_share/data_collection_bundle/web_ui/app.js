import {
  FilesetResolver,
  HandLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

const collectorSelect = document.getElementById("collectorSelect");
const notesInput = document.getElementById("notesInput");
const targetFramesInput = document.getElementById("targetFramesInput");
const intervalInput = document.getElementById("intervalInput");
const labelList = document.getElementById("labelList");
const refreshButton = document.getElementById("refreshButton");
const startButton = document.getElementById("startButton");
const stopButton = document.getElementById("stopButton");
const cameraDeviceSelect = document.getElementById("cameraDeviceSelect");
const cameraRetryButton = document.getElementById("cameraRetryButton");
const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const overlayCtx = overlay.getContext("2d");
const cameraMessage = document.getElementById("cameraMessage");
const engineStatus = document.getElementById("engineStatus");
const serverStatus = document.getElementById("serverStatus");
const shareUrls = document.getElementById("shareUrls");
const saveRoot = document.getElementById("saveRoot");
const secureHint = document.getElementById("secureHint");
const selectedLabelText = document.getElementById("selectedLabelText");
const savedFramesText = document.getElementById("savedFramesText");
const detectionText = document.getElementById("detectionText");
const progressFill = document.getElementById("progressFill");
const sessionText = document.getElementById("sessionText");
const datasetStats = document.getElementById("datasetStats");

const jointList = [
  [0, 1, 2], [1, 2, 3], [2, 3, 4],
  [0, 5, 6], [5, 6, 7], [6, 7, 8],
  [0, 9, 10], [9, 10, 11], [10, 11, 12],
  [0, 13, 14], [13, 14, 15], [14, 15, 16],
  [0, 17, 18], [17, 18, 19], [18, 19, 20],
];
const drawConnections = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
  [5, 9], [9, 13], [13, 17],
];
const fingertipIndices = [4, 8, 12, 16, 20];
const palmBaseIndices = [5, 9, 13, 17];

const processingCanvas = document.createElement("canvas");
const processingCtx = processingCanvas.getContext("2d");

let appConfig = null;
let serverInfo = null;
let selectedLabel = null;
let handLandmarker = null;
let videoReady = false;
let currentSessionId = null;
let savedFrames = 0;
let captureTimer = null;
let countdownTimer = null;
let countdownSeconds = 0;
let frameUploadBusy = false;
let latestDetection = null;
let previewRunning = false;
let lastInferenceTimestamp = 0;
let inferenceIntervalMs = 90;
let mediaStream = null;
let availableCameras = [];
let currentCameraId = "";
let cameraConnectBusy = false;
let currentStats = {};

async function boot() {
  bindEvents();
  await loadConfig();
  await Promise.allSettled([initializeCamera(), setupHandTracker(), healthCheck()]);
  startPreviewLoop();
}

function bindEvents() {
  refreshButton.addEventListener("click", async () => {
    await loadConfig();
  });
  collectorSelect.addEventListener("change", () => {
    window.localStorage.setItem("signCollectorId", collectorSelect.value);
    renderLabels();
  });
  startButton.addEventListener("click", startCaptureSession);
  stopButton.addEventListener("click", () => stopCaptureSession(false));
  cameraRetryButton.addEventListener("click", async () => {
    await connectCamera(cameraDeviceSelect.value || currentCameraId || "");
  });
  cameraDeviceSelect.addEventListener("change", async () => {
    const nextId = cameraDeviceSelect.value;
    if (nextId === currentCameraId) {
      return;
    }
    await connectCamera(nextId);
  });
  video.addEventListener("loadedmetadata", syncOverlaySize);
  window.addEventListener("resize", syncOverlaySize);
}

async function loadConfig() {
  const response = await fetch("/api/config");
  const payload = await response.json();
  appConfig = payload.config;
  currentStats = payload.stats || {};
  serverInfo = payload.serverInfo;
  inferenceIntervalMs = appConfig.preview_inference_ms ?? 90;
  targetFramesInput.value = String(appConfig.target_frames ?? 180);
  intervalInput.value = String(appConfig.capture_interval_ms ?? 220);
  renderCollectorOptions();
  renderServerInfo();
  renderLabels();
  renderStats(currentStats);
}

async function healthCheck() {
  try {
    const response = await fetch("/api/health");
    const payload = await response.json();
    if (payload.ok) {
      serverStatus.textContent = "서버 연결 완료";
      serverStatus.className = "status-value ok";
    }
  } catch (error) {
    serverStatus.textContent = "서버 연결 실패";
    serverStatus.className = "status-value error";
  }
}

function setCameraMessage(message, tone = "warning") {
  if (!cameraMessage) {
    return;
  }
  cameraMessage.textContent = message;
  cameraMessage.className = `camera-message${tone ? ` ${tone}` : ""}`;
}

function hideCameraMessage() {
  if (!cameraMessage) {
    return;
  }
  cameraMessage.className = "camera-message hidden";
}

function localhostUrls() {
  return (serverInfo?.urls || []).filter(
    (url) => url.startsWith("http://127.0.0.1:") || url.startsWith("http://localhost:")
  );
}

function remoteHttpCameraHelp() {
  const localChoices = localhostUrls();
  const localText = localChoices.length ? localChoices.join(" 또는 ") : "http://localhost:8000";
  return `현재 주소 ${location.origin} 에서는 브라우저가 카메라를 막을 수 있습니다. 같은 컴퓨터에서는 ${localText} 로 열고, 다른 기기에서는 HTTPS 주소로 접속하세요.`;
}

function setCameraControlsDisabled(disabled) {
  cameraRetryButton.disabled = disabled;
  cameraDeviceSelect.disabled = disabled;
}

function renderCameraOptions(selectedId = "") {
  cameraDeviceSelect.innerHTML = "";

  if (!availableCameras.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "사용 가능한 카메라 없음";
    cameraDeviceSelect.appendChild(option);
    return;
  }

  for (const [index, device] of availableCameras.entries()) {
    const option = document.createElement("option");
    option.value = device.deviceId;
    option.textContent = device.label || `카메라 ${index + 1}`;
    cameraDeviceSelect.appendChild(option);
  }

  if (selectedId && availableCameras.some((device) => device.deviceId === selectedId)) {
    cameraDeviceSelect.value = selectedId;
  } else {
    cameraDeviceSelect.value = availableCameras[0].deviceId;
  }
}

async function listVideoDevices() {
  if (!navigator.mediaDevices?.enumerateDevices) {
    return [];
  }
  const devices = await navigator.mediaDevices.enumerateDevices();
  return devices.filter((device) => device.kind === "videoinput");
}

async function stopCurrentStream() {
  if (mediaStream) {
    for (const track of mediaStream.getTracks()) {
      track.stop();
    }
  }
  mediaStream = null;
  video.srcObject = null;
  videoReady = false;
}

function cameraErrorMessage(error) {
  if (!navigator.mediaDevices?.getUserMedia) {
    return remoteHttpCameraHelp();
  }

  if (!window.isSecureContext && location.hostname !== "localhost" && location.hostname !== "127.0.0.1") {
    return remoteHttpCameraHelp();
  }

  if (error?.name === "NotAllowedError") {
    return "브라우저에서 카메라 권한이 거부되었습니다. 주소창 왼쪽의 카메라 권한을 허용한 뒤 다시 연결하세요.";
  }

  if (error?.name === "NotFoundError") {
    return "사용 가능한 카메라를 찾지 못했습니다. 다른 앱이 카메라를 사용 중인지 확인하세요.";
  }

  if (error?.name === "NotReadableError") {
    return "카메라가 다른 앱에서 이미 사용 중입니다. 줌, 웹엑스, 카메라 앱을 종료한 뒤 다시 연결하세요.";
  }

  if (error?.name === "OverconstrainedError") {
    return "선택한 카메라를 열지 못했습니다. 다른 카메라를 선택하거나 다시 연결하세요.";
  }

  return error?.message || "카메라 연결에 실패했습니다.";
}

async function refreshCameraDevices(selectedId = "") {
  availableCameras = await listVideoDevices();
  renderCameraOptions(selectedId || currentCameraId);
}

async function initializeCamera() {
  await refreshCameraDevices(window.localStorage.getItem("signCameraId") || "");
  await connectCamera(window.localStorage.getItem("signCameraId") || "");
}

async function connectCamera(deviceId = "") {
  if (cameraConnectBusy) {
    return;
  }

  cameraConnectBusy = true;
  setCameraControlsDisabled(true);
  setCameraMessage("카메라 연결 중입니다.", "warning");

  try {
    await stopCurrentStream();

    const openStream = async (requestedDeviceId) => {
      const videoConstraints = requestedDeviceId
        ? { deviceId: { exact: requestedDeviceId }, width: { ideal: 960 }, height: { ideal: 720 } }
        : { facingMode: "user", width: { ideal: 960 }, height: { ideal: 720 } };

      return navigator.mediaDevices.getUserMedia({
        video: videoConstraints,
        audio: false,
      });
    };

    let stream;
    try {
      stream = await openStream(deviceId);
    } catch (error) {
      if (deviceId) {
        stream = await openStream("");
      } else {
        throw error;
      }
    }

    mediaStream = stream;
    video.srcObject = stream;
    await video.play();
    videoReady = true;
    syncOverlaySize();

    const activeTrack = stream.getVideoTracks()[0];
    currentCameraId = activeTrack?.getSettings?.().deviceId || deviceId || "";
    if (currentCameraId) {
      window.localStorage.setItem("signCameraId", currentCameraId);
    }

    await refreshCameraDevices(currentCameraId);
    hideCameraMessage();
    if (!currentSessionId) {
      detectionText.textContent = "대기 중";
    }
  } catch (error) {
    await stopCurrentStream();
    currentCameraId = "";
    latestDetection = null;
    detectionText.textContent = "카메라 연결 실패";
    setCameraMessage(cameraErrorMessage(error), "error");
  } finally {
    setCameraControlsDisabled(false);
    cameraConnectBusy = false;
  }
}

async function setupHandTracker() {
  try {
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
    );
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
      },
      numHands: 2,
      runningMode: "VIDEO",
      minTrackingConfidence: 0.7,
      minHandDetectionConfidence: 0.7,
      minHandPresenceConfidence: 0.7,
    });
    engineStatus.textContent = "브라우저 추적기 준비 완료";
    engineStatus.className = "status-value ok";
  } catch (error) {
    console.error(error);
    engineStatus.textContent = "손 추적기 로드 실패";
    engineStatus.className = "status-value error";
  }
}

function renderCollectorOptions() {
  const collectors = appConfig?.collectors ?? [];
  const savedCollectorId = window.localStorage.getItem("signCollectorId");
  const currentValue = collectorSelect.value;
  collectorSelect.innerHTML = "";

  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = "수집자를 선택하세요";
  collectorSelect.appendChild(placeholder);

  for (const collector of collectors) {
    const option = document.createElement("option");
    option.value = collector.id;
    option.textContent = collector.name;
    collectorSelect.appendChild(option);
  }

  const preferred = [savedCollectorId, currentValue, collectors[0]?.id].find((value) =>
    collectors.some((collector) => collector.id === value)
  );
  collectorSelect.value = preferred ?? "";
}

function renderServerInfo() {
  if (!serverInfo) {
    return;
  }

  shareUrls.innerHTML = (serverInfo.urls || [])
    .map((url) => `<div>${escapeHtml(url)}</div>`)
    .join("") || "표시할 주소가 없습니다.";
  saveRoot.textContent = serverInfo.saveRoot || "확인 불가";

  const needsSecureContext =
    location.hostname !== "localhost" &&
    location.hostname !== "127.0.0.1" &&
    location.protocol !== "https:";

  if (needsSecureContext || serverInfo.cameraSecureHint) {
    secureHint.textContent =
      "다른 기기에서 카메라를 쓰려면 HTTPS가 필요할 수 있습니다. HTTP에서는 일부 브라우저가 카메라를 막습니다.";
    secureHint.className = "status-note warning";
    if (!window.isSecureContext && location.hostname !== "localhost" && location.hostname !== "127.0.0.1") {
      setCameraMessage(remoteHttpCameraHelp(), "error");
    }
  } else {
    secureHint.textContent = "같은 와이파이에서 접속해도 저장은 이 컴퓨터에만 됩니다.";
    secureHint.className = "status-note";
  }
}

function renderLabels() {
  if (!appConfig) {
    return;
  }

  const selectedCollectorId = collectorSelect.value;
  const activeLabelIds = new Set(appConfig.collection_label_ids || appConfig.labels.map((label) => label.id));
  const labels = appConfig.labels.filter((label) => {
    if (!activeLabelIds.has(label.id)) {
      return false;
    }
    return selectedCollectorId && (label.assignees || []).includes(selectedCollectorId);
  });

  if (selectedLabel && !labels.some((label) => label.id === selectedLabel.id)) {
    selectedLabel = null;
    selectedLabelText.textContent = "미선택";
  }

  labelList.innerHTML = "";

  if (!labels.length) {
    labelList.innerHTML = '<div class="empty-note">수집자를 선택하면 배정된 라벨이 표시됩니다.</div>';
    return;
  }

  for (const label of labels) {
    const stats = statsForLabel(selectedCollectorId, label);
    const card = document.createElement("button");
    card.type = "button";
    card.disabled = Boolean(currentSessionId || countdownTimer);
    card.className = `label-card${selectedLabel && selectedLabel.id === label.id ? " active" : ""}`;
    card.innerHTML = `
      <div class="label-top">
        <span class="label-code">${escapeHtml(label.code)}</span>
        <span class="label-name">${escapeHtml(label.name)}</span>
      </div>
      <div class="label-progress">${stats.frames} 프레임 · ${stats.sessions} 세션 저장</div>
    `;
    card.addEventListener("click", () => {
      selectedLabel = label;
      selectedLabelText.textContent = `${label.code} · ${label.name}`;
      renderLabels();
    });
    labelList.appendChild(card);
  }
}

function statsForLabel(collectorId, label) {
  const value = currentStats?.[collectorId]?.[buildLabelKey(label)] || {};
  if (typeof value === "number") {
    return { frames: value, sessions: 0 };
  }
  return { frames: value.frames || 0, sessions: value.sessions || 0 };
}

function renderStats(stats) {
  datasetStats.innerHTML = "";
  const collectorIds = Object.keys(stats || {});
  if (!collectorIds.length) {
    datasetStats.innerHTML = '<div class="empty-note">아직 저장된 데이터가 없습니다.</div>';
    return;
  }

  for (const collectorId of collectorIds.sort()) {
    const card = document.createElement("div");
    card.className = "stats-card";
    card.innerHTML = `<h3>${escapeHtml(collectorNameById(collectorId))}</h3>`;

    const lines = Object.entries(stats[collectorId])
      .sort((a, b) => a[0].localeCompare(b[0], "ko"))
      .map(([labelKey, value]) => {
        const label = labelByKey(labelKey);
        const labelText = label ? `${label.code} · ${label.name}` : labelKey;
        const frames = typeof value === "number" ? value : value.frames || 0;
        const sessions = typeof value === "number" ? 0 : value.sessions || 0;
        return `<div class="stats-line"><span>${escapeHtml(labelText)}</span><span>${frames} 프레임 · ${sessions} 세션</span></div>`;
      })
      .join("");

    card.innerHTML += lines;
    datasetStats.appendChild(card);
  }
}

function collectorNameById(collectorId) {
  const collector = (appConfig?.collectors || []).find((entry) => entry.id === collectorId);
  return collector ? collector.name : collectorId;
}

function labelByKey(labelKey) {
  return (appConfig?.labels || []).find((label) => buildLabelKey(label) === labelKey);
}

function buildLabelKey(label) {
  return `label_${label.id}_${safeSlug(label.code)}`;
}

function safeSlug(value) {
  return String(value)
    .trim()
    .replace(/[^\p{L}\p{N}_-]+/gu, "_")
    .replace(/^_+|_+$/g, "") || "unknown";
}

function syncOverlaySize() {
  overlay.width = video.clientWidth;
  overlay.height = video.clientHeight;
  processingCanvas.width = video.videoWidth || 960;
  processingCanvas.height = video.videoHeight || 720;
}

function startPreviewLoop() {
  if (previewRunning) {
    return;
  }
  previewRunning = true;
  requestAnimationFrame(runPreviewLoop);
}

function runPreviewLoop(timestamp) {
  if (
    previewRunning &&
    handLandmarker &&
    videoReady &&
    video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA &&
    timestamp - lastInferenceTimestamp >= inferenceIntervalMs
  ) {
    lastInferenceTimestamp = timestamp;
    updateDetection(timestamp);
  }
  requestAnimationFrame(runPreviewLoop);
}

function updateDetection(timestamp) {
  if (!processingCtx || !handLandmarker) {
    return;
  }

  syncOverlaySize();
  processingCtx.save();
  processingCtx.clearRect(0, 0, processingCanvas.width, processingCanvas.height);
  processingCtx.scale(-1, 1);
  processingCtx.drawImage(video, -processingCanvas.width, 0, processingCanvas.width, processingCanvas.height);
  processingCtx.restore();

  const result = handLandmarker.detectForVideo(processingCanvas, timestamp);
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);

  if (!result.landmarks || !result.landmarks.length) {
    latestDetection = {
      state: "no-hand",
      updatedAt: Date.now(),
    };
    if (!currentSessionId) {
      detectionText.textContent = "손 없음";
    }
    return;
  }

  const detectedHands = result.landmarks.map((landmarks, index) => {
    const handedness = extractHandedness(result, index);
    drawLandmarks(landmarks, handedness.label === "right");
    return { landmarks, handedness, features: computeFeatures(landmarks) };
  });
  const rightHand = detectedHands.find((hand) => hand.handedness.label === "right");
  const orderedHands = [...detectedHands].sort((first, second) =>
    first.handedness.label.localeCompare(second.handedness.label)
  );
  const bothHandsFeatures = orderedHands.length >= 2
    ? [...orderedHands[0].features, ...orderedHands[1].features]
    : null;

  latestDetection = {
    updatedAt: Date.now(),
    rightHandFeatures: rightHand?.features || null,
    bothHandsFeatures,
    handCount: detectedHands.length,
  };

  if (!currentSessionId) {
    detectionText.textContent = bothHandsFeatures
      ? "양손 감지"
      : rightHand
        ? `오른손 감지 (${Math.round(rightHand.handedness.score * 100)}%)`
        : "오른손 없음";
  }
}

function extractHandedness(result, index) {
  const category = result.handedness?.[index]?.[0];
  const raw = (category?.categoryName || category?.displayName || "").toLowerCase();
  // Processing input is mirrored to match the OpenCV feature contract.
  // MediaPipe handedness therefore identifies the opposite physical hand.
  const label = raw === "right" ? "left" : "right";
  return {
    label,
    score: Number(category?.score || 0),
  };
}

async function startCaptureSession() {
  if (!videoReady || !handLandmarker) {
    alert("카메라 또는 손 추적기가 아직 준비되지 않았습니다.");
    return;
  }

  const collector = selectedCollector();
  if (!collector) {
    alert("수집자를 먼저 선택하세요.");
    return;
  }

  if (!selectedLabel) {
    alert("수집할 라벨을 선택하세요.");
    return;
  }

  countdownSeconds = 5;
  startButton.disabled = true;
  stopButton.disabled = false;
  collectorSelect.disabled = true;
  refreshButton.disabled = true;
  renderLabels();
  updateCountdownDisplay();
  countdownTimer = window.setInterval(async () => {
    countdownSeconds -= 1;
    if (countdownSeconds > 0) {
      updateCountdownDisplay();
      return;
    }
    window.clearInterval(countdownTimer);
    countdownTimer = null;
    await beginCaptureSession(collector);
  }, 1000);
  renderLabels();
}

function updateCountdownDisplay() {
  const requiredHands = selectedLabel.collection_mode === "both_hands" ? "양손 자세" : "오른손 자세";
  detectionText.textContent = `${countdownSeconds}초 후 저장 시작`;
  sessionText.textContent = `${selectedLabel.code} ${requiredHands}를 준비하세요.`;
}

async function beginCaptureSession(collector) {
  const response = await fetch("/api/session/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      collectorId: collector.id,
      collectorName: collector.name,
      notes: notesInput.value.trim(),
      labelId: selectedLabel.id,
      labelName: selectedLabel.name,
      labelCode: selectedLabel.code,
    }),
  });
  const payload = await response.json();
  if (!response.ok) {
    startButton.disabled = false;
    stopButton.disabled = true;
    collectorSelect.disabled = false;
    refreshButton.disabled = false;
    renderLabels();
    detectionText.textContent = "세션 시작 실패";
    sessionText.textContent = payload.error || "서버에서 수집 세션을 만들지 못했습니다.";
    return;
  }

  currentSessionId = payload.sessionId;
  savedFrames = 0;
  savedFramesText.textContent = "0";
  progressFill.style.width = "0%";
  sessionText.textContent = payload.csvPath;
  detectionText.textContent = selectedLabel.collection_mode === "both_hands"
    ? "양손 대기 중"
    : "오른손 대기 중";
  startButton.disabled = true;
  stopButton.disabled = false;

  const intervalMs = Number(appConfig.capture_interval_ms);
  captureTimer = window.setInterval(captureFrame, intervalMs);
}

async function stopCaptureSession(completed = false) {
  if (countdownTimer) {
    window.clearInterval(countdownTimer);
    countdownTimer = null;
    countdownSeconds = 0;
    startButton.disabled = false;
    stopButton.disabled = true;
    collectorSelect.disabled = false;
    refreshButton.disabled = false;
    detectionText.textContent = "준비 취소";
    sessionText.textContent = "수집 시작 전 취소되어 저장된 프레임이 없습니다.";
    renderLabels();
    return;
  }

  if (!currentSessionId) {
    return;
  }

  window.clearInterval(captureTimer);
  captureTimer = null;

  const sessionId = currentSessionId;
  currentSessionId = null;

  await fetch("/api/session/end", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sessionId }),
  });

  startButton.disabled = false;
  stopButton.disabled = true;
  collectorSelect.disabled = false;
  refreshButton.disabled = false;
  detectionText.textContent = completed ? "목표 수량 저장 완료" : "수집 중지";
  const completedLabel = selectedLabel;
  await loadConfig();
  if (completed) {
    const nextLabel = selectNextLabel(completedLabel);
    sessionText.textContent = nextLabel
      ? `${completedLabel.code} 저장 완료. 다음 작업 ${nextLabel.code} 자세를 준비한 뒤 수집을 시작하세요.`
      : `${completedLabel.code} 저장 완료. 이 담당자의 작업 큐 마지막 라벨입니다.`;
  } else {
    sessionText.textContent = "세션이 저장되었습니다.";
  }
}

async function captureFrame() {
  if (!currentSessionId || frameUploadBusy) {
    return;
  }

  if (!latestDetection || Date.now() - latestDetection.updatedAt > 1200) {
    detectionText.textContent = "손 재인식 중";
    return;
  }

  const requiresBothHands = selectedLabel.collection_mode === "both_hands";
  const features = requiresBothHands ? latestDetection.bothHandsFeatures : latestDetection.rightHandFeatures;
  if (!features) {
    detectionText.textContent = requiresBothHands ? "양손을 모두 보여주세요" : "오른손만 저장합니다";
    return;
  }

  frameUploadBusy = true;
  try {
    const row = [selectedLabel.id, ...features];
    const response = await fetch("/api/session/frame", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sessionId: currentSessionId,
        row,
      }),
    });
    const payload = await response.json();
    savedFrames = payload.framesSaved;
    savedFramesText.textContent = String(savedFrames);
    detectionText.textContent = requiresBothHands ? "양손 저장 중" : "오른손 저장 중";

    const target = Number(appConfig.target_frames);
    const ratio = Math.min(100, (savedFrames / target) * 100);
    progressFill.style.width = `${ratio}%`;

    if (payload.complete || savedFrames >= target) {
      await stopCaptureSession(true);
    }
  } catch (error) {
    console.error(error);
    detectionText.textContent = "저장 오류";
  } finally {
    frameUploadBusy = false;
  }
}

function selectedCollector() {
  return (appConfig?.collectors || []).find((collector) => collector.id === collectorSelect.value) || null;
}

function selectNextLabel(completedLabel) {
  const collectorId = collectorSelect.value;
  const allowedIds = new Set(appConfig.collection_label_ids || []);
  const labels = appConfig.labels.filter(
    (label) => allowedIds.has(label.id) && (label.assignees || []).includes(collectorId)
  );
  const completedIndex = labels.findIndex((label) => label.id === completedLabel.id);
  const nextLabel = labels[completedIndex + 1] || null;
  if (nextLabel) {
    selectedLabel = nextLabel;
    selectedLabelText.textContent = `${nextLabel.code} · ${nextLabel.name}`;
    renderLabels();
  }
  return nextLabel;
}

function drawLandmarks(landmarks, isRightHand) {
  const width = overlay.width;
  const height = overlay.height;
  overlayCtx.lineWidth = 2;
  overlayCtx.strokeStyle = isRightHand ? "#70e1c8" : "#ff807a";
  overlayCtx.fillStyle = isRightHand ? "#eef2ff" : "#ffd3d0";

  for (const [a, b] of drawConnections) {
    overlayCtx.beginPath();
    overlayCtx.moveTo(landmarks[a].x * width, landmarks[a].y * height);
    overlayCtx.lineTo(landmarks[b].x * width, landmarks[b].y * height);
    overlayCtx.stroke();
  }

  for (const point of landmarks) {
    overlayCtx.beginPath();
    overlayCtx.arc(point.x * width, point.y * height, 4, 0, Math.PI * 2);
    overlayCtx.fill();
  }
}

function computeFeatures(landmarks) {
  const joint = landmarks.map((point) => [point.x, point.y, point.z]);
  const angles = computeAngles(joint).map((value) => value / 180.0);
  const wrist = joint[0];
  const relativeJoint = joint.map((point) => subtract(point, wrist));
  const palmScale =
    mean(palmBaseIndices.map((index) => norm(subtract(joint[index], wrist)))) || 1e-6;

  const normalizedJoint = relativeJoint.flatMap((point) =>
    point.map((value) => value / palmScale)
  );
  const fingertipDistances = fingertipIndices.map(
    (index) => norm(subtract(joint[index], wrist)) / palmScale
  );

  return [...angles, ...normalizedJoint, ...fingertipDistances];
}

function computeAngles(joint) {
  return jointList.map(([a, b, c]) => {
    const vec1 = normalize(subtract(joint[a], joint[b]));
    const vec2 = normalize(subtract(joint[c], joint[b]));
    const dot = clamp(dotProduct(vec1, vec2), -1, 1);
    return (Math.acos(dot) * 180) / Math.PI;
  });
}

function subtract(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function norm(vector) {
  return Math.sqrt(dotProduct(vector, vector));
}

function normalize(vector) {
  const length = norm(vector) || 1e-6;
  return vector.map((value) => value / length);
}

function dotProduct(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function mean(values) {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

boot().catch((error) => {
  console.error(error);
  engineStatus.textContent = "앱 초기화 실패";
  engineStatus.className = "status-value error";
  detectionText.textContent = "카메라 초기화 실패";
  if (error?.message) {
    secureHint.textContent = error.message;
    secureHint.className = "status-note warning";
    setCameraMessage(error.message, "error");
  }
});
