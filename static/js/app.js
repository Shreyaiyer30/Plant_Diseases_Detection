/**
 * PlantCure v2 — app.js
 * ======================
 * Camera Technology Used:
 *   - navigator.mediaDevices.getUserMedia()  →  WebRTC W3C API
 *     Accesses device camera, returns a live MediaStream
 *   - HTMLMediaElement (<video>)             →  Displays the live stream
 *   - HTML5 Canvas API                       →  Grabs one frame on capture
 *   - Canvas.toBlob()                        →  Converts frame to JPG binary
 *   - Laplacian Variance Algorithm           →  Detects image blur
 *   - ImageCapture API (where available)     →  Higher quality capture
 *
 * Full flow:
 *   Camera → getUserMedia() → <video> live preview
 *   → snap button → Canvas.drawImage() → toBlob() → File
 *   → FormData POST /api/predict → CNN → JSON result → UI
 */

"use strict";

// ═══════════════════════════════════════════════════════════════════
// ELEMENT REFERENCES
// ═══════════════════════════════════════════════════════════════════
const fileInput     = document.getElementById("fileInput");
const browseBtn     = document.getElementById("browseBtn");
const cameraBtn     = document.getElementById("cameraBtn");
const dropZone      = document.getElementById("dropZone");
const previewZone   = document.getElementById("previewZone");
const previewImg    = document.getElementById("previewImg");
const previewName   = document.getElementById("previewNm") || document.getElementById("previewName");
const removeBtn     = document.getElementById("removeBtn");
const analyzeBtn    = document.getElementById("analyzeBtn");
const btnLabel      = document.getElementById("btnLabel");
const btnSpinner    = document.getElementById("btnSpin") || document.getElementById("btnSpinner");

const resultCard    = document.getElementById("resultCard");
const resultHead    = document.getElementById("resultHead");
const resultEmoji   = document.getElementById("resIco") || document.getElementById("resultEmoji");
const resultDis     = document.getElementById("resDis") || document.getElementById("resultDisease");
const badgeStatus   = document.getElementById("badgeSt") || document.getElementById("badgeStatus");
const badgeSeverity = document.getElementById("badgeSev") || document.getElementById("badgeSeverity");
const ringFg        = document.getElementById("ringFg");
const confNum       = document.getElementById("confNum");
const resultDesc    = document.getElementById("resDesc") || document.getElementById("resultDesc");
const plantProfileBlk = document.getElementById("plantProfileBlk");
const plantAbout = document.getElementById("plantAbout");
const plantIssues = document.getElementById("plantIssues");
const plantLight = document.getElementById("plantLight");
const plantWater = document.getElementById("plantWater");
const plantPropagation = document.getElementById("plantPropagation");
const resultImageBlk = document.getElementById("resultImageBlk");
const resultPreviewImg = document.getElementById("resultPreviewImg");
const listSymptoms  = document.getElementById("listSym") || document.getElementById("listSymptoms");
const listTreatment = document.getElementById("listTreat") || document.getElementById("listTreatment");
const listPrevent   = document.getElementById("listPrev") || document.getElementById("listPrevention");
const topPredBlk    = document.getElementById("topPredBlk");
const listTopPred   = document.getElementById("listTopPred");
const warningBlk    = document.getElementById("warningBlk");
const listWarnings  = document.getElementById("listWarnings");
const demoNotice    = document.getElementById("demoNotice");
const tryAgainBtn   = document.getElementById("tryAgain") || document.getElementById("tryAgainBtn");
const saveBtn       = document.getElementById("saveBtn");

const errorCard     = document.getElementById("errorCard");
const errorMsg      = document.getElementById("errorMsg");
const retryBtn      = document.getElementById("retryBtn");

// Camera modal elements
const cameraModal   = document.getElementById("camModal") || document.getElementById("cameraModal");
const camVideo      = document.getElementById("camVideo");
const camCanvas     = document.getElementById("camCanvas");
const snapBtn       = document.getElementById("snapBtn");
const closeCamBtn   = document.getElementById("closeCam") || document.getElementById("closeCamBtn");
const cancelCamBtn  = document.getElementById("cancelCam") || document.getElementById("cancelCamBtn");
const switchCamBtn  = document.getElementById("switchCam") || document.getElementById("switchCamBtn");
const torchBtn      = document.getElementById("torchBtn");
const camStatus     = document.getElementById("camStatus");
const qualityBar    = document.getElementById("qFill") || document.getElementById("qualityBar");
const qualityLabel  = document.getElementById("qLabel") || document.getElementById("qualityLabel");
const countdownEl   = document.getElementById("cdEl") || document.getElementById("countdown");

// Chat elements
const chatSection   = document.getElementById("chatSection");
const chatMsgs      = document.getElementById("chatMsgs");
const chatInput     = document.getElementById("chatInput");
const sendBtn       = document.getElementById("sendBtn");

// ═══════════════════════════════════════════════════════════════════
// STATE
// ═══════════════════════════════════════════════════════════════════
let selectedFile    = null;
let previewObjectUrl = null;
let lastResult      = null;
let cameraStream    = null;         // active MediaStream
let facingMode      = "environment"; // "environment"=rear, "user"=front
let torchOn         = false;
let torchSupported  = false;
let qualityInterval = null;         // timer for live quality check
let isCapturing     = false;
let videoDevices    = [];
let currentDeviceIndex = 0;

if (previewZone && resultCard && previewZone.parentElement === resultCard.parentElement) {
  previewZone.insertAdjacentElement("afterend", resultCard);
}
let frameGuardInterval = null;
let autoSwitchedForBlackFeed = false;

// ═══════════════════════════════════════════════════════════════════
// FILE BROWSE & DRAG-DROP
// ═══════════════════════════════════════════════════════════════════
browseBtn?.addEventListener("click", () => fileInput?.click());
fileInput?.addEventListener("change", e => handleFile(e.target.files[0]));

dropZone?.addEventListener("dragover", e => {
  e.preventDefault();
  dropZone.classList.add("over");
});
dropZone?.addEventListener("dragleave", () => {
  dropZone.classList.remove("over");
});
dropZone?.addEventListener("drop", e => {
  e.preventDefault();
  dropZone.classList.remove("over");
  handleFile(e.dataTransfer.files[0]);
});

function handleFile(file) {
  if (!file) return;

  const allowed = ["image/png", "image/jpeg", "image/jpg", "image/webp"];
  if (!allowed.includes(file.type)) {
    showError("Unsupported file type. Please use PNG, JPG, or JPEG.");
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    showError("File too large. Maximum allowed size is 10 MB.");
    return;
  }

  selectedFile = file;
  if (previewObjectUrl) {
    URL.revokeObjectURL(previewObjectUrl);
    previewObjectUrl = null;
  }
  previewObjectUrl = URL.createObjectURL(file);
  previewImg.src = previewObjectUrl;
  previewImg.alt = "Selected leaf preview";
  previewImg.onerror = () => {
    showError("Could not show this image in the browser. Try uploading with Browse File (JPG/PNG).");
  };
  previewName.textContent = `${file.name}  (${(file.size / 1024).toFixed(1)} KB)`;
  dropZone.classList.add("hidden");
  previewZone.classList.remove("hidden");
  hideCards();
}

async function optimizeImageForUpload(file) {
  if (!file || !file.type.startsWith("image/")) return file;
  if (file.size <= 1024 * 1024) return file;

  return new Promise((resolve) => {
    const img = new Image();
    const objectUrl = URL.createObjectURL(file);

    img.onload = () => {
      const maxSide = 1280;
      const scale = Math.min(1, maxSide / Math.max(img.width, img.height));
      const canvas = document.createElement("canvas");
      canvas.width = Math.max(1, Math.round(img.width * scale));
      canvas.height = Math.max(1, Math.round(img.height * scale));

      const ctx = canvas.getContext("2d");
      if (!ctx) {
        URL.revokeObjectURL(objectUrl);
        resolve(file);
        return;
      }

      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      canvas.toBlob((blob) => {
        URL.revokeObjectURL(objectUrl);
        if (!blob) {
          resolve(file);
          return;
        }
        const optimized = new File([blob], file.name.replace(/\.\w+$/, ".jpg"), {
          type: "image/jpeg",
          lastModified: Date.now()
        });
        resolve(optimized.size < file.size ? optimized : file);
      }, "image/jpeg", 0.82);
    };

    img.onerror = () => {
      URL.revokeObjectURL(objectUrl);
      resolve(file);
    };

    img.src = objectUrl;
  });
}

// ═══════════════════════════════════════════════════════════════════
// RESET
// ═══════════════════════════════════════════════════════════════════
[removeBtn, tryAgainBtn, retryBtn].forEach(b => b?.addEventListener("click", reset));

function reset() {
  selectedFile = null;
  fileInput.value = "";
  if (previewObjectUrl) {
    URL.revokeObjectURL(previewObjectUrl);
    previewObjectUrl = null;
  }
  previewImg.src = "";
  previewImg.alt = "";
  previewImg.onerror = null;
  previewZone.classList.add("hidden");
  dropZone.classList.remove("hidden");
  hideCards();
  hideCards();
  const openChatBtn = document.getElementById('open-chatbot');
  if (openChatBtn) openChatBtn.style.display = 'none';
  const chatbotWidget = document.getElementById('chatbot-widget');
  if (chatbotWidget) chatbotWidget.style.display = 'none';
}
function hideCards() {
  resultCard.classList.add("hidden");
  errorCard.classList.add("hidden");
}

// ═══════════════════════════════════════════════════════════════════
// CAMERA — CORE
// ═══════════════════════════════════════════════════════════════════

/**
 * Opens the camera modal and starts the live video stream.
 *
 * TECHNOLOGY: navigator.mediaDevices.getUserMedia(constraints)
 *   - Part of the WebRTC specification (W3C standard)
 *   - Prompts user for camera permission on first use
 *   - Returns a Promise<MediaStream>
 *   - MediaStream is piped into <video> element for live display
 *
 * CONSTRAINTS used:
 *   facingMode: "environment"  → rear/back camera (best for leaf photos)
 *   facingMode: "user"         → front/selfie camera
 *   width/height               → request HD resolution from camera
 */
cameraBtn?.addEventListener("click", openCamera);
[closeCamBtn, cancelCamBtn].forEach(b => b?.addEventListener("click", closeCamera));
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && cameraModal && !cameraModal.classList.contains("hidden")) {
    closeCamera();
  }
});

async function openCamera() {
  // Check if browser supports the camera API
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showError(
      "Camera not supported in this browser. " +
      "Try Chrome or Firefox on a phone or laptop."
    );
    return;
  }
  if (location.protocol !== "https:" &&
      location.hostname !== "localhost" &&
      location.hostname !== "127.0.0.1") {
    showError("Camera requires HTTPS in production. Use a secure URL.");
    return;
  }

  setCamStatus("Starting camera...", "info");
  cameraModal.classList.remove("hidden");

  await startStream(facingMode);
}

async function startStream(facing) {
  // Stop any existing stream before starting a new one
  stopStream();
  await loadVideoDevices();

  const selectedDevice = videoDevices[currentDeviceIndex];
  const constraints = {
    video: selectedDevice
      ? {
          deviceId: { exact: selectedDevice.deviceId },
          width: { ideal: 1920 },
          height: { ideal: 1080 }
        }
      : {
          facingMode: { ideal: facing },  // mobile fallback
          width: { ideal: 1920 },
          height: { ideal: 1080 },
        },
    audio: false, // we don't need audio
  };

  try {
    // getUserMedia() triggers browser permission prompt
    // Returns MediaStream object with video track(s)
    cameraStream = await navigator.mediaDevices.getUserMedia(constraints);

    // Pipe the live stream into the <video> element
    camVideo.srcObject = cameraStream;

    // Wait for video metadata to load (so we know resolution)
    await new Promise(resolve => {
      camVideo.onloadedmetadata = () => {
        camVideo.play().catch(() => {});
        resolve();
      };
    });
    await waitForVideoFrame();

    // Check camera capabilities (torch/flash support)
    await checkTorchSupport();

    const track = cameraStream.getVideoTracks()[0];
    const settings = track.getSettings();
    setCamStatus(
      `Camera ready  ${settings.width || ""}×${settings.height || ""}`,
      "success"
    );

    // Start live image quality monitoring
    startQualityMonitor();
    startFrameGuard();

  } catch (err) {
    if (err.name === "OverconstrainedError" || err.name === "NotFoundError") {
      try {
        cameraStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        camVideo.srcObject = cameraStream;
        await new Promise(resolve => {
          camVideo.onloadedmetadata = () => {
            camVideo.play().catch(() => {});
            resolve();
          };
        });
        await waitForVideoFrame();
        await checkTorchSupport();
        startQualityMonitor();
        startFrameGuard();
        setCamStatus("Camera ready", "success");
      } catch (fallbackErr) {
        handleCameraError(fallbackErr, facing);
      }
    } else {
      handleCameraError(err, facing);
    }
  }
}

async function waitForVideoFrame(timeoutMs = 2500) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    if (camVideo.readyState >= 2 && camVideo.videoWidth > 0 && camVideo.videoHeight > 0) {
      // Let one paint cycle complete before capture.
      await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
      return true;
    }
    await new Promise(r => setTimeout(r, 60));
  }
  return false;
}

async function loadVideoDevices() {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    videoDevices = devices.filter(d => d.kind === "videoinput");
    if (videoDevices.length > 0 && currentDeviceIndex >= videoDevices.length) {
      currentDeviceIndex = 0;
    }
  } catch (e) {
    videoDevices = [];
  }
}

function frameLooksValid() {
  if (!camVideo || camVideo.readyState < 2 || !camVideo.videoWidth || !camVideo.videoHeight) {
    return false;
  }
  const c = document.createElement("canvas");
  c.width = 64;
  c.height = 48;
  const ctx = c.getContext("2d");
  if (!ctx) return false;
  ctx.drawImage(camVideo, 0, 0, c.width, c.height);
  const data = ctx.getImageData(0, 0, c.width, c.height).data;
  let sum = 0;
  let nonBlack = 0;
  for (let i = 0; i < data.length; i += 4) {
    const y = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
    sum += y;
    if (y > 8) nonBlack++;
  }
  const avg = sum / (data.length / 4);
  return avg > 8 && nonBlack > 50;
}

function startFrameGuard() {
  stopFrameGuard();
  autoSwitchedForBlackFeed = false;
  let badFrames = 0;
  frameGuardInterval = setInterval(async () => {
    if (!cameraStream || cameraModal.classList.contains("hidden")) return;
    if (frameLooksValid()) {
      badFrames = 0;
      return;
    }
    badFrames++;
    if (badFrames >= 4) {
      if (!autoSwitchedForBlackFeed && videoDevices.length > 1) {
        autoSwitchedForBlackFeed = true;
        badFrames = 0;
        currentDeviceIndex = (currentDeviceIndex + 1) % videoDevices.length;
        setCamStatus("Camera feed is black. Switching camera...", "error");
        await startStream(facingMode);
      } else {
        setCamStatus("No real camera image. Try Switch, then allow camera in Windows privacy.", "error");
      }
    }
  }, 1000);
}

function stopFrameGuard() {
  if (frameGuardInterval) {
    clearInterval(frameGuardInterval);
    frameGuardInterval = null;
  }
}

function stopStream() {
  if (cameraStream) {
    cameraStream.getTracks().forEach(t => t.stop());
    cameraStream = null;
  }
  stopQualityMonitor();
  stopFrameGuard();
}

function closeCamera() {
  stopStream();
  torchOn = false;
  torchSupported = false;
  if (torchBtn) {
    torchBtn.textContent = "🔦";
    torchBtn.classList.remove("active");
  }
  camVideo.srcObject = null;
  cameraModal.classList.add("hidden");
}

/**
 * Handle camera open errors gracefully.
 * Common errors:
 *   NotAllowedError  → user denied permission
 *   NotFoundError    → no camera found on device
 *   NotReadableError → camera in use by another app
 *   OverconstrainedError → requested resolution not supported
 */
function handleCameraError(err, facing) {
  console.error("Camera error:", err.name, err.message);

  if (err.name === "NotAllowedError") {
    setCamStatus("Permission denied. Allow camera access in browser settings.", "error");
    showError(
      "Camera access was denied.\n" +
      "Fix: Click the 🔒 lock icon in your browser address bar → Allow Camera."
    );
  } else if (err.name === "NotFoundError") {
    setCamStatus("No camera found on this device.", "error");
    showError("No camera found. Use 'Browse File' to upload an image instead.");
    closeCamera();
  } else if (err.name === "NotReadableError") {
    setCamStatus("Camera is in use by another app.", "error");
    showError("Camera is busy. Close other apps using the camera and try again.");
  } else if (err.name === "OverconstrainedError" && facing === "environment") {
    // Rear camera not available — try front camera as fallback
    console.warn("Rear camera failed, trying front camera...");
    setCamStatus("Rear camera unavailable, switching to front...", "info");
    facingMode = "user";
    startStream("user");
  } else {
    setCamStatus(`Camera error: ${err.message}`, "error");
    showError(`Camera error: ${err.name}. Try refreshing the page.`);
  }
}

// ═══════════════════════════════════════════════════════════════════
// FRONT / REAR CAMERA TOGGLE
// ═══════════════════════════════════════════════════════════════════

/**
 * Switches between front (selfie) and rear camera.
 * Used on mobile phones that have both cameras.
 * TECHNOLOGY: Restarts getUserMedia with opposite facingMode constraint.
 */
switchCamBtn?.addEventListener("click", async () => {
  await loadVideoDevices();
  if (videoDevices.length > 1) {
    currentDeviceIndex = (currentDeviceIndex + 1) % videoDevices.length;
    setCamStatus(`Switching camera ${currentDeviceIndex + 1}/${videoDevices.length}...`, "info");
  } else {
    facingMode = facingMode === "environment" ? "user" : "environment";
    const label = facingMode === "environment" ? "Rear Camera" : "Front Camera";
    setCamStatus(`Switching to ${label}...`, "info");
  }
  await startStream(facingMode);
});

// ═══════════════════════════════════════════════════════════════════
// TORCH / FLASHLIGHT
// ═══════════════════════════════════════════════════════════════════

/**
 * TECHNOLOGY: MediaStreamTrack.applyConstraints({ advanced: [{ torch: true }] })
 *   - Part of the ImageCapture API spec
 *   - Only works on Android Chrome with rear camera
 *   - Not supported on iOS Safari or desktop browsers
 */
async function checkTorchSupport() {
  torchSupported = false;
  if (!cameraStream) return;

  const track = cameraStream.getVideoTracks()[0];
  if (!track) return;

  const capabilities = track.getCapabilities?.() || {};
  torchSupported = !!capabilities.torch;

  if (torchBtn) {
    torchBtn.style.display = torchSupported ? "inline-flex" : "none";
  }
}

torchBtn?.addEventListener("click", async () => {
  if (!torchSupported || !cameraStream) return;

  const track = cameraStream.getVideoTracks()[0];
  torchOn = !torchOn;

  try {
    await track.applyConstraints({ advanced: [{ torch: torchOn }] });
    torchBtn.textContent = torchOn ? "💡" : "🔦";
    torchBtn.classList.toggle("active", torchOn);
  } catch (e) {
    console.warn("Torch control failed:", e);
  }
});

// ═══════════════════════════════════════════════════════════════════
// LIVE IMAGE QUALITY CHECK (Laplacian Variance Algorithm)
// ═══════════════════════════════════════════════════════════════════

/**
 * ALGORITHM: Laplacian Variance for Blur Detection
 *
 * How it works:
 *   1. Sample the live video frame onto a small canvas (100×75px)
 *   2. Convert to greyscale using luminance formula:
 *        grey = 0.299*R + 0.587*G + 0.114*B
 *   3. Apply Laplacian kernel (edge detector):
 *        [ 0, -1,  0]
 *        [-1,  4, -1]
 *        [ 0, -1,  0]
 *      This kernel computes the second derivative of image intensity.
 *      Sharp edges → large values. Blurry areas → values near zero.
 *   4. Compute variance of all Laplacian values:
 *        variance = mean(laplacian²) - mean(laplacian)²
 *   5. Low variance (<50)  = BLURRY  (no edges detected)
 *      High variance (>150) = SHARP   (many clear edges)
 *
 * Why Laplacian?
 *   - Fast: runs every 500ms without slowing the browser
 *   - No external library needed — pure canvas pixel math
 *   - Reliable indicator of image quality for CNN input
 */
const QUALITY_CANVAS = document.createElement("canvas");
const QUALITY_CTX    = QUALITY_CANVAS.getContext("2d");
QUALITY_CANVAS.width  = 100;  // small sample size for speed
QUALITY_CANVAS.height = 75;

function computeBlurScore() {
  if (!camVideo || camVideo.readyState < 2) return 0;

  // Step 1: Draw current video frame to small canvas
  QUALITY_CTX.drawImage(camVideo, 0, 0, 100, 75);
  const imageData = QUALITY_CTX.getImageData(0, 0, 100, 75);
  const pixels    = imageData.data;  // flat array: [R,G,B,A, R,G,B,A, ...]

  const W = 100, H = 75;

  // Step 2: Convert to greyscale using luminance weights
  const grey = new Float32Array(W * H);
  for (let i = 0; i < W * H; i++) {
    const r = pixels[i * 4];
    const g = pixels[i * 4 + 1];
    const b = pixels[i * 4 + 2];
    grey[i] = 0.299 * r + 0.587 * g + 0.114 * b;
  }

  // Step 3: Apply Laplacian kernel (edge detector)
  // Skip border pixels (1px margin) to avoid index errors
  let sum = 0, sumSq = 0, count = 0;
  for (let y = 1; y < H - 1; y++) {
    for (let x = 1; x < W - 1; x++) {
      const idx = y * W + x;
      // Laplacian: center × 4 minus 4 neighbours
      const lap = (
          4 * grey[idx]
        -     grey[idx - 1]      // left
        -     grey[idx + 1]      // right
        -     grey[idx - W]      // above
        -     grey[idx + W]      // below
      );
      sum   += lap;
      sumSq += lap * lap;
      count++;
    }
  }

  // Step 4: Compute variance
  const mean     = sum / count;
  const variance = (sumSq / count) - (mean * mean);
  return Math.max(0, variance);  // variance cannot be negative
}

function startQualityMonitor() {
  stopQualityMonitor();
  qualityInterval = setInterval(updateQualityBar, 500);  // check every 500ms
}

function stopQualityMonitor() {
  if (qualityInterval) {
    clearInterval(qualityInterval);
    qualityInterval = null;
  }
}

function updateQualityBar() {
  if (!qualityBar || !qualityLabel) return;

  const score = computeBlurScore();

  // Map variance score to a 0-100% quality percentage
  // <50 = blurry, 50-150 = fair, >150 = sharp
  const pct = Math.min(100, Math.round(score / 2));

  qualityBar.style.width = pct + "%";

  if (pct < 25) {
    // Very blurry
    qualityBar.style.background = "#ef4444";
    qualityLabel.textContent = "Very Blurry — Move closer or hold steady";
    qualityLabel.style.color = "#ef4444";
    snapBtn.disabled = false;  // still allow capture but warn
  } else if (pct < 50) {
    // Blurry
    qualityBar.style.background = "#f97316";
    qualityLabel.textContent = "Blurry — Try to hold phone still";
    qualityLabel.style.color = "#f97316";
    snapBtn.disabled = false;
  } else if (pct < 70) {
    // Fair
    qualityBar.style.background = "#eab308";
    qualityLabel.textContent = "Fair Quality — Hold steady";
    qualityLabel.style.color = "#eab308";
    snapBtn.disabled = false;
  } else {
    // Good / Sharp
    qualityBar.style.background = "#22c55e";
    qualityLabel.textContent = "Good Quality — Ready to capture!";
    qualityLabel.style.color = "#22c55e";
    snapBtn.disabled = false;
  }
}

// ═══════════════════════════════════════════════════════════════════
// CAPTURE WITH COUNTDOWN
// ═══════════════════════════════════════════════════════════════════

/**
 * TECHNOLOGY: HTML5 Canvas API — captureFrame()
 *
 * How it works:
 *   1. User clicks "Capture Photo" button
 *   2. 3-second countdown gives user time to steady the phone
 *   3. On 0: canvas.drawImage(videoElement) copies the CURRENT
 *      live frame from the <video> element to the canvas
 *   4. canvas.toBlob() converts the canvas pixel data to a
 *      JPG binary blob (same format as a photo file)
 *   5. new File([blob], name) wraps it as a File object
 *   6. This File goes through the exact same pipeline as an
 *      uploaded photo — no special code needed downstream
 *
 * Canvas resolution:
 *   We set canvas width/height to the actual video resolution
 *   (e.g. 1920×1080 for HD camera) to get maximum quality.
 *   JPEG quality is set to 0.92 (92%) — good balance of
 *   quality vs file size.
 */
snapBtn?.addEventListener("click", () => {
  if (isCapturing) return;
  startCountdownCapture();
});

function startCountdownCapture() {
  isCapturing = true;
  snapBtn.disabled = true;
  let count = 3;

  if (countdownEl) {
    countdownEl.textContent = count;
    countdownEl.classList.remove("hidden");
  }

  const timer = setInterval(() => {
    count--;
    if (countdownEl) countdownEl.textContent = count > 0 ? count : "📸";

    if (count <= 0) {
      clearInterval(timer);
      setTimeout(() => {
        captureFrame();
        if (countdownEl) countdownEl.classList.add("hidden");
        isCapturing = false;
        snapBtn.disabled = false;
      }, 200);
    }
  }, 1000);
}

function captureFrame() {
  if (!camVideo || camVideo.readyState < 2 || !camVideo.videoWidth || !camVideo.videoHeight) {
    showError("Camera frame not ready. Wait until the live preview is visible, then capture again.");
    return;
  }

  const vw = camVideo.videoWidth;
  const vh = camVideo.videoHeight;
  camCanvas.width  = vw;
  camCanvas.height = vh;

  const ctx = camCanvas.getContext("2d", { willReadFrequently: false });
  if (!ctx) {
    showError("Unable to access capture canvas.");
    return;
  }

  if (ctx.resetTransform) ctx.resetTransform();
  else ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, vw, vh);

  const track = cameraStream?.getVideoTracks?.()[0];
  const settings = track?.getSettings?.() || {};
  const isFront = settings.facingMode === "user" || facingMode === "user";

  if (isFront) {
    ctx.translate(vw, 0);
    ctx.scale(-1, 1);
  }

  try {
    ctx.drawImage(camVideo, 0, 0, vw, vh);
  } catch (e) {
    console.error(e);
    showError("Could not read camera frame. Try another browser or use Browse File.");
    return;
  }

  if (!frameLooksValid()) {
    showError("Camera feed is black. Click Switch camera, allow camera access in Windows, and try again.");
    return;
  }

  camCanvas.toBlob(
    (blob) => {
      if (!blob || blob.size < 400) {
        showError("Capture was too small or empty. Allow camera fully, add light, and try again.");
        return;
      }
      const filename = `capture_${Date.now()}.jpg`;
      const file = new File([blob], filename, { type: "image/jpeg" });
      closeCamera();
      handleFile(file);
    },
    "image/jpeg",
    0.92
  );
}

// ═══════════════════════════════════════════════════════════════════
// STATUS HELPERS
// ═══════════════════════════════════════════════════════════════════
function setCamStatus(msg, type = "info") {
  if (!camStatus) return;
  camStatus.textContent = msg;
  camStatus.className = "cam-status-txt";
  if (type === "success") camStatus.classList.add("cam-st-ok");
  else if (type === "error") camStatus.classList.add("cam-st-err");
  else camStatus.classList.add("cam-st-info");
}

// ═══════════════════════════════════════════════════════════════════
// ANALYZE — send to Flask backend
// ═══════════════════════════════════════════════════════════════════
analyzeBtn?.addEventListener("click", analyze);

async function analyze() {
  if (!selectedFile) return;
  setBusy(true);
  hideCards();

  try {
    const uploadFile = await optimizeImageForUpload(selectedFile);
    const form = new FormData();
    form.append("image", uploadFile);
    const res  = await fetch("/api/predict", { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) {
      if (data.issues && data.issues.length) {
        let errHtml = `<strong>${data.message || "Quality Issues"}</strong><ul style="margin-top:10px; text-align:left;">`;
        data.issues.forEach(i => errHtml += `<li>${i}</li>`);
        errHtml += "</ul>";
        if (data.suggestions && data.suggestions.length) {
          errHtml += `<br><strong>Suggestions:</strong><ul style="text-align:left;">`;
          data.suggestions.forEach(s => errHtml += `<li>${s}</li>`);
          errHtml += "</ul>";
        }
        showError(errHtml);
        return;
      }
      throw new Error(data.error || data.description || "Server error");
    }
    showResult(data);
  } catch (err) {
    showError(err.message || "Network error — is the server running?");
  } finally {
    setBusy(false);
  }
}

function setBusy(busy) {
  analyzeBtn.disabled = busy;
  btnLabel.classList.toggle("hidden", busy);
  btnSpinner.classList.toggle("hidden", !busy);
}

// ═══════════════════════════════════════════════════════════════════
// SHOW RESULT
// ═══════════════════════════════════════════════════════════════════
function showResult(data) {
  errorCard.classList.add("hidden");
  resultCard.classList.remove("hidden");

  const healthy = data.status === "healthy";
  const isUncertain = data.status === "uncertain";
  const cls = isUncertain ? "uncertain" : (data.status || "diseased");

  // Status mapping for badges
  const statusEl = document.getElementById("badgeSt");
  if (statusEl) {
    statusEl.textContent = isUncertain ? "Uncertain" : (healthy ? "Healthy" : "Diseased");
    statusEl.className = "status-badge " + (healthy ? "status-healthy" : "status-diseased");
  }

  resultEmoji.textContent  = healthy ? "✅" : (isUncertain ? "❓" : (data.status === "error" ? "⚠️" : "🦠"));
  resultDis.textContent    = data.disease || "Unknown";
  
  if (badgeSeverity) {
    badgeSeverity.textContent = `Severity: ${data.severity || "Normal"}`;
  }

  // Update horizontal confidence bar
  const conf = parseFloat(data.confidence) || 0;
  if (ringFg) {
    ringFg.style.width = conf + "%";
    // Color logic
    if (healthy) ringFg.style.background = "var(--g6)";
    else if (isUncertain) ringFg.style.background = "var(--a5)";
    else ringFg.style.background = "var(--r5)";
  }
  
  if (confNum) {
    confNum.textContent = Math.round(conf) + "%";
  }

  resultDesc.textContent = data.result_summary || data.description || "—";
  
  // Minimal data filling for 3-column grid
  fillList(listSymptoms,  data.symptoms   || []);
  fillList(listTreatment, data.treatment  || []);
  fillList(listPrevent,   data.prevention || []);
  
  lastResult = data;
  openChatForResult(data);
  resultCard.scrollIntoView({ behavior: "smooth", block: "start" });
}

function fillList(ul, items) {
  ul.innerHTML = items.length
    ? items.map(i => `<li>${i}</li>`).join("")
    : "<li>No data available.</li>";
}

function animateNum(el, target, dur) {
  const start = performance.now();
  const tick  = now => {
    const p = Math.min((now - start) / dur, 1);
    el.textContent = Math.round(target * (1 - Math.pow(1 - p, 3)));
    if (p < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

// ═══════════════════════════════════════════════════════════════════
// SAVE REPORT AS TEXT FILE
// ═══════════════════════════════════════════════════════════════════
saveBtn?.addEventListener("click", () => {
  if (!lastResult) return;
  const lines = [
    "PLANTCURE — DIAGNOSIS REPORT",
    "=".repeat(40),
    `Disease    : ${lastResult.disease}`,
    `Status     : ${lastResult.status}`,
    `Confidence : ${lastResult.confidence}%`,
    `Severity   : ${lastResult.severity}`,
    `Date/Time  : ${new Date().toLocaleString()}`,
    "",
    "DESCRIPTION:",
    (lastResult.result_summary || lastResult.description || "No summary available"), "",
    "SYMPTOMS:",
    ...(lastResult.symptoms   || []).map(s => `  • ${s}`), "",
    "TREATMENT:",
    ...(lastResult.treatment  || []).map(s => `  • ${s}`), "",
    "PREVENTION:",
    ...(lastResult.prevention || []).map(s => `  • ${s}`),
  ];
  const blob = new Blob([lines.join("\n")], { type: "text/plain" });
  const a    = Object.assign(document.createElement("a"), {
    href    : URL.createObjectURL(blob),
    download: `PlantCure_Report_${Date.now()}.txt`,
  });
  a.click();
  URL.revokeObjectURL(a.href);
});

// ═══════════════════════════════════════════════════════════════════
// ERROR DISPLAY
// ═══════════════════════════════════════════════════════════════════
function showError(msg) {
  resultCard.classList.add("hidden");
  errorMsg.innerHTML = msg; // Changed to innerHTML to support list format
  errorCard.classList.remove("hidden");
  errorCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// ═══════════════════════════════════════════════════════════════════
// AI CHAT ASSISTANT (Updated)
// ═══════════════════════════════════════════════════════════════════
function openChatForResult(data) {
  const diseaseEl = document.getElementById('detected-disease');
  if (diseaseEl) {
    diseaseEl.value = data.disease || "Unknown";
    diseaseEl.dataset.confidence = data.confidence || 0;
  }
  const chatTitle = document.getElementById('chat-disease-title');
  const chatIntro = document.getElementById('chatbot-intro');
  if (chatTitle) {
    chatTitle.textContent = data.disease || "Treatment support";
  }
  if (chatIntro) {
    chatIntro.textContent = data.result_summary || data.description || "Ask for treatment, prevention, organic remedies, or chemical options.";
  }
  const openChatBtn = document.getElementById('open-chatbot');
  if (openChatBtn) {
    openChatBtn.style.display = 'block';
  }
  
  const chatMessages = document.getElementById('chat-messages');
  if (chatMessages) {
    chatMessages.innerHTML = '';
    const intro = `Diagnosis loaded: ${data.disease || "Unknown"}.\n\n${data.result_summary || data.description || "Ask what to do next."}`;
    const msgDiv = document.createElement('div');
    msgDiv.className = 'bot-message';
    msgDiv.textContent = intro;
    chatMessages.appendChild(msgDiv);
  }
}
