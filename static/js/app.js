/**
 * FaceMask AI - Interactive Web Dashboard Logic
 * Handles tabs, drag-and-drop image uploads, 1-click test samples,
 * and live browser webcam video inference.
 */

document.addEventListener("DOMContentLoaded", () => {
    // ----------------------------------------------------
    // 1. Tab Navigation
    // ----------------------------------------------------
    const tabs = [
        { btn: document.getElementById("tab-btn-image"), panel: document.getElementById("tab-content-image") },
        { btn: document.getElementById("tab-btn-webcam"), panel: document.getElementById("tab-content-webcam") },
        { btn: document.getElementById("tab-btn-reports"), panel: document.getElementById("tab-content-reports") },
    ];

    tabs.forEach(({ btn, panel }) => {
        btn.addEventListener("click", () => {
            tabs.forEach(t => {
                t.btn.classList.remove("active");
                t.btn.setAttribute("aria-selected", "false");
                t.panel.classList.remove("active");
            });
            btn.classList.add("active");
            btn.setAttribute("aria-selected", "true");
            panel.classList.add("active");

            // Pause webcam if switching away from webcam tab
            if (btn.id !== "tab-btn-webcam" && isWebcamActive) {
                stopWebcam();
            }
        });
    });

    // ----------------------------------------------------
    // 2. Image Upload & Drag-and-Drop
    // ----------------------------------------------------
    const dropzone = document.getElementById("image-dropzone");
    const fileInput = document.getElementById("file-input");
    const btnBrowse = document.getElementById("btn-browse");
    const emptyState = document.getElementById("empty-state");
    const resultContent = document.getElementById("result-content");
    const previewImage = document.getElementById("preview-image");
    const loadingSpinner = document.getElementById("loading-spinner");
    const latencyTag = document.getElementById("image-latency-tag");
    const detectionsList = document.getElementById("detections-list");

    btnBrowse.addEventListener("click", (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    dropzone.addEventListener("click", () => fileInput.click());

    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.classList.add("dragover");
    });

    dropzone.addEventListener("dragleave", () => {
        dropzone.classList.remove("dragover");
    });

    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.classList.remove("dragover");
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files[0]) {
            handleFileUpload(e.target.files[0]);
        }
    });

    // 1-Click Samples
    document.getElementById("btn-sample-mask").addEventListener("click", () => {
        sendSampleRequest("mask");
    });
    document.getElementById("btn-sample-nomask").addEventListener("click", () => {
        sendSampleRequest("no_mask");
    });

    function showLoading() {
        emptyState.classList.add("hidden");
        resultContent.classList.remove("hidden");
        loadingSpinner.classList.remove("hidden");
    }

    function hideLoading() {
        loadingSpinner.classList.add("hidden");
    }

    async function handleFileUpload(file) {
        showLoading();
        const formData = new FormData();
        formData.append("file", file);

        try {
            const res = await fetch("/predict_image", {
                method: "POST",
                body: formData
            });
            const data = await res.json();
            renderImageResult(data);
        } catch (err) {
            console.error(err);
            alert("Error analyzing image: " + err.message);
        } finally {
            hideLoading();
        }
    }

    async function sendSampleRequest(sampleType) {
        showLoading();
        const formData = new FormData();
        formData.append("sample", sampleType);

        try {
            const res = await fetch("/predict_image", {
                method: "POST",
                body: formData
            });
            const data = await res.json();
            renderImageResult(data);
        } catch (err) {
            console.error(err);
            alert("Error analyzing sample: " + err.message);
        } finally {
            hideLoading();
        }
    }

    function renderImageResult(data) {
        if (!data.success) {
            alert(data.error || "Detection failed.");
            return;
        }

        previewImage.src = data.image_data;
        latencyTag.textContent = `${data.inference_time_ms} ms`;

        detectionsList.innerHTML = "";
        if (data.detections.length === 0) {
            detectionsList.innerHTML = `
                <div class="detection-item">
                    <span>No human face detected in this frame.</span>
                </div>
            `;
            return;
        }

        data.detections.forEach((det, idx) => {
            const isMask = det.label === "Mask";
            const badgeClass = isMask ? "mask" : "no-mask";

            const item = document.createElement("div");
            item.className = "detection-item";
            item.innerHTML = `
                <div class="detection-tag">
                    <span>Face #${idx + 1}</span>
                    <span class="badge-status ${badgeClass}">${det.label}</span>
                </div>
                <div>
                    <strong>${det.confidence}%</strong> confidence
                </div>
            `;
            detectionsList.appendChild(item);
        });
    }

    // ----------------------------------------------------
    // 3. Live Webcam Stream Inference
    // ----------------------------------------------------
    const video = document.getElementById("webcam-video");
    const overlay = document.getElementById("webcam-overlay");
    const overlayCtx = overlay.getContext("2d");
    const btnStartWebcam = document.getElementById("btn-start-webcam");
    const btnStopWebcam = document.getElementById("btn-stop-webcam");
    const webcamPlaceholder = document.getElementById("webcam-placeholder");
    const webcamHud = document.getElementById("webcam-hud");
    const webcamControlsBar = document.getElementById("webcam-controls-bar");
    const fpsBadge = document.getElementById("webcam-fps");
    const latencyBadge = document.getElementById("webcam-latency");
    const liveFaceCount = document.getElementById("live-face-count");
    const liveMaskBadge = document.getElementById("live-mask-badge");

    let stream = null;
    let isWebcamActive = false;
    let frameTimer = null;
    let captureCanvas = document.createElement("canvas");
    let captureCtx = captureCanvas.getContext("2d");
    let frameCount = 0;
    let lastFpsUpdate = performance.now();

    btnStartWebcam.addEventListener("click", startWebcam);
    btnStopWebcam.addEventListener("click", stopWebcam);

    async function startWebcam() {
        try {
            stream = await navigator.mediaDevices.getUserMedia({
                video: { width: { ideal: 640 }, height: { ideal: 480 } },
                audio: false
            });
            video.srcObject = stream;
            await video.play();

            isWebcamActive = true;
            webcamPlaceholder.classList.add("hidden");
            webcamHud.classList.remove("hidden");
            webcamControlsBar.classList.remove("hidden");

            video.addEventListener("loadedmetadata", () => {
                overlay.width = video.videoWidth;
                overlay.height = video.videoHeight;
                captureCanvas.width = video.videoWidth;
                captureCanvas.height = video.videoHeight;
            });

            loopInference();
        } catch (err) {
            console.error("Camera access error:", err);
            alert("Could not access camera. Please ensure camera permissions are granted.");
        }
    }

    function stopWebcam() {
        isWebcamActive = false;
        if (frameTimer) clearTimeout(frameTimer);
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            stream = null;
        }
        video.srcObject = null;
        overlayCtx.clearRect(0, 0, overlay.width, overlay.height);

        webcamPlaceholder.classList.remove("hidden");
        webcamHud.classList.add("hidden");
        webcamControlsBar.classList.add("hidden");
        liveFaceCount.textContent = "0";
        liveMaskBadge.className = "badge-neutral";
        liveMaskBadge.textContent = "Idle";
    }

    async function loopInference() {
        if (!isWebcamActive) return;

        if (video.readyState === video.HAVE_ENOUGH_DATA) {
            captureCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
            const b64Data = captureCanvas.toDataURL("image/jpeg", 0.7);

            try {
                const t0 = performance.now();
                const res = await fetch("/predict_frame", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ image: b64Data })
                });
                const data = await res.json();
                const t1 = performance.now();

                // Draw overlay
                drawOverlay(data.detections || []);

                // Update FPS & Latency
                frameCount++;
                if (t1 - lastFpsUpdate >= 1000) {
                    const currentFps = Math.round((frameCount * 1000) / (t1 - lastFpsUpdate));
                    fpsBadge.textContent = `${currentFps} FPS`;
                    frameCount = 0;
                    lastFpsUpdate = t1;
                }
                latencyBadge.textContent = `${Math.round(t1 - t0)} ms`;

                // Update Sidebar Stats
                const numFaces = (data.detections || []).length;
                liveFaceCount.textContent = numFaces;

                if (numFaces > 0) {
                    const hasMask = data.detections.some(d => d.label === "Mask");
                    const hasNoMask = data.detections.some(d => d.label === "No Mask");

                    if (hasMask && !hasNoMask) {
                        liveMaskBadge.className = "badge-status mask";
                        liveMaskBadge.textContent = "Protected (Mask)";
                    } else if (hasNoMask && !hasMask) {
                        liveMaskBadge.className = "badge-status no-mask";
                        liveMaskBadge.textContent = "Alert (No Mask)";
                    } else {
                        liveMaskBadge.className = "badge-status no-mask";
                        liveMaskBadge.textContent = "Mixed Status";
                    }
                } else {
                    liveMaskBadge.className = "badge-neutral";
                    liveMaskBadge.textContent = "Searching...";
                }
            } catch (err) {
                console.error("Frame processing error:", err);
            }
        }

        // Throttle ~10 FPS for optimal CPU responsiveness
        frameTimer = setTimeout(loopInference, 100);
    }

    function drawOverlay(detections) {
        overlayCtx.clearRect(0, 0, overlay.width, overlay.height);

        detections.forEach(det => {
            const [x, y, w, h] = det.box;
            const isMask = det.label === "Mask";
            const color = isMask ? "#10b981" : "#f43f5e";

            // Glowing bounding box
            overlayCtx.strokeStyle = color;
            overlayCtx.lineWidth = 3;
            overlayCtx.shadowColor = color;
            overlayCtx.shadowBlur = 12;
            overlayCtx.strokeRect(x, y, w, h);

            // Reset shadow for text
            overlayCtx.shadowBlur = 0;

            // Text Banner
            const text = `${det.label} ${det.confidence}%`;
            overlayCtx.font = "bold 15px 'Outfit', sans-serif";
            const textMetrics = overlayCtx.measureText(text);
            const textWidth = textMetrics.width;
            const textHeight = 18;

            const bannerY = y > 24 ? y - textHeight - 4 : y + h + 4;
            overlayCtx.fillStyle = color;
            overlayCtx.fillRect(x, bannerY, textWidth + 12, textHeight + 6);

            overlayCtx.fillStyle = "#ffffff";
            overlayCtx.fillText(text, x + 6, bannerY + textHeight - 2);
        });
    }
});
