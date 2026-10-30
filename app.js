import { SplatViewer } from "./splat.js";
import { loadModel, reconstruct } from "./pipeline.js";

const $ = (id) => document.getElementById(id);
const video = $("cam"), statusEl = $("status"), btn = $("capture"), dl = $("download"), thumb = $("thumb");
const viewer = new SplatViewer($("viewer"));
let lastPly = null, busy = false, modelMsg = "";
const setStatus = (t) => (statusEl.textContent = t);

async function startCamera() {
  try {
    video.srcObject = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
    await video.play();
    btn.disabled = false;
    setStatus(modelMsg || "Camera ready. Loading depth model in the background…");
  } catch (e) {
    setStatus("Camera unavailable: " + e.message + " (HTTPS and permission are required)");
  }
}

// Warm up the model while the user frames the shot (first visit downloads ~100 MB, then it is cached).
loadModel((m) => { modelMsg = m; if (!busy) setStatus(m); })
  .then(() => { modelMsg = "Camera and model ready. Press Capture."; if (!busy) setStatus(modelMsg); })
  .catch((e) => setStatus("Model failed to load: " + e.message));

btn.addEventListener("click", async () => {
  if (!video.videoWidth || busy) return;
  busy = true; btn.disabled = true;
  try {
    const c = document.createElement("canvas");
    c.width = video.videoWidth; c.height = video.videoHeight;
    c.getContext("2d").drawImage(video, 0, 0);
    thumb.src = c.toDataURL("image/jpeg", 0.7); thumb.style.display = "block";

    lastPly = await reconstruct(c, c.width, c.height, setStatus);
    setStatus("Rendering Gaussian splats…");
    const n = viewer.load(lastPly);
    dl.disabled = false;
    setStatus(`Done – ${n.toLocaleString()} Gaussians. Drag to tilt, scroll/pinch to zoom.`);
  } catch (e) {
    console.error(e); setStatus("Error: " + e.message);
  } finally {
    busy = false; btn.disabled = false;
  }
});

dl.addEventListener("click", () => {
  if (!lastPly) return;
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([lastPly], { type: "application/octet-stream" }));
  a.download = "pointcloud.ply"; a.click();
});

startCamera();
