import { SplatViewer } from "./splat.js";
import { loadModel, estimateDepth, buildPLY, fitMetric } from "./pipeline.js";

const $ = (id) => document.getElementById(id);
const video = $("cam"), photo = $("photo"), loupe = $("loupe"), statusEl = $("status"), thumb = $("thumb");
const btn = $("capture"), retake = $("retake"), dl = $("download");
const refbar = $("refbar"), refType = $("refType"), refLen = $("refLen"), addRef = $("addRef"), applyScale = $("applyScale"), refList = $("refList");
const hoverEl = $("hover"), probeEl = $("probe"), bottom = $("bottom");
const presets = (window.APP_CONFIG && window.APP_CONFIG.REFERENCE_PRESETS) || [];
const setStatus = (t) => (statusEl.textContent = t);

// ---------- 3D hover read-out (bottom panel) ----------
const fmt = (v, u) => (u === "m" ? v.toFixed(3) + " m" : v.toFixed(2) + " u");
function onHover(h) {
  if (!h) { hoverEl.hidden = true; probeEl.hidden = true; return; }
  probeEl.hidden = false; probeEl.style.left = h.sx + "px"; probeEl.style.top = h.sy + "px";
  hoverEl.hidden = false;
  hoverEl.textContent = `X ${fmt(h.x, h.units)}  Y ${fmt(h.y, h.units)}  Depth ${fmt(h.depth, h.units)}` +
    (h.units === "m" ? "" : "  (relative)");
  hoverEl.style.left = Math.max(4, Math.min(h.px + 14, bottom.clientWidth - hoverEl.offsetWidth - 4)) + "px";
  hoverEl.style.top = Math.max(4, Math.min(h.py + 14, bottom.clientHeight - hoverEl.offsetHeight - 4)) + "px";
}
const viewer = new SplatViewer($("viewer"), { onHover });

// ---------- state ----------
let frame = null;        // full-resolution captured canvas
let depth = null;        // { img, disp, w, h, W, H } relative depth kept for re-fitting
let metric = null;       // { a, b } once references are applied
let refs = [];           // [{ p1, p2, lengthM, label }] pixel coords in the full-resolution frame
let pending = null;      // reference being pinned: { label, lengthM, pts: [] }
let preview = null;      // point currently being placed (finger/mouse down)
let lastPly = null, busy = false, modelMsg = "";

// ---------- camera + model ----------
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

// ---------- reference picker ----------
presets.forEach((p, i) => refType.add(new Option(p.label, i)));
function syncPreset() {
  const p = presets[refType.value];
  refLen.value = p && p.mm ? p.mm : "";
  if (p && !p.mm) refLen.focus();
}
refType.addEventListener("change", syncPreset);
syncPreset();

function showPhoto(on) {
  photo.hidden = !on; refbar.hidden = !on; retake.hidden = !on; video.hidden = on; loupe.hidden = true;
}

function redraw() {
  if (!frame) return;
  if (photo.width !== frame.width) { photo.width = frame.width; photo.height = frame.height; }
  const g = photo.getContext("2d"), lw = Math.max(2, frame.width / 400);
  g.drawImage(frame, 0, 0);
  const mark = (p, col) => {
    g.strokeStyle = col; g.lineWidth = lw; g.beginPath();
    g.arc(p.x, p.y, lw * 4, 0, 7);
    g.moveTo(p.x - lw * 8, p.y); g.lineTo(p.x + lw * 8, p.y);
    g.moveTo(p.x, p.y - lw * 8); g.lineTo(p.x, p.y + lw * 8); g.stroke();
  };
  g.font = `bold ${lw * 8}px sans-serif`;
  refs.forEach((r, i) => {
    g.strokeStyle = "#ffd54a"; g.lineWidth = lw; g.beginPath(); g.moveTo(r.p1.x, r.p1.y); g.lineTo(r.p2.x, r.p2.y); g.stroke();
    mark(r.p1, "#ffd54a"); mark(r.p2, "#ffd54a");
    g.fillStyle = "#ffd54a"; g.fillText(String(i + 1), (r.p1.x + r.p2.x) / 2 + lw * 4, (r.p1.y + r.p2.y) / 2 - lw * 4);
  });
  if (pending) {
    pending.pts.forEach((p) => mark(p, "#4fe0ff"));
    if (preview) mark(preview, "#ffffff");
  }
  if (pending && preview) {                       // 4x magnifier so the end points can be placed precisely
    const lg = loupe.getContext("2d");
    lg.imageSmoothingEnabled = false; lg.clearRect(0, 0, 120, 120);
    lg.drawImage(frame, preview.x - 15, preview.y - 15, 30, 30, 0, 0, 120, 120);
    lg.strokeStyle = "#ff3b3b"; lg.lineWidth = 1; lg.beginPath();
    lg.moveTo(60, 0); lg.lineTo(60, 120); lg.moveTo(0, 60); lg.lineTo(120, 60); lg.stroke();
    loupe.hidden = false;
  } else loupe.hidden = true;
}

function renderRefs() {
  refList.innerHTML = "";
  refs.forEach((r, i) => {
    const chip = document.createElement("span"); chip.className = "chip";
    chip.textContent = `${i + 1}. ${r.label} ${(r.lengthM * 1000).toFixed(r.lengthM < 0.1 ? 2 : 0)} mm`;
    const x = document.createElement("button"); x.textContent = "✕"; x.title = "Remove";
    x.onclick = () => { refs.splice(i, 1); renderRefs(); redraw(); };
    chip.append(x); refList.append(chip);
  });
}

function toImage(e) {                              // pointer -> full-resolution frame pixels (object-fit: contain)
  const r = photo.getBoundingClientRect(), s = Math.min(r.width / frame.width, r.height / frame.height);
  const ox = (r.width - frame.width * s) / 2, oy = (r.height - frame.height * s) / 2;
  return { x: Math.min(Math.max((e.clientX - r.left - ox) / s, 0), frame.width - 1),
           y: Math.min(Math.max((e.clientY - r.top - oy) / s, 0), frame.height - 1) };
}
photo.addEventListener("pointerdown", (e) => {
  if (!pending) return;
  photo.setPointerCapture(e.pointerId); preview = toImage(e); redraw();
});
photo.addEventListener("pointermove", (e) => { if (preview) { preview = toImage(e); redraw(); } });
photo.addEventListener("pointerup", () => {
  if (!preview || !pending) return;
  pending.pts.push(preview); preview = null;
  if (pending.pts.length === 2) {
    refs.push({ p1: pending.pts[0], p2: pending.pts[1], lengthM: pending.lengthM, label: pending.label });
    pending = null; addRef.textContent = "Add reference"; renderRefs();
    setStatus(refs.length < 2 ? "Add one more reference (placed at a different distance), then Apply metric scale."
                              : "Add a third reference to check the error, or press Apply metric scale.");
  } else setStatus(`Now press and drag to place end 2 of "${pending.label}".`);
  redraw();
});

addRef.addEventListener("click", () => {
  if (!frame) return;
  if (pending) { pending = null; preview = null; addRef.textContent = "Add reference"; redraw(); return; }
  const mm = parseFloat(refLen.value);
  if (!(mm > 0)) { setStatus("Enter the real length in mm first."); refLen.focus(); return; }
  const p = presets[refType.value];
  pending = { label: p ? p.label.split(" (")[0].split(":")[0] : "Reference", lengthM: mm / 1000, pts: [] };
  addRef.textContent = "Cancel";
  setStatus(`Press and drag to place end 1 of "${pending.label}" (${mm} mm). Release to set it.`);
});

function rebuild() {
  lastPly = buildPLY(depth.img, depth.disp, metric);
  return viewer.load(lastPly, { units: metric ? "m" : "units" });
}

applyScale.addEventListener("click", () => {
  if (!depth) { setStatus("Depth is still being estimated - wait a moment, then apply the scale."); return; }
  const fit = fitMetric(refs, depth.disp, depth.w, depth.h, depth.W, depth.H);
  if (!fit.ok) { setStatus("Cannot apply scale: " + fit.reason + "."); return; }
  metric = { a: fit.a, b: fit.b };
  const n = rebuild();
  setStatus(`Metric scale applied (${n.toLocaleString()} Gaussians, units = metres). Reference errors: ` +
    fit.errorsPct.map((v) => v.toFixed(1) + "%").join(", ") + (fit.warning ? ". Note: " + fit.warning : "") + ". Drag to tilt, scroll/pinch to zoom.");
});

// ---------- capture / retake / download ----------
btn.addEventListener("click", async () => {
  if (!video.videoWidth || busy) return;
  busy = true; btn.disabled = true;
  try {
    frame = document.createElement("canvas");
    frame.width = video.videoWidth; frame.height = video.videoHeight;
    frame.getContext("2d").drawImage(video, 0, 0);
    thumb.src = frame.toDataURL("image/jpeg", 0.7); thumb.style.display = "block";
    refs = []; pending = null; preview = null; metric = null; depth = null; renderRefs();
    showPhoto(true); redraw();

    depth = await estimateDepth(frame, frame.width, frame.height, setStatus);
    setStatus("Rendering Gaussian splats…");
    const n = rebuild();
    dl.disabled = false;
    setStatus(`Relative depth ready (${n.toLocaleString()} Gaussians). For metres: add 2+ reference objects, then Apply metric scale. Drag to tilt, scroll/pinch to zoom.`);
  } catch (e) {
    console.error(e); setStatus("Error: " + e.message);
  } finally {
    busy = false; btn.disabled = false;
  }
});

retake.addEventListener("click", () => {
  showPhoto(false); thumb.style.display = "none"; frame = null; pending = null; preview = null; addRef.textContent = "Add reference";
  setStatus("Camera ready. Press Capture.");
});

dl.addEventListener("click", () => {
  if (!lastPly) return;
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([lastPly], { type: "application/octet-stream" }));
  a.download = metric ? "pointcloud_metric.ply" : "pointcloud.ply"; a.click();
});

startCamera();
