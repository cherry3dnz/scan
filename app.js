import { SplatViewer } from "./splat.js";
import { loadModel, estimateDepth, buildPLY, fitMetric } from "./pipeline.js";

const $ = (id) => document.getElementById(id);
const video = $("cam"), photo = $("photo"), loupe = $("loupe"), statusEl = $("status"), thumb = $("thumb");
const btn = $("capture"), retake = $("retake"), dl = $("download");
const refbar = $("refbar"), refType = $("refType"), refLen = $("refLen"), addRef = $("addRef"), applyScale = $("applyScale"), refList = $("refList");
const hoverEl = $("hover"), probeEl = $("probe"), bottom = $("bottom");
const app = $("app"), stage = $("stage"), zoomTag = $("zoomTag"), doneBtn = $("doneBtn"), fitBtn = $("fitBtn");
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
let preview = null;      // point currently being placed (finger/mouse down), image pixels
let lastPly = null, busy = false, modelMsg = "";

// pinning mode / zoom state
let pinMode = false;                       // full-screen pinning layout (3D viewer hidden)
const view = { z: 1, cx: 0, cy: 0 };       // zoom relative to "fit" (>= 1) and the image point at the stage centre
const ptrs = new Map();                    // active pointers: id -> {x, y} in stage CSS pixels
let gesture = "none";                      // "none" | "pin" | "pan" | "pinch"
let pinch = null, panLast = null, curPos = null, raf = 0;
const MAX_CSS_SCALE = 20;                  // max on-screen CSS px per image pixel
const MARGIN = 40;                         // image may be panned this far (CSS px) past the stage edge

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
  if (!on) zoomTag.hidden = true;
}

// ---------- view transform (image pixels <-> stage CSS pixels) ----------
const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
const mid = (a, b) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });

function geom() {
  const W = stage.clientWidth, H = stage.clientHeight;
  return { W, H, fit: Math.min(W / frame.width, H / frame.height) };
}
// Normal layout always shows the whole photo; pinning mode uses the zoom/pan state in `view`.
function xform() {
  const { W, H, fit } = geom();
  if (!pinMode) return { W, H, fit, s: fit, tx: (W - frame.width * fit) / 2, ty: (H - frame.height * fit) / 2 };
  const s = fit * view.z;
  return { W, H, fit, s, tx: W / 2 - view.cx * s, ty: H / 2 - view.cy * s };
}
function clampView() {
  if (!frame) return;
  const { W, H, fit } = geom();
  if (!(fit > 0)) return;
  view.z = clamp(view.z, 1, Math.max(1, MAX_CSS_SCALE / fit));
  const s = fit * view.z, ax = (W / 2 - MARGIN) / s, ay = (H / 2 - MARGIN) / s;
  view.cx = ax * 2 >= frame.width ? frame.width / 2 : clamp(view.cx, ax, frame.width - ax);
  view.cy = ay * 2 >= frame.height ? frame.height / 2 : clamp(view.cy, ay, frame.height - ay);
}
function resetView() { view.z = 1; view.cx = frame ? frame.width / 2 : 0; view.cy = frame ? frame.height / 2 : 0; }

function screenToImg(pos) { const { s, tx, ty } = xform(); return { x: (pos.x - tx) / s, y: (pos.y - ty) / s }; }
function toImage(pos) {                            // same, clamped to the frame (what a pin stores)
  const p = screenToImg(pos);
  return { x: clamp(p.x, 0, frame.width - 1), y: clamp(p.y, 0, frame.height - 1) };
}
// Set the zoom and keep the image point `anchor` under the screen position `pos` (this also pans).
function zoomAbout(pos, z, anchor) {
  const { W, H, fit } = geom();
  view.z = z; clampView();
  const s = fit * view.z;
  view.cx = anchor.x - (pos.x - W / 2) / s; view.cy = anchor.y - (pos.y - H / 2) / s;
  clampView();
}

// ---------- drawing ----------
function redraw() {
  if (!frame) return;
  const W = stage.clientWidth, H = stage.clientHeight;
  if (!W || !H) return;
  const dpr = Math.min(window.devicePixelRatio || 1, 3), bw = Math.round(W * dpr), bh = Math.round(H * dpr);
  if (photo.width !== bw || photo.height !== bh) { photo.width = bw; photo.height = bh; }
  const g = photo.getContext("2d"), { s, tx, ty, fit } = xform();

  g.setTransform(1, 0, 0, 1, 0, 0); g.fillStyle = "#000"; g.fillRect(0, 0, bw, bh);
  g.imageSmoothingEnabled = s < 3;                 // show hard pixels when zoomed in far
  g.imageSmoothingQuality = "high";
  g.setTransform(dpr * s, 0, 0, dpr * s, dpr * tx, dpr * ty);
  g.drawImage(frame, 0, 0);

  // overlays are drawn in screen space so they keep a constant size at any zoom
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  const P = (p) => ({ x: p.x * s + tx, y: p.y * s + ty });
  const seg = (a, b, col, dashed) => {
    const A = P(a), B = P(b);
    g.setLineDash(dashed ? [6, 5] : []);
    for (const [c, w] of [["rgba(0,0,0,.8)", 3.5], [col, 1.5]]) {
      g.strokeStyle = c; g.lineWidth = w; g.beginPath(); g.moveTo(A.x, A.y); g.lineTo(B.x, B.y); g.stroke();
    }
    g.setLineDash([]);
  };
  const mark = (p, col) => {                       // ring + four ticks with an open centre so the target pixel stays visible
    const q = P(p);
    for (const [c, w] of [["rgba(0,0,0,.85)", 4], [col, 1.6]]) {
      g.strokeStyle = c; g.lineWidth = w; g.beginPath(); g.arc(q.x, q.y, 9, 0, 7);
      for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
        g.moveTo(q.x + dx * 3, q.y + dy * 3); g.lineTo(q.x + dx * 16, q.y + dy * 16);
      }
      g.stroke();
    }
  };
  g.font = "bold 13px sans-serif"; g.lineJoin = "round";
  refs.forEach((r, i) => {
    seg(r.p1, r.p2, "#ffd54a"); mark(r.p1, "#ffd54a"); mark(r.p2, "#ffd54a");
    const a = P(r.p1), b = P(r.p2), lx = (a.x + b.x) / 2 + 8, ly = (a.y + b.y) / 2 - 8;
    g.lineWidth = 3; g.strokeStyle = "#000"; g.strokeText(String(i + 1), lx, ly);
    g.fillStyle = "#ffd54a"; g.fillText(String(i + 1), lx, ly);
  });
  if (pending) {
    if (pending.pts.length && preview) seg(pending.pts[0], preview, "#4fe0ff", true);
    pending.pts.forEach((p) => mark(p, "#4fe0ff"));
    if (preview) mark(preview, "#ffffff");
  }

  zoomTag.hidden = !pinMode; if (pinMode) zoomTag.textContent = (s / fit).toFixed(1) + "×";

  // magnifier: shown on the side of the screen opposite the finger so the hand never covers it
  if (pinMode && pending && preview && curPos) {
    const LB = loupe.width, LS = LB / 2;           // backing store is 2x the 180 px CSS size
    const mag = clamp(Math.max(6, s * 2), 6, 30), src = LS / mag;   // always more detail than the main view
    const lg = loupe.getContext("2d");
    lg.fillStyle = "#000"; lg.fillRect(0, 0, LB, LB);
    lg.imageSmoothingEnabled = false;
    lg.drawImage(frame, preview.x - src / 2, preview.y - src / 2, src, src, 0, 0, LB, LB);
    const c = LB / 2;
    for (const [col, w] of [["rgba(0,0,0,.8)", 3], ["#ff3b3b", 1.2]]) {
      lg.strokeStyle = col; lg.lineWidth = w; lg.beginPath();
      lg.moveTo(c, 0); lg.lineTo(c, c - 10); lg.moveTo(c, c + 10); lg.lineTo(c, LB);
      lg.moveTo(0, c); lg.lineTo(c - 10, c); lg.moveTo(c + 10, c); lg.lineTo(LB, c); lg.stroke();
    }
    const left = curPos.x >= W / 2;
    loupe.style.left = left ? "10px" : "auto"; loupe.style.right = left ? "auto" : "10px";
    loupe.hidden = false;
  } else loupe.hidden = true;
}
const schedule = () => { if (!raf) raf = requestAnimationFrame(() => { raf = 0; redraw(); }); };
new ResizeObserver(() => { if (frame) { clampView(); redraw(); } }).observe(stage);

// ---------- pinning mode ----------
function setPinMode(on) {
  if (pinMode === on) return;
  pinMode = on; app.classList.toggle("pinning", on);
  doneBtn.hidden = fitBtn.hidden = !on;
  ptrs.clear(); gesture = "none"; preview = null; curPos = null;
  if (!on) { pending = null; addRef.textContent = "Add reference"; }
  if (frame) { clampView(); redraw(); }
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

function commitPoint() {
  pending.pts.push(preview); preview = null;
  if (pending.pts.length === 2) {
    refs.push({ p1: pending.pts[0], p2: pending.pts[1], lengthM: pending.lengthM, label: pending.label });
    pending = null; addRef.textContent = "Add reference"; renderRefs();
    setStatus(refs.length < 2 ? "Saved. Add one more reference at a different distance, then Apply metric scale."
                              : "Saved. Add a third reference to check the error, or press Apply metric scale.");
  } else setStatus(`End 1 set. Pinch/drag to move around if needed, then press and drag to place end 2 of "${pending.label}".`);
  redraw();
}

const posOf = (e) => { const r = photo.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top }; };
function startPinch() {
  const [a, b] = [...ptrs.values()];
  pinch = { d0: Math.max(dist(a, b), 1), z0: view.z, anchor: screenToImg(mid(a, b)) };
  gesture = "pinch";
}

// One finger: place a pin (while a reference is being added) or pan. Two fingers: pinch-zoom + pan.
photo.addEventListener("pointerdown", (e) => {
  if (!pinMode || !frame || ptrs.size >= 2) return;
  e.preventDefault(); photo.setPointerCapture(e.pointerId);
  const pos = posOf(e); ptrs.set(e.pointerId, pos); curPos = pos;
  if (ptrs.size === 2) { preview = null; startPinch(); }
  else {
    const wantPan = !pending || e.button === 1 || e.button === 2 || (e.pointerType === "mouse" && e.shiftKey);
    if (wantPan) { gesture = "pan"; panLast = pos; }
    else { gesture = "pin"; preview = toImage(pos); }
  }
  redraw();
});
photo.addEventListener("pointermove", (e) => {
  if (!ptrs.has(e.pointerId)) return;
  const pos = posOf(e); ptrs.set(e.pointerId, pos); curPos = pos;
  if (gesture === "pinch" && ptrs.size === 2) {
    const [a, b] = [...ptrs.values()];
    zoomAbout(mid(a, b), pinch.z0 * dist(a, b) / pinch.d0, pinch.anchor);
  } else if (gesture === "pan") {
    const { s } = xform();
    view.cx -= (pos.x - panLast.x) / s; view.cy -= (pos.y - panLast.y) / s; panLast = pos; clampView();
  } else if (gesture === "pin") preview = toImage(pos);
  schedule();
});
function endPointer(e) {
  if (!ptrs.has(e.pointerId)) return;
  ptrs.delete(e.pointerId);
  if (gesture === "pinch") {
    // the remaining finger keeps panning but never places a pin by accident
    if (ptrs.size === 1) { gesture = "pan"; panLast = [...ptrs.values()][0]; } else gesture = "none";
  } else if (ptrs.size === 0) {
    const place = gesture === "pin" && e.type === "pointerup" && preview && pending;
    gesture = "none";
    if (place) commitPoint(); else preview = null;
  }
  schedule();
}
photo.addEventListener("pointerup", endPointer);
photo.addEventListener("pointercancel", endPointer);
photo.addEventListener("wheel", (e) => {
  if (!pinMode || !frame) return;
  e.preventDefault();
  const pos = posOf(e), dy = e.deltaMode === 1 ? e.deltaY * 33 : e.deltaY;
  zoomAbout(pos, view.z * Math.exp(-dy * 0.0015), screenToImg(pos)); schedule();
}, { passive: false });
photo.addEventListener("contextmenu", (e) => e.preventDefault());

fitBtn.addEventListener("click", () => { resetView(); redraw(); });
doneBtn.addEventListener("click", () => {
  setPinMode(false);
  setStatus(refs.length < 2 ? "Add at least two references (at different distances), then Apply metric scale."
                            : "Press Apply metric scale, or add more references.");
});

addRef.addEventListener("click", () => {
  if (!frame) return;
  if (pending) {                                   // cancel the reference being pinned
    pending = null; preview = null; addRef.textContent = "Add reference"; redraw();
    setStatus("Cancelled. Add another reference or press Done.");
    return;
  }
  const mm = parseFloat(refLen.value);
  if (!(mm > 0)) { setStatus("Enter the real length in mm first."); refLen.focus(); return; }
  const p = presets[refType.value];
  pending = { label: p ? p.label.split(" (")[0].split(":")[0] : "Reference", lengthM: mm / 1000, pts: [] };
  addRef.textContent = "Cancel";
  setPinMode(true);                                // full-screen photo, 3D viewer hidden
  setStatus(`Pinch to zoom, drag to pan. Then press and drag to place end 1 of "${pending.label}" (${mm} mm); release to set.`);
  redraw();
});

function rebuild() {
  lastPly = buildPLY(depth.img, depth.disp, metric);
  return viewer.load(lastPly, { units: metric ? "m" : "units" });
}

applyScale.addEventListener("click", () => {
  if (!depth) { setStatus("Depth is still being estimated - wait a moment, then apply the scale."); return; }
  const fit = fitMetric(refs, depth.disp, depth.w, depth.h, depth.W, depth.H);
  if (!fit.ok) { setStatus("Cannot apply scale: " + fit.reason + "."); return; }   // stay in pinning mode so it can be fixed
  setPinMode(false);                               // bring the 3D viewer back
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
    setPinMode(false); resetView();
    showPhoto(true); redraw();

    depth = await estimateDepth(frame, frame.width, frame.height, setStatus);
    setStatus("Rendering Gaussian splats…");
    const n = rebuild();
    dl.disabled = false;
    setStatus(`Relative depth ready (${n.toLocaleString()} Gaussians). For metres: tap Add reference to pin 2+ reference objects (full-screen, pinch to zoom), then Apply metric scale. Drag to tilt, scroll/pinch to zoom.`);
  } catch (e) {
    console.error(e); setStatus("Error: " + e.message);
  } finally {
    busy = false; btn.disabled = false;
  }
});

retake.addEventListener("click", () => {
  setPinMode(false);
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
