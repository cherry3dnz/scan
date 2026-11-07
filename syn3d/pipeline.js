// Steps 2-4 in the browser: undistort -> monocular depth (Transformers.js) -> 3DGS-layout PLY.
const CFG = globalThis.APP_CONFIG || {};
const MODEL_ID = CFG.MODEL_ID || "onnx-community/depth-anything-v2-small";
const MAX_SIDE = CFG.MAX_SIDE || 640;
const STRIDE = CFG.POINT_STRIDE || 2;
const HFOV = CFG.HFOV_DEG || 65;
const NEAR = 0.5, FAR = 5.0, SH_C0 = 0.28209479177387814;
const TRANSFORMERS_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";

let estimatorPromise = null;

export function loadModel(onProgress = () => {}) {
  if (estimatorPromise) return estimatorPromise;
  estimatorPromise = (async () => {
    const { pipeline, env } = await import(TRANSFORMERS_URL);
    env.allowLocalModels = false;
    const progress_callback = (p) => {
      if (p.status === "progress" && p.total) onProgress(`Downloading model… ${Math.round(p.progress)}%`);
      else if (p.status === "ready") onProgress("Model ready");
    };
    const base = { progress_callback };
    if (CFG.MODEL_DTYPE) base.dtype = CFG.MODEL_DTYPE;
    if (navigator.gpu) {
      try { return await pipeline("depth-estimation", MODEL_ID, { ...base, device: "webgpu" }); }
      catch (e) { console.warn("WebGPU failed, falling back to WASM", e); }
    }
    return await pipeline("depth-estimation", MODEL_ID, base);
  })();
  estimatorPromise.catch(() => (estimatorPromise = null)); // allow retry
  return estimatorPromise;
}

// ---------- Step 2: lens undistortion (OpenCV model: k1,k2,k3 radial + p1,p2 tangential) ----------
export function intrinsics(w, h) {
  const c = CFG.CALIBRATION;
  if (c) {
    const sx = w / c.width, sy = h / c.height;
    return { fx: c.fx * sx, fy: c.fy * sy, cx: c.cx * sx, cy: c.cy * sy };
  }
  const f = w / 2 / Math.tan((HFOV * Math.PI) / 360);
  return { fx: f, fy: f, cx: w / 2, cy: h / 2 };
}

export function undistortImage(img, cal = CFG.CALIBRATION) {
  if (!cal) return img; // no calibration: assume the camera/phone already corrected the lens
  const { width: w, height: h, data } = img, K = intrinsics(w, h);
  const { k1 = 0, k2 = 0, k3 = 0, p1 = 0, p2 = 0 } = cal;
  const out = { width: w, height: h, data: new Uint8ClampedArray(data.length) };
  for (let v = 0; v < h; v++) {
    for (let u = 0; u < w; u++) {
      const x = (u - K.cx) / K.fx, y = (v - K.cy) / K.fy, r2 = x * x + y * y;
      const rad = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;
      const xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x);
      const yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y;
      const su = xd * K.fx + K.cx, sv = yd * K.fy + K.cy;
      const o = (v * w + u) * 4;
      if (su < 0 || sv < 0 || su > w - 1 || sv > h - 1) { out.data[o + 3] = 255; continue; }
      const u0 = Math.floor(su), v0 = Math.floor(sv), u1 = Math.min(u0 + 1, w - 1), v1 = Math.min(v0 + 1, h - 1);
      const a = su - u0, b = sv - v0;
      for (let ch = 0; ch < 3; ch++) {
        out.data[o + ch] =
          data[(v0 * w + u0) * 4 + ch] * (1 - a) * (1 - b) + data[(v0 * w + u1) * 4 + ch] * a * (1 - b) +
          data[(v1 * w + u0) * 4 + ch] * (1 - a) * b + data[(v1 * w + u1) * 4 + ch] * a * b;
      }
      out.data[o + 3] = 255;
    }
  }
  return out;
}

export function resizeBilinear(src, sw, sh, dw, dh) {
  const out = new Float32Array(dw * dh);
  for (let y = 0; y < dh; y++) {
    const fy = Math.min(Math.max(((y + 0.5) * sh) / dh - 0.5, 0), sh - 1), y0 = Math.floor(fy), y1 = Math.min(y0 + 1, sh - 1), wy = fy - y0;
    for (let x = 0; x < dw; x++) {
      const fx = Math.min(Math.max(((x + 0.5) * sw) / dw - 0.5, 0), sw - 1), x0 = Math.floor(fx), x1 = Math.min(x0 + 1, sw - 1), wx = fx - x0;
      out[y * dw + x] = src[y0 * sw + x0] * (1 - wx) * (1 - wy) + src[y0 * sw + x1] * wx * (1 - wy) +
        src[y1 * sw + x0] * (1 - wx) * wy + src[y1 * sw + x1] * wx * wy;
    }
  }
  return out;
}

// ---------- Metric scale from reference objects ----------
// Normalisation shared by the fit and by buildPLY: dn = (disparity - lo) / span (2nd..98th percentile).
export function normParams(disp) {
  const s = Float32Array.from(disp).sort();
  const lo = s[Math.floor(s.length * 0.02)], hi = s[Math.floor(s.length * 0.98)];
  return { lo, span: Math.max(hi - lo, 1e-6) };
}

export function patchMedian(a, w, h, x, y, r = 2) {
  const cx = Math.round(x), cy = Math.round(y), v = [];
  for (let j = -r; j <= r; j++) for (let i = -r; i <= r; i++) {
    const xx = cx + i, yy = cy + j;
    if (xx >= 0 && yy >= 0 && xx < w && yy < h) v.push(a[yy * w + xx]);
  }
  v.sort((p, q) => p - q);
  return v[v.length >> 1];
}

// refs: [{p1:{x,y}, p2:{x,y}, lengthM}] with pixel coordinates in the full-resolution captured frame (W x H).
// disp: relative disparity at working resolution (w x h). Solves 1/Z = a*dn + b (Z in metres) so that every
// reference's 3D end-to-end distance equals its known length (Levenberg-Marquardt on relative errors).
export function fitMetric(refs, disp, w, h, W, H) {
  if (refs.length < 2) return { ok: false, reason: "add at least two references" };
  const { lo, span } = normParams(disp), K = intrinsics(W, H);
  const items = refs.map((r) => ({
    L: r.lengthM,
    pix: Math.hypot(r.p2.x - r.p1.x, r.p2.y - r.p1.y),
    e: [r.p1, r.p2].map((p) => ({
      x: (p.x - K.cx) / K.fx, y: -(p.y - K.cy) / K.fy,
      d: (patchMedian(disp, w, h, (p.x * w) / W, (p.y * h) / H) - lo) / span,
    })),
  }));
  if (items.some((it) => it.pix < 8)) return { ok: false, reason: "a reference is too short in pixels - pin its two ends further apart" };

  // initial guess: sideways approximation Z = L*f/pixels, then straight-line fit of 1/Z against d
  const xs = items.map((it) => (it.e[0].d + it.e[1].d) / 2), ys = items.map((it) => it.pix / (it.L * K.fx));
  const n = xs.length, mx = xs.reduce((s, v) => s + v, 0) / n, my = ys.reduce((s, v) => s + v, 0) / n;
  let sxx = 0, sxy = 0;
  xs.forEach((x, i) => { sxx += (x - mx) ** 2; sxy += (x - mx) * (ys[i] - my); });
  if (sxx < 1e-6) return { ok: false, reason: "the references are at the same depth - place one nearer and one farther" };
  let a = sxy / sxx, b = my - a * mx;

  const res = (A, B) => items.map((it) => {
    const P = it.e.map((q) => { const Z = 1 / Math.max(A * q.d + B, 1e-3); return [q.x * Z, q.y * Z, -Z]; });
    return (Math.hypot(P[0][0] - P[1][0], P[0][1] - P[1][1], P[0][2] - P[1][2]) - it.L) / it.L;
  });
  let r = res(a, b), cost = r.reduce((s, v) => s + v * v, 0), lam = 1e-3;
  for (let it = 0; it < 60; it++) {
    const ea = 1e-4 * (Math.abs(a) + 1e-2), eb = 1e-4 * (Math.abs(b) + 1e-2);
    const ra = res(a + ea, b), rb = res(a, b + eb);
    let A00 = 0, A01 = 0, A11 = 0, g0 = 0, g1 = 0;
    r.forEach((ri, i) => {
      const j0 = (ra[i] - ri) / ea, j1 = (rb[i] - ri) / eb;
      A00 += j0 * j0; A01 += j0 * j1; A11 += j1 * j1; g0 += j0 * ri; g1 += j1 * ri;
    });
    let improved = false;
    for (let k = 0; k < 8; k++) {
      const m00 = A00 * (1 + lam) + 1e-12, m11 = A11 * (1 + lam) + 1e-12, det = m00 * m11 - A01 * A01;
      if (Math.abs(det) < 1e-18) { lam *= 10; continue; }
      const da = -(m11 * g0 - A01 * g1) / det, db = -(-A01 * g0 + m00 * g1) / det;
      const rn = res(a + da, b + db), cn = rn.reduce((s, v) => s + v * v, 0);
      if (cn < cost) { a += da; b += db; r = rn; cost = cn; lam = Math.max(lam / 3, 1e-9); improved = true; break; }
      lam *= 10;
    }
    if (!improved || cost < 1e-14) break;
  }
  if (items.some((it) => it.e.some((q) => a * q.d + b <= 1e-3)))
    return { ok: false, reason: "inconsistent depth ordering - check the pinned points and lengths" };
  const dm = items.map((it) => (it.e[0].d + it.e[1].d) / 2);
  const spread = Math.max(...dm) - Math.min(...dm);
  const notes = [];
  if (spread < 0.08) notes.push("references are at similar depths, so the result is unreliable");
  if (items.length === 2) notes.push("two references fit exactly, add a third to check the error");
  return { ok: true, a, b, errorsPct: r.map((v) => v * 100), warning: notes.join("; ") };
}

// ---------- Step 4: point cloud -> binary PLY with 3D Gaussian Splatting properties ----------
// metric = {a, b} from fitMetric(): points are in metres. Without it, depth is relative (arbitrary units).
export function buildPLY(img, disp, metric = null) {
  const { width: w, height: h, data: px } = img, K = intrinsics(w, h);
  const { lo, span } = normParams(disp);
  const nx = Math.ceil(w / STRIDE), ny = Math.ceil(h / STRIDE), n = nx * ny, P = 17;
  const d = new Float32Array(n * P);
  let i = 0;
  for (let v = 0; v < h; v += STRIDE) {
    for (let u = 0; u < w; u += STRIDE) {
      const dnu = (disp[v * w + u] - lo) / span;
      const z = metric ? 1 / Math.max(metric.a * dnu + metric.b, 1 / 500)                            // metres
        : 1 / (1 / FAR + Math.min(Math.max(dnu, 0), 1) * (1 / NEAR - 1 / FAR));                      // relative
      const o = i++ * P, p = (v * w + u) * 4;
      d[o] = ((u - K.cx) / K.fx) * z;                                 // x
      d[o + 1] = (-(v - K.cy) / K.fy) * z;                            // y (up)
      d[o + 2] = -z;                                                  // z (camera looks down -Z)
      for (let c = 0; c < 3; c++) d[o + 6 + c] = (px[p + c] / 255 - 0.5) / SH_C0; // f_dc_0..2
      d[o + 9] = Math.log(0.9 / 0.1);                                 // opacity (logit)
      const s = Math.log(Math.max((z / K.fx) * STRIDE * 0.7, 1e-5));
      d[o + 10] = d[o + 11] = d[o + 12] = s;                          // log scales
      d[o + 13] = 1;                                                  // rot_0 (w), rest 0
    }
  }
  const names = ["x","y","z","nx","ny","nz","f_dc_0","f_dc_1","f_dc_2","opacity","scale_0","scale_1","scale_2","rot_0","rot_1","rot_2","rot_3"];
  const head = new TextEncoder().encode("ply\nformat binary_little_endian 1.0\nelement vertex " + n + "\n" +
    names.map((p) => "property float " + p + "\n").join("") + "end_header\n");
  const out = new Uint8Array(head.length + d.byteLength);
  out.set(head, 0); out.set(new Uint8Array(d.buffer), head.length);
  return out.buffer;
}

// Returns a copy of a binary PLY without the vertices flagged in `del` (Uint8Array, one flag per vertex).
export function filterPLY(buf, del) {
  const head = new TextDecoder().decode(new Uint8Array(buf, 0, Math.min(buf.byteLength, 4096)));
  const end = head.indexOf("end_header\n") + 11;
  const props = head.slice(0, end).split("\n").filter((l) => l.startsWith("property")).length;
  const rec = props * 4;
  let keep = 0;
  for (let i = 0; i < del.length; i++) if (!del[i]) keep++;
  const hb = new TextEncoder().encode(head.slice(0, end).replace(/element vertex \d+/, "element vertex " + keep));
  const out = new Uint8Array(hb.length + keep * rec);
  out.set(hb, 0);
  const src = new Uint8Array(buf, end);
  let o = hb.length;
  for (let i = 0; i < del.length; i++) if (!del[i]) { out.set(src.subarray(i * rec, (i + 1) * rec), o); o += rec; }
  return out.buffer;
}

// ---------- Capture -> relative depth (kept in memory so the scale can be re-fitted later) ----------
export async function estimateDepth(source, sw, sh, onStatus = () => {}) {
  const s = Math.min(1, MAX_SIDE / Math.max(sw, sh)), w = Math.round(sw * s), h = Math.round(sh * s);
  const c = document.createElement("canvas"); c.width = w; c.height = h;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(source, 0, 0, w, h);

  onStatus("Correcting lens distortion…");
  const img = undistortImage(ctx.getImageData(0, 0, w, h));
  ctx.putImageData(new ImageData(img.data, w, h), 0, 0);

  const estimator = await loadModel(onStatus);
  onStatus("Estimating depth…");
  const blob = await new Promise((r) => c.toBlob(r, "image/png"));
  const url = URL.createObjectURL(blob);
  let out;
  try { out = await estimator(url); } finally { URL.revokeObjectURL(url); }
  const t = out.predicted_depth, [dh, dw] = t.dims.slice(-2);
  return { img, disp: resizeBilinear(t.data, dw, dh, w, h), w, h, W: sw, H: sh };
}

// Convenience: relative-depth PLY in one call (same as before).
export async function reconstruct(source, sw, sh, onStatus = () => {}) {
  const s = await estimateDepth(source, sw, sh, onStatus);
  onStatus("Building point cloud (.ply)…");
  return buildPLY(s.img, s.disp);
}
