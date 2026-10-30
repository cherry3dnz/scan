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

// ---------- Step 4: point cloud -> binary PLY with 3D Gaussian Splatting properties ----------
export function buildPLY(img, disp) {
  const { width: w, height: h, data: px } = img, K = intrinsics(w, h);
  const sorted = Float32Array.from(disp).sort();
  const lo = sorted[Math.floor(sorted.length * 0.02)], hi = sorted[Math.floor(sorted.length * 0.98)];
  const span = Math.max(hi - lo, 1e-6);
  const nx = Math.ceil(w / STRIDE), ny = Math.ceil(h / STRIDE), n = nx * ny, P = 17;
  const d = new Float32Array(n * P);
  let i = 0;
  for (let v = 0; v < h; v += STRIDE) {
    for (let u = 0; u < w; u += STRIDE) {
      const dn = Math.min(Math.max((disp[v * w + u] - lo) / span, 0), 1);
      const z = 1 / (1 / FAR + dn * (1 / NEAR - 1 / FAR));          // disparity ~ inverse depth
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

// ---------- Full pipeline from a <video>/<canvas> frame ----------
export async function reconstruct(source, sw, sh, onStatus = () => {}) {
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
  const disp = resizeBilinear(t.data, dw, dh, w, h);

  onStatus("Building point cloud (.ply)…");
  return buildPLY(img, disp);
}
