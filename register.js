// register.js - merge a new camera view into the master point cloud.
// Pipeline: ghost render of master (guide) -> depth fit to master -> coarse-to-fine rigid ICP -> depth-buffer merge.
// Conventions (same as pipeline.js): metres, camera looks down -Z, X right, Y up.
// A pose is camera-to-world: { R: [9 numbers, row-major], t: [x, y, z] }  =>  p_master = R * p_cam + t
import { intrinsics, normParams } from "./pipeline.js";
import { parsePLY } from "./splat.js";

const CFG = globalThis.APP_CONFIG || {};
const MAX_SIDE = CFG.MAX_SIDE || 640;
const STRIDE = 13;                       // floats per splat produced by parsePLY
const SH_C0 = 0.28209479177387814;
const REC = 17;                          // floats per vertex in the PLY written by buildPLY

export const DEFAULTS = {
  pointStride: 2,                        // sampling of the new depth map (pixels)
  rounds: 2,                             // fit -> ICP rounds
  minCoverage: 0.25,                     // fraction of the frame the master must cover at the guess pose
  minFitInliers: 0.5,                    // fraction of overlap pixels that agree with the fitted depth
  maxFitRelErr: 0.12,                    // median relative depth error of the fit
  minIcpInliers: 0.4,                    // fraction of ICP source points with a match
  maxIcpRmse: 0.03,                      // metres
  edgeTol: 0.08,                         // drop new points at depth jumps (relative)
  maxZ: 8,                               // drop new points farther than this (m)
  voxel: 0.01,                           // dedupe voxel for added points (m)
  tolAbs: 0.015, tolRel: 0.02,           // "same surface" tolerance in the depth test (m, fraction)
  // coarse-to-fine ICP. Capture radius of the first stage is cell (m): keep the guess pose within about that.
  stages: [{ cell: 0.12, iters: 10 }, { cell: 0.04, iters: 10 }, { cell: 0.015, iters: 10 }],
};

// ---------- small matrix helpers (3x3 row-major) ----------
const mulRV = (R, v) => [R[0] * v[0] + R[1] * v[1] + R[2] * v[2], R[3] * v[0] + R[4] * v[1] + R[5] * v[2], R[6] * v[0] + R[7] * v[1] + R[8] * v[2]];
const mulRR = (A, B) => { const o = new Array(9); for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) o[r * 3 + c] = A[r * 3] * B[c] + A[r * 3 + 1] * B[3 + c] + A[r * 3 + 2] * B[6 + c]; return o; };
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (v) => { const l = Math.hypot(v[0], v[1], v[2]) || 1; return [v[0] / l, v[1] / l, v[2] / l]; };

// ---------- 1. Pose from the 3D viewer (what the user chose to stand at) ----------
export function poseFromLookAt(eye, target, up = [0, 1, 0]) {
  const z = norm([eye[0] - target[0], eye[1] - target[1], eye[2] - target[2]]);   // camera backward (-Z looks at target)
  const x = norm(cross(up, z)), y = cross(z, x);
  return { R: [x[0], y[0], z[0], x[1], y[1], z[1], x[2], y[2], z[2]], t: [eye[0], eye[1], eye[2]] };
}
// Same camera maths as SplatViewer._frame. Set viewer.spinSpeed = 0 first so the view stays where the user left it.
export function poseFromViewer(v) {
  const D = v.dist * v.zoom, [tx, ty, tz] = v.target;
  const eye = [tx + D * Math.sin(v.theta) * Math.cos(v.phi), ty + D * Math.sin(v.phi), tz + D * Math.cos(v.theta) * Math.cos(v.phi)];
  return poseFromLookAt(eye, v.target);
}
export function workingSize(sw, sh) {   // same rule as estimateDepth()
  const s = Math.min(1, MAX_SIDE / Math.max(sw, sh));
  return { w: Math.round(sw * s), h: Math.round(sh * s) };
}

// ---------- 2. Synthetic camera image: CPU z-buffer of the master ----------
// master = parsePLY(buf). Returns colour (rgba), depth (m, 0 = empty), idx (master point per pixel), coverage 0..1.
export function renderSynthetic(master, pose, w, h) {
  const K = intrinsics(w, h), { data, count } = master, R = pose.R, t = pose.t;
  const depth = new Float32Array(w * h), idx = new Int32Array(w * h).fill(-1), rgba = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < count; i++) {
    const o = i * STRIDE, dx = data[o] - t[0], dy = data[o + 1] - t[1], dz = data[o + 2] - t[2];
    const z = -(R[2] * dx + R[5] * dy + R[8] * dz);
    if (z < 0.05) continue;
    const xc = R[0] * dx + R[3] * dy + R[6] * dz, yc = R[1] * dx + R[4] * dy + R[7] * dz;
    const u = Math.round((K.fx * xc) / z + K.cx), v = Math.round((-K.fy * yc) / z + K.cy);
    const rad = Math.min(3, Math.max(0, Math.round(Math.sqrt(data[o + 7]) * (K.fx / z) * 0.7)));   // splat footprint (px)
    for (let yy = v - rad; yy <= v + rad; yy++) {
      if (yy < 0 || yy >= h) continue;
      for (let xx = u - rad; xx <= u + rad; xx++) {
        if (xx < 0 || xx >= w) continue;
        const p = yy * w + xx;
        if (depth[p] === 0 || z < depth[p]) {
          depth[p] = z; idx[p] = i;
          rgba[p * 4] = data[o + 3] * 255; rgba[p * 4 + 1] = data[o + 4] * 255; rgba[p * 4 + 2] = data[o + 5] * 255; rgba[p * 4 + 3] = 255;
        }
      }
    }
  }
  let n = 0; for (let p = 0; p < w * h; p++) if (depth[p] > 0) n++;
  return { w, h, depth, idx, rgba, coverage: n / (w * h) };
}

// ---------- Ghost overlay for the live camera feed ----------
// mode "edges": cyan outlines (depth jumps, colour edges, coverage border). mode "image": faded colour render.
export function ghostImageData(synth, mode = "edges") {
  const { w, h, depth, rgba } = synth, out = new Uint8ClampedArray(w * h * 4);
  if (mode === "image") { out.set(rgba); for (let p = 3; p < out.length; p += 4) if (out[p]) out[p] = 255; return new ImageData(out, w, h); }
  const lum = (p) => rgba[p * 4] * 0.3 + rgba[p * 4 + 1] * 0.59 + rgba[p * 4 + 2] * 0.11;
  for (let y = 0; y < h - 1; y++) for (let x = 0; x < w - 1; x++) {
    const p = y * w + x, d = depth[p];
    let e = false;
    for (const q of [p + 1, p + w]) {
      if ((d > 0) !== (depth[q] > 0)) e = true;
      else if (d > 0 && (Math.abs(depth[q] - d) > 0.04 * d || Math.abs(lum(q) - lum(p)) > 40)) e = true;
    }
    if (e) for (const q of [p, p + 1, p + w]) { const o = q * 4; out[o] = 79; out[o + 1] = 224; out[o + 2] = 255; out[o + 3] = 255; }   // 2 px thick
  }
  return new ImageData(out, w, h);
}
// Draws the ghost on `canvas` (stacked over the <video>, same CSS box). Matches object-fit: contain of the video.
export function drawGhost(canvas, synth, { mode = "edges", alpha = 0.85, videoW, videoH } = {}) {
  const dpr = Math.min(window.devicePixelRatio || 1, 3), cw = canvas.clientWidth, ch = canvas.clientHeight;
  canvas.width = Math.round(cw * dpr); canvas.height = Math.round(ch * dpr);
  const g = canvas.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, cw, ch);
  const tmp = document.createElement("canvas"); tmp.width = synth.w; tmp.height = synth.h;
  tmp.getContext("2d").putImageData(ghostImageData(synth, mode), 0, 0);
  const sc = Math.min(cw / videoW, ch / videoH), dw = videoW * sc, dh = videoH * sc;
  g.globalAlpha = alpha; g.imageSmoothingEnabled = true;
  g.drawImage(tmp, (cw - dw) / 2, (ch - dh) / 2, dw, dh);
  g.globalAlpha = 1;
}

// ---------- 3. Fit the new photo's relative depth to the master: 1/Z = a*dn + b (robust) ----------
export function fitDepthToMaster(disp, synthDepth, w, h) {
  const { lo, span } = normParams(disp), xs = [], ys = [];
  for (let y = 1; y < h - 1; y += 2) for (let x = 1; x < w - 1; x += 2) {
    const p = y * w + x, d = synthDepth[p];
    if (!(d > 0)) continue;
    const n = [synthDepth[p + 1], synthDepth[p - 1], synthDepth[p + w], synthDepth[p - w]];
    if (n.some((q) => !(q > 0) || Math.abs(q - d) > 0.03 * d)) continue;     // skip master holes and depth jumps
    xs.push((disp[p] - lo) / span); ys.push(1 / d);
  }
  const n = xs.length;
  if (n < 800) return { ok: false, reason: "the photo overlaps the master too little to fit the scale" };
  let a = 0, b = 0, s = 1; const wt = new Float64Array(n).fill(1), res = new Float64Array(n);
  for (let it = 0; it < 20; it++) {
    let Sw = 0, Sx = 0, Sy = 0, Sxx = 0, Sxy = 0;
    for (let i = 0; i < n; i++) { const q = wt[i]; Sw += q; Sx += q * xs[i]; Sy += q * ys[i]; Sxx += q * xs[i] * xs[i]; Sxy += q * xs[i] * ys[i]; }
    const den = Sw * Sxx - Sx * Sx;
    if (Math.abs(den) < 1e-12) return { ok: false, reason: "the overlap has no depth variation to fit" };
    a = (Sw * Sxy - Sx * Sy) / den; b = (Sy - a * Sx) / Sw;
    for (let i = 0; i < n; i++) res[i] = Math.abs(ys[i] - a * xs[i] - b);
    const med = Float64Array.from(res).sort()[n >> 1];
    s = Math.max(1.4826 * med, 1e-4);
    for (let i = 0; i < n; i++) wt[i] = res[i] <= 1.5 * s ? 1 : (1.5 * s) / res[i];   // Huber weights
  }
  const rel = [];
  for (let i = 0; i < n; i++) if (res[i] < 2.5 * s) { const q = a * xs[i] + b; if (q <= 1e-3) return { ok: false, reason: "inconsistent depth ordering" }; rel.push(Math.abs(ys[i] / q - 1)); }
  rel.sort((p, q) => p - q);
  const inlierFrac = rel.length / n, relErr = rel.length ? rel[rel.length >> 1] : 1;
  if (!(a > 0)) return { ok: false, reason: "depth does not match the master (check the alignment)" };
  return { ok: true, a, b, inlierFrac, relErr, n };
}

// ---------- 4. New points from the photo (camera frame), with depth-jump removal ----------
export function newPointsFromDepth(img, disp, w, h, ab, o = DEFAULTS) {
  const K = intrinsics(w, h), { lo, span } = normParams(disp), st = o.pointStride, zm = new Float32Array(w * h);
  for (let p = 0; p < w * h; p++) zm[p] = 1 / Math.max(ab.a * ((disp[p] - lo) / span) + ab.b, 1 / 500);
  const cap = Math.ceil(w / st) * Math.ceil(h / st), pos = new Float32Array(cap * 3), col = new Uint8Array(cap * 3), zs = new Float32Array(cap), pix = new Int32Array(cap);
  let n = 0;
  for (let v = st; v < h - st; v += st) for (let u = st; u < w - st; u += st) {
    const p = v * w + u, z = zm[p];
    if (z > o.maxZ) continue;
    const nb = [zm[p + st], zm[p - st], zm[p + st * w], zm[p - st * w]];
    if (nb.some((q) => Math.abs(q - z) > o.edgeTol * z)) continue;
    pos[n * 3] = ((u - K.cx) / K.fx) * z; pos[n * 3 + 1] = (-(v - K.cy) / K.fy) * z; pos[n * 3 + 2] = -z;
    col[n * 3] = img.data[p * 4]; col[n * 3 + 1] = img.data[p * 4 + 1]; col[n * 3 + 2] = img.data[p * 4 + 2];
    zs[n] = z; pix[n] = p; n++;
  }
  return { n, pos: pos.subarray(0, n * 3), col: col.subarray(0, n * 3), z: zs.subarray(0, n), pix: pix.subarray(0, n) };
}
export function transformPoints(pos, pose) {
  const out = new Float32Array(pos.length), { R, t } = pose;
  for (let i = 0; i < pos.length; i += 3) {
    const x = pos[i], y = pos[i + 1], z = pos[i + 2];
    out[i] = R[0] * x + R[1] * y + R[2] * z + t[0]; out[i + 1] = R[3] * x + R[4] * y + R[5] * z + t[1]; out[i + 2] = R[6] * x + R[7] * y + R[8] * z + t[2];
  }
  return out;
}

// ---------- 5. Rigid ICP (coarse -> fine) ----------
const cellKey = (ix, iy, iz) => ((Math.max(-1023, Math.min(1023, ix)) + 1024) * 2048 + (Math.max(-1023, Math.min(1023, iy)) + 1024)) * 2048 + (Math.max(-1023, Math.min(1023, iz)) + 1024);
function buildGrid(master, cell) {
  const { data, count } = master, map = new Map(), inv = 1 / cell;
  for (let i = 0; i < count; i++) {
    const o = i * STRIDE, k = cellKey(Math.floor(data[o] * inv), Math.floor(data[o + 1] * inv), Math.floor(data[o + 2] * inv));
    const l = map.get(k); if (l) l.push(i); else map.set(k, [i]);
  }
  return { cell, map };
}
function nearest(grid, data, x, y, z, maxD2) {
  const inv = 1 / grid.cell, ix = Math.floor(x * inv), iy = Math.floor(y * inv), iz = Math.floor(z * inv);
  let best = -1, bd = maxD2;
  for (let a = -1; a <= 1; a++) for (let b = -1; b <= 1; b++) for (let c = -1; c <= 1; c++) {
    const l = grid.map.get(cellKey(ix + a, iy + b, iz + c)); if (!l) continue;
    for (let j = 0; j < l.length; j++) {
      const o = l[j] * STRIDE, dx = data[o] - x, dy = data[o + 1] - y, dz = data[o + 2] - z, d2 = dx * dx + dy * dy + dz * dz;
      if (d2 < bd) { bd = d2; best = l[j]; }
    }
  }
  return best;
}
function jacobi4(A) {                       // eigen-decomposition of a symmetric 4x4
  const a = A.map((r) => r.slice()), V = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0; for (let i = 0; i < 4; i++) for (let j = i + 1; j < 4; j++) off += a[i][j] * a[i][j];
    if (off < 1e-20) break;
    for (let p = 0; p < 3; p++) for (let q = p + 1; q < 4; q++) {
      if (Math.abs(a[p][q]) < 1e-300) continue;
      const th = (a[q][q] - a[p][p]) / (2 * a[p][q]), t = (th >= 0 ? 1 : -1) / (Math.abs(th) + Math.sqrt(th * th + 1));
      const c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < 4; k++) { const x = a[k][p], y = a[k][q]; a[k][p] = c * x - s * y; a[k][q] = s * x + c * y; }
      for (let k = 0; k < 4; k++) { const x = a[p][k], y = a[q][k]; a[p][k] = c * x - s * y; a[q][k] = s * x + c * y; }
      for (let k = 0; k < 4; k++) { const x = V[k][p], y = V[k][q]; V[k][p] = c * x - s * y; V[k][q] = s * x + c * y; }
    }
  }
  let m = 0; for (let i = 1; i < 4; i++) if (a[i][i] > a[m][m]) m = i;
  return [V[0][m], V[1][m], V[2][m], V[3][m]];       // eigenvector of the largest eigenvalue
}
function horn(P, Q, k) {                    // best rigid transform P -> Q (Horn's quaternion method)
  const pc = [0, 0, 0], qc = [0, 0, 0];
  for (let i = 0; i < k; i++) for (let c = 0; c < 3; c++) { pc[c] += P[i * 3 + c] / k; qc[c] += Q[i * 3 + c] / k; }
  const S = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < k; i++) for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) S[a][b] += (P[i * 3 + a] - pc[a]) * (Q[i * 3 + b] - qc[b]);
  const [Sxx, Sxy, Sxz] = S[0], [Syx, Syy, Syz] = S[1], [Szx, Szy, Szz] = S[2];
  const q = jacobi4([[Sxx + Syy + Szz, Syz - Szy, Szx - Sxz, Sxy - Syx],
    [Syz - Szy, Sxx - Syy - Szz, Sxy + Syx, Szx + Sxz],
    [Szx - Sxz, Sxy + Syx, -Sxx + Syy - Szz, Syz + Szy],
    [Sxy - Syx, Szx + Sxz, Syz + Szy, -Sxx - Syy + Szz]]);
  const l = Math.hypot(...q), [w, x, y, z] = q.map((v) => v / l);
  const R = [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
    2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
    2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)];
  const Rp = mulRV(R, pc);
  return { R, t: [qc[0] - Rp[0], qc[1] - Rp[1], qc[2] - Rp[2]] };
}
// src: Float32Array xyz in the master frame (already placed with the guess pose). Returns the correction T so
// that T(src) ~ master: p' = R p + t.
export function icp(master, src, stages = DEFAULTS.stages, maxPts = 8000, trim = 0.8) {
  const step = Math.max(1, Math.floor(src.length / 3 / maxPts)), m = Math.floor(src.length / 3 / step);
  if (m < 100) return { ok: false, reason: "too few points for ICP" };
  const cur = new Float64Array(m * 3);
  for (let i = 0; i < m; i++) for (let c = 0; c < 3; c++) cur[i * 3 + c] = src[i * step * 3 + c];
  let Rt = [1, 0, 0, 0, 1, 0, 0, 0, 1], tt = [0, 0, 0], rmse = Infinity, frac = 0;
  for (const st of stages) {
    const grid = buildGrid(master, st.cell), maxD2 = st.cell * st.cell;
    for (let it = 0; it < st.iters; it++) {
      const si = [], tg = [], d2s = [];
      for (let i = 0; i < m; i++) {
        const x = cur[i * 3], y = cur[i * 3 + 1], z = cur[i * 3 + 2], j = nearest(grid, master.data, x, y, z, maxD2);
        if (j < 0) continue;
        const o = j * STRIDE, dx = master.data[o] - x, dy = master.data[o + 1] - y, dz = master.data[o + 2] - z;
        si.push(i); tg.push(j); d2s.push(dx * dx + dy * dy + dz * dz);
      }
      frac = si.length / m;
      if (si.length < 100) return { ok: false, reason: "no overlap found between the photo and the master" };
      const ord = si.map((_, i) => i).sort((p, q) => d2s[p] - d2s[q]).slice(0, Math.max(50, Math.floor(si.length * trim)));
      const P = new Float64Array(ord.length * 3), Q = new Float64Array(ord.length * 3);
      let sum = 0;
      ord.forEach((k, r) => {
        for (let c = 0; c < 3; c++) { P[r * 3 + c] = cur[si[k] * 3 + c]; Q[r * 3 + c] = master.data[tg[k] * STRIDE + c]; }
        sum += d2s[k];
      });
      rmse = Math.sqrt(sum / ord.length);
      const T = horn(P, Q, ord.length);
      for (let i = 0; i < m; i++) { const v = mulRV(T.R, [cur[i * 3], cur[i * 3 + 1], cur[i * 3 + 2]]); cur[i * 3] = v[0] + T.t[0]; cur[i * 3 + 1] = v[1] + T.t[1]; cur[i * 3 + 2] = v[2] + T.t[2]; }
      Rt = mulRR(T.R, Rt); const tv = mulRV(T.R, tt); tt = [tv[0] + T.t[0], tv[1] + T.t[1], tv[2] + T.t[2]];
      const ang = Math.acos(Math.min(1, Math.max(-1, (T.R[0] + T.R[4] + T.R[8] - 1) / 2)));
      if (ang < 1e-4 && Math.hypot(...T.t) < 1e-4) break;
    }
  }
  return { ok: true, R: Rt, t: tt, rmse, frac };
}
export const composePose = (delta, pose) => ({ R: mulRR(delta.R, pose.R), t: (() => { const v = mulRV(delta.R, pose.t); return [v[0] + delta.t[0], v[1] + delta.t[1], v[2] + delta.t[2]]; })() });

// ---------- 6. Merge: depth-buffer test against the master, then append to the PLY ----------
export function appendPLY(buf, rec, nNew) {
  const head = new TextDecoder().decode(new Uint8Array(buf, 0, Math.min(buf.byteLength, 4096)));
  const end = head.indexOf("end_header\n") + 11;
  const props = head.slice(0, end).split("\n").filter((l) => l.startsWith("property")).length;
  if (props !== REC) throw new Error("Unexpected PLY layout (" + props + " properties)");
  const old = parseInt(/element vertex (\d+)/.exec(head)[1], 10);
  const hb = new TextEncoder().encode(head.slice(0, end).replace(/element vertex \d+/, "element vertex " + (old + nNew)));
  const body = new Uint8Array(buf, end, old * REC * 4);
  const out = new Uint8Array(hb.length + body.length + nNew * REC * 4);
  out.set(hb, 0); out.set(body, hb.length); out.set(new Uint8Array(rec.buffer, rec.byteOffset, nNew * REC * 4), hb.length + body.length);
  return out.buffer;
}
// np: points from newPointsFromDepth(); pose: final camera pose. Returns { buf, added, same, behind, none, front }.
export function mergeInto(masterBuf, master, np, pose, w, h, o = DEFAULTS) {
  const K = intrinsics(w, h), synth = renderSynthetic(master, pose, w, h), P = transformPoints(np.pos, pose);
  const rec = new Float32Array(np.n * REC), seen = new Set(), inv = 1 / o.voxel;
  const st = { added: 0, same: 0, behind: 0, none: 0, front: 0 };
  for (let i = 0; i < np.n; i++) {
    const z = np.z[i], x = np.pix[i] % w, y = (np.pix[i] / w) | 0, tol = Math.max(o.tolAbs, o.tolRel * z);
    let covered = false, near = false, minD = Infinity;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      const xx = x + dx, yy = y + dy; if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
      const d = synth.depth[yy * w + xx]; if (!(d > 0)) continue;
      covered = true; minD = Math.min(minD, d); if (Math.abs(d - z) <= tol) near = true;
    }
    let kind;
    if (!covered) kind = "none"; else if (near) { st.same++; continue; } else if (z < minD) kind = "front"; else { st.behind++; continue; }
    const key = cellKey(Math.floor(P[i * 3] * inv), Math.floor(P[i * 3 + 1] * inv), Math.floor(P[i * 3 + 2] * inv));
    if (seen.has(key)) continue; seen.add(key);
    const r = st.added++ * REC;
    st[kind]++;
    rec[r] = P[i * 3]; rec[r + 1] = P[i * 3 + 1]; rec[r + 2] = P[i * 3 + 2];
    for (let c = 0; c < 3; c++) rec[r + 6 + c] = (np.col[i * 3 + c] / 255 - 0.5) / SH_C0;
    rec[r + 9] = Math.log(9);
    rec[r + 10] = rec[r + 11] = rec[r + 12] = Math.log(Math.max((z / K.fx) * o.pointStride * 0.7, 1e-5));
    rec[r + 13] = 1;
  }
  return { buf: st.added ? appendPLY(masterBuf, rec, st.added) : masterBuf, ...st };
}

// ---------- Orchestrator ----------
// masterBuf: master PLY (ArrayBuffer). view: result of estimateDepth() for the new photo { img, disp, w, h }.
// pose0: pose the ghost was rendered from (poseFromViewer at capture time).
export function registerView(masterBuf, view, pose0, opts = {}) {
  const o = { ...DEFAULTS, ...opts }, { img, disp, w, h } = view, master = parsePLY(masterBuf);
  const fail = (reason, extra = {}) => ({ ok: false, reason, ...extra });
  let pose = pose0, fit = null, ic = null;
  for (let round = 0; round < o.rounds; round++) {
    const synth = renderSynthetic(master, pose, w, h);
    if (round === 0 && synth.coverage < o.minCoverage)
      return fail(`the master covers only ${(synth.coverage * 100).toFixed(0)}% of this view - choose a view that shows more of it`);
    fit = fitDepthToMaster(disp, synth.depth, w, h);
    if (!fit.ok) return fail(fit.reason);
    if (fit.inlierFrac < o.minFitInliers || fit.relErr > o.maxFitRelErr)
      return fail(`depth does not agree with the master (${(fit.inlierFrac * 100).toFixed(0)}% agree, ${(fit.relErr * 100).toFixed(1)}% error) - line up with the ghost and retry`, { fit });
    const np = newPointsFromDepth(img, disp, w, h, fit, o);
    // ICP only on points whose pixel the master covers: new geometry has no counterpart and must not count as a mismatch
    const ov = [];
    for (let i = 0; i < np.n; i++) {
      const x = np.pix[i] % w, y = (np.pix[i] / w) | 0;
      let cov = false;
      for (let dy = -2; dy <= 2 && !cov; dy += 2) for (let dx = -2; dx <= 2; dx += 2) {
        const xx = x + dx, yy = y + dy;
        if (xx >= 0 && yy >= 0 && xx < w && yy < h && synth.depth[yy * w + xx] > 0) { cov = true; break; }
      }
      if (cov) ov.push(np.pos[i * 3], np.pos[i * 3 + 1], np.pos[i * 3 + 2]);
    }
    ic = icp(master, transformPoints(Float32Array.from(ov), pose), o.stages);
    if (!ic.ok) return fail(ic.reason);
    pose = composePose(ic, pose);
  }
  if (ic.frac < o.minIcpInliers || ic.rmse > o.maxIcpRmse)
    return fail(`could not register the view (match ${(ic.frac * 100).toFixed(0)}%, residual ${(ic.rmse * 1000).toFixed(0)} mm) - retake closer to the ghost`, { fit, icp: ic });
  const np = newPointsFromDepth(img, disp, w, h, fit, o);       // final points: last fit, final pose
  const m = mergeInto(masterBuf, master, np, pose, w, h, o);
  return { ok: true, merged: m.buf, pose, fit, icp: { rmse: ic.rmse, frac: ic.frac }, added: m.added, same: m.same, behind: m.behind, front: m.front, none: m.none, candidates: np.n };
}
