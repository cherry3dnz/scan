// cloudops.js - point-cloud registration, comparison and merging, entirely in the browser.
// No dependencies, no DOM. Works on the PLY layout that pipeline.js buildPLY() writes (17 float32 per vertex), and reads
// other binary PLY files (xyz + uchar rgb, or full 3DGS files) by mapping them onto that layout.
//
// Conventions (same as pipeline.js / register.js): metres, camera looks down -Z, X right, Y up.
// A transform is { R: [9 numbers, row-major], t: [x, y, z], s?: scale (default 1) }  =>  p' = s * R * p + t
// compose(A, B) applies B first, then A (same as register.js composePose(delta, pose)).
//
// The same three jobs the WebPointCloud server does with scipy / NumPy, done client-side:
//   Registration : KDTree + ICP (point-to-plane or point-to-point, trimmed, coarse-to-fine, optional scale) + PCA coarse start
//   Comparison   : cloud-to-cloud distance (C2C), signed change, statistics, histogram, heat colours, overlay PLY
//   Merging      : keep only the points the master does not already cover, match colours, drop outliers (SOR), write PLY

export const SH_C0 = 0.28209479177387814;
export const NAMES17 = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"];
const P17 = 17;
const LOGIT09 = Math.log(0.9 / 0.1);

// =====================================================================================================================
// 1. Transforms
// =====================================================================================================================
export const identity = () => ({ R: [1, 0, 0, 0, 1, 0, 0, 0, 1], t: [0, 0, 0], s: 1 });
const mulRV = (R, v) => [R[0] * v[0] + R[1] * v[1] + R[2] * v[2], R[3] * v[0] + R[4] * v[1] + R[5] * v[2], R[6] * v[0] + R[7] * v[1] + R[8] * v[2]];
const mulRR = (A, B) => { const o = new Array(9); for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) o[r * 3 + c] = A[r * 3] * B[c] + A[r * 3 + 1] * B[3 + c] + A[r * 3 + 2] * B[6 + c]; return o; };
const transpose = (R) => [R[0], R[3], R[6], R[1], R[4], R[7], R[2], R[5], R[8]];

export function compose(A, B) {                       // apply B, then A
  const sa = A.s || 1, sb = B.s || 1, v = mulRV(A.R, B.t);
  return { R: mulRR(A.R, B.R), t: [sa * v[0] + A.t[0], sa * v[1] + A.t[1], sa * v[2] + A.t[2]], s: sa * sb };
}
export function invert(T) {
  const s = 1 / (T.s || 1), Rt = transpose(T.R), v = mulRV(Rt, T.t);
  return { R: Rt, t: [-s * v[0], -s * v[1], -s * v[2]], s };
}
export function rotAxis(ax, ay, az) {                 // Rodrigues: rotation vector -> matrix
  const a = Math.hypot(ax, ay, az); if (a < 1e-12) return [1, 0, 0, 0, 1, 0, 0, 0, 1];
  const x = ax / a, y = ay / a, z = az / a, c = Math.cos(a), s = Math.sin(a), C = 1 - c;
  return [c + x * x * C, x * y * C - z * s, x * z * C + y * s, y * x * C + z * s, c + y * y * C, y * z * C - x * s, z * x * C - y * s, z * y * C + x * s, c + z * z * C];
}
export const rotationAngle = (R) => Math.acos(Math.min(1, Math.max(-1, (R[0] + R[4] + R[8] - 1) / 2)));
export function applyToPositions(T, pos) {
  const out = new Float32Array(pos.length), R = T.R, t = T.t, s = T.s || 1;
  for (let i = 0; i < pos.length; i += 3) {
    const x = pos[i], y = pos[i + 1], z = pos[i + 2];
    out[i] = s * (R[0] * x + R[1] * y + R[2] * z) + t[0]; out[i + 1] = s * (R[3] * x + R[4] * y + R[5] * z) + t[1]; out[i + 2] = s * (R[6] * x + R[7] * y + R[8] * z) + t[2];
  }
  return out;
}
function quatFromR(R) {                               // -> [w, x, y, z]
  const tr = R[0] + R[4] + R[8]; let w, x, y, z;
  if (tr > 0) { const S = Math.sqrt(tr + 1) * 2; w = S / 4; x = (R[7] - R[5]) / S; y = (R[2] - R[6]) / S; z = (R[3] - R[1]) / S; }
  else if (R[0] > R[4] && R[0] > R[8]) { const S = Math.sqrt(1 + R[0] - R[4] - R[8]) * 2; w = (R[7] - R[5]) / S; x = S / 4; y = (R[1] + R[3]) / S; z = (R[2] + R[6]) / S; }
  else if (R[4] > R[8]) { const S = Math.sqrt(1 + R[4] - R[0] - R[8]) * 2; w = (R[2] - R[6]) / S; x = (R[1] + R[3]) / S; y = S / 4; z = (R[5] + R[7]) / S; }
  else { const S = Math.sqrt(1 + R[8] - R[0] - R[4]) * 2; w = (R[3] - R[1]) / S; x = (R[2] + R[6]) / S; y = (R[5] + R[7]) / S; z = S / 4; }
  const l = Math.hypot(w, x, y, z); return [w / l, x / l, y / l, z / l];
}

// =====================================================================================================================
// 2. Small linear algebra: symmetric 3x3 eigen (Jacobi), Horn's quaternion method, SPD solve
// =====================================================================================================================
function jacobi3(a, V) {                              // a: Float64Array(9) symmetric, destroyed. V <- eigenvectors as columns
  V[0] = 1; V[1] = 0; V[2] = 0; V[3] = 0; V[4] = 1; V[5] = 0; V[6] = 0; V[7] = 0; V[8] = 1;
  for (let sweep = 0; sweep < 30; sweep++) {
    const off = a[1] * a[1] + a[2] * a[2] + a[5] * a[5], dg = a[0] * a[0] + a[4] * a[4] + a[8] * a[8];
    if (off <= 1e-26 * dg || off < 1e-300) break;
    for (let pq = 0; pq < 3; pq++) {
      const p = pq === 2 ? 1 : 0, q = pq === 0 ? 1 : 2, apq = a[p * 3 + q];
      if (Math.abs(apq) < 1e-300) continue;
      const th = (a[q * 3 + q] - a[p * 3 + p]) / (2 * apq), t = (th >= 0 ? 1 : -1) / (Math.abs(th) + Math.sqrt(th * th + 1));
      const c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < 3; k++) { const x = a[k * 3 + p], y = a[k * 3 + q]; a[k * 3 + p] = c * x - s * y; a[k * 3 + q] = s * x + c * y; }
      for (let k = 0; k < 3; k++) { const x = a[p * 3 + k], y = a[q * 3 + k]; a[p * 3 + k] = c * x - s * y; a[q * 3 + k] = s * x + c * y; }
      for (let k = 0; k < 3; k++) { const x = V[k * 3 + p], y = V[k * 3 + q]; V[k * 3 + p] = c * x - s * y; V[k * 3 + q] = s * x + c * y; }
    }
  }
}
function jacobi4(A) {                                 // largest-eigenvalue eigenvector of a symmetric 4x4
  const a = A.map((r) => r.slice()), V = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0; for (let i = 0; i < 4; i++) for (let j = i + 1; j < 4; j++) off += a[i][j] * a[i][j];
    if (off < 1e-24) break;
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
  return [V[0][m], V[1][m], V[2][m], V[3][m]];
}
// Best rotation (and optionally scale) taking centred P onto centred Q, Horn 1987. P, Q: Float64Array xyz, k points.
function horn(P, Q, k, withScale) {
  const pc = [0, 0, 0], qc = [0, 0, 0];
  for (let i = 0; i < k; i++) for (let c = 0; c < 3; c++) { pc[c] += P[i * 3 + c] / k; qc[c] += Q[i * 3 + c] / k; }
  const S = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]; let sp = 0;
  for (let i = 0; i < k; i++) {
    const px = P[i * 3] - pc[0], py = P[i * 3 + 1] - pc[1], pz = P[i * 3 + 2] - pc[2], p = [px, py, pz];
    sp += px * px + py * py + pz * pz;
    for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) S[a][b] += p[a] * (Q[i * 3 + b] - qc[b]);
  }
  const [Sxx, Sxy, Sxz] = S[0], [Syx, Syy, Syz] = S[1], [Szx, Szy, Szz] = S[2];
  const q = jacobi4([[Sxx + Syy + Szz, Syz - Szy, Szx - Sxz, Sxy - Syx], [Syz - Szy, Sxx - Syy - Szz, Sxy + Syx, Szx + Sxz],
    [Szx - Sxz, Sxy + Syx, -Sxx + Syy - Szz, Syz + Szy], [Sxy - Syx, Szx + Sxz, Syz + Szy, -Sxx - Syy + Szz]]);
  const l = Math.hypot(...q), [w, x, y, z] = q.map((v) => v / l);
  const R = [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y), 2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
    2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)];
  let s = 1;
  if (withScale && sp > 1e-18) {                      // least-squares scale given R: s = sum(q' . R p') / sum(|p'|^2)
    let num = 0;
    for (let i = 0; i < k; i++) {
      const v = mulRV(R, [P[i * 3] - pc[0], P[i * 3 + 1] - pc[1], P[i * 3 + 2] - pc[2]]);
      num += v[0] * (Q[i * 3] - qc[0]) + v[1] * (Q[i * 3 + 1] - qc[1]) + v[2] * (Q[i * 3 + 2] - qc[2]);
    }
    s = Math.max(num / sp, 1e-6);
  }
  const Rp = mulRV(R, pc);
  return { R, s, t: [qc[0] - s * Rp[0], qc[1] - s * Rp[1], qc[2] - s * Rp[2]] };
}
function solveSPD(A, b, n) {                          // Cholesky, in place on copies. Returns { x, cond } or null
  const L = Float64Array.from(A), x = Float64Array.from(b);
  let mn = Infinity, mx = 0;
  for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) {
    let s = L[i * n + j]; for (let k = 0; k < j; k++) s -= L[i * n + k] * L[j * n + k];
    if (i === j) { if (!(s > 1e-18)) return null; const d = Math.sqrt(s); L[i * n + i] = d; mn = Math.min(mn, d); mx = Math.max(mx, d); } else L[i * n + j] = s / L[j * n + j];
  }
  for (let i = 0; i < n; i++) { let s = x[i]; for (let k = 0; k < i; k++) s -= L[i * n + k] * x[k]; x[i] = s / L[i * n + i]; }
  for (let i = n - 1; i >= 0; i--) { let s = x[i]; for (let k = i + 1; k < n; k++) s -= L[k * n + i] * x[k]; x[i] = s / L[i * n + i]; }
  return { x, cond: (mn / mx) ** 2 };
}

// =====================================================================================================================
// 3. KDTree - exact nearest / k-nearest neighbours (replaces scipy cKDTree)
// =====================================================================================================================
function select(idx, pos, ax, left, right, k) {       // quickselect: idx[k] gets the k-th smallest along axis ax
  while (right > left) {
    const mid = (left + right) >> 1, a = pos[idx[left] * 3 + ax], b = pos[idx[mid] * 3 + ax], c = pos[idx[right] * 3 + ax];
    const pv = a < b ? (b < c ? b : a < c ? c : a) : (a < c ? a : b < c ? c : b);
    let i = left, j = right;
    while (i <= j) {
      while (pos[idx[i] * 3 + ax] < pv) i++;
      while (pos[idx[j] * 3 + ax] > pv) j--;
      if (i <= j) { const t = idx[i]; idx[i] = idx[j]; idx[j] = t; i++; j--; }
    }
    if (k <= j) right = j; else if (k >= i) left = i; else return;
  }
}
export class KDTree {
  // pos: Float32Array/Float64Array xyz (finite values). The array is referenced, not copied.
  constructor(pos, leaf = 10) {
    this.pos = pos; this.n = (pos.length / 3) | 0;
    this.idx = new Uint32Array(this.n); for (let i = 0; i < this.n; i++) this.idx[i] = i;
    this.lo = []; this.hi = []; this.ax = []; this.sp = []; this.l = []; this.r = [];
    this._bi = -1; this._bd = Infinity;
    if (this.n) this._build(leaf);
    this.lo = Int32Array.from(this.lo); this.hi = Int32Array.from(this.hi); this.ax = Int8Array.from(this.ax);
    this.sp = Float64Array.from(this.sp); this.l = Int32Array.from(this.l); this.r = Int32Array.from(this.r);
  }
  _build(leaf) {
    const { pos, idx, lo, hi, ax, sp, l, r } = this, stack = [[0, this.n, -1, 0]];
    while (stack.length) {
      const [a, b, parent, side] = stack.pop(), id = lo.length;
      lo.push(a); hi.push(b); ax.push(-1); sp.push(0); l.push(-1); r.push(-1);
      if (parent >= 0) (side ? r : l)[parent] = id;
      if (b - a <= leaf) continue;
      let x0 = Infinity, y0 = Infinity, z0 = Infinity, x1 = -Infinity, y1 = -Infinity, z1 = -Infinity;
      for (let i = a; i < b; i++) {
        const o = idx[i] * 3, x = pos[o], y = pos[o + 1], z = pos[o + 2];
        if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; if (z < z0) z0 = z; if (z > z1) z1 = z;
      }
      const ex = x1 - x0, ey = y1 - y0, ez = z1 - z0, k = ex >= ey && ex >= ez ? 0 : ey >= ez ? 1 : 2, m = (a + b) >> 1;
      select(idx, pos, k, a, b - 1, m);
      ax[id] = k; sp[id] = pos[idx[m] * 3 + k];
      stack.push([m, b, id, 1], [a, m, id, 0]);
    }
  }
  // Index of the nearest point closer than sqrt(maxD2), or -1. The squared distance is left in tree._bd.
  nearest(x, y, z, maxD2 = Infinity) {
    this._bi = -1; this._bd = maxD2;
    if (this.n) this._nn(0, x, y, z);
    return this._bi;
  }
  _nn(node, x, y, z) {
    const k = this.ax[node];
    if (k < 0) {
      const { pos, idx } = this;
      for (let i = this.lo[node], e = this.hi[node]; i < e; i++) {
        const o = idx[i] * 3, dx = pos[o] - x, dy = pos[o + 1] - y, dz = pos[o + 2] - z, d2 = dx * dx + dy * dy + dz * dz;
        if (d2 < this._bd) { this._bd = d2; this._bi = idx[i]; }
      }
      return;
    }
    const diff = (k === 0 ? x : k === 1 ? y : z) - this.sp[node];
    if (diff < 0) { this._nn(this.l[node], x, y, z); if (diff * diff < this._bd) this._nn(this.r[node], x, y, z); }
    else { this._nn(this.r[node], x, y, z); if (diff * diff < this._bd) this._nn(this.l[node], x, y, z); }
  }
  // k nearest points, ascending. Fills outIdx / outD2 (length >= k); returns how many were found.
  knn(x, y, z, k, outIdx, outD2, maxD2 = Infinity) {
    this._kc = 0; this._kk = k; this._ki = outIdx; this._kd = outD2; this._kmax = maxD2;
    if (this.n) this._kn(0, x, y, z);
    return this._kc;
  }
  _kn(node, x, y, z) {
    const k = this.ax[node], K = this._kk;
    if (k < 0) {
      const { pos, idx } = this, oi = this._ki, od = this._kd;
      for (let i = this.lo[node], e = this.hi[node]; i < e; i++) {
        const o = idx[i] * 3, dx = pos[o] - x, dy = pos[o + 1] - y, dz = pos[o + 2] - z, d2 = dx * dx + dy * dy + dz * dz;
        if (d2 >= this._kmax || (this._kc === K && d2 >= od[K - 1])) continue;
        let j = this._kc < K ? this._kc++ : K - 1;
        while (j > 0 && od[j - 1] > d2) { od[j] = od[j - 1]; oi[j] = oi[j - 1]; j--; }
        od[j] = d2; oi[j] = idx[i];
      }
      return;
    }
    const diff = (k === 0 ? x : k === 1 ? y : z) - this.sp[node], near = diff < 0 ? this.l[node] : this.r[node], far = diff < 0 ? this.r[node] : this.l[node];
    this._kn(near, x, y, z);
    const bound = this._kc === K ? Math.min(this._kd[K - 1], this._kmax) : this._kmax;
    if (diff * diff < bound) this._kn(far, x, y, z);
  }
}

// Median distance from a sample of points to their nearest other point: the typical point spacing.
export function estimateSpacing(tree, sample = 400) {
  const n = tree.n; if (n < 3) return 0.01;
  const step = Math.max(1, Math.floor(n / sample)), oi = new Int32Array(2), od = new Float64Array(2), d = [];
  for (let i = 0; i < n; i += step) { const o = i * 3; if (tree.knn(tree.pos[o], tree.pos[o + 1], tree.pos[o + 2], 2, oi, od) === 2) d.push(Math.sqrt(od[1])); }
  d.sort((a, b) => a - b); return Math.max(d.length ? d[d.length >> 1] : 0.01, 1e-5);   // duplicates give 0: floor it
}

// Lazy surface normals: smallest eigenvector of the covariance of the k nearest points, computed only when asked for.
export class NormalCache {
  constructor(tree, k = 12) {
    this.tree = tree; this.k = k; this.nrm = new Float32Array(tree.n * 3); this.curv = new Float32Array(tree.n);
    this.state = new Uint8Array(tree.n);              // 0 = not computed, 1 = ok, 2 = unreliable
    this._oi = new Int32Array(k); this._od = new Float64Array(k); this._A = new Float64Array(9); this._V = new Float64Array(9);
  }
  // Returns true when the normal of point i is reliable; the normal is then in this.nrm[3i..3i+2], curvature in this.curv[i].
  ok(i) {
    if (this.state[i] === 0) this._compute(i);
    return this.state[i] === 1;
  }
  _compute(i) {
    const { tree, k, _oi: oi, _od: od, _A: A, _V: V } = this, p = tree.pos, o = i * 3;
    const c = tree.knn(p[o], p[o + 1], p[o + 2], k, oi, od);
    if (c < 4) { this.state[i] = 2; return; }
    let mx = 0, my = 0, mz = 0;
    for (let j = 0; j < c; j++) { const q = oi[j] * 3; mx += p[q]; my += p[q + 1]; mz += p[q + 2]; }
    mx /= c; my /= c; mz /= c;
    let xx = 0, xy = 0, xz = 0, yy = 0, yz = 0, zz = 0;
    for (let j = 0; j < c; j++) {
      const q = oi[j] * 3, dx = p[q] - mx, dy = p[q + 1] - my, dz = p[q + 2] - mz;
      xx += dx * dx; xy += dx * dy; xz += dx * dz; yy += dy * dy; yz += dy * dz; zz += dz * dz;
    }
    A[0] = xx; A[1] = xy; A[2] = xz; A[3] = xy; A[4] = yy; A[5] = yz; A[6] = xz; A[7] = yz; A[8] = zz;
    jacobi3(A, V);
    let m = 0; if (A[4] < A[m * 4]) m = 1; if (A[8] < A[m * 4]) m = 2;
    const tr = A[0] + A[4] + A[8];
    if (!(tr > 1e-18)) { this.state[i] = 2; return; }
    this.nrm[o] = V[m]; this.nrm[o + 1] = V[3 + m]; this.nrm[o + 2] = V[6 + m];
    this.curv[i] = A[m * 4] / tr; this.state[i] = 1;
  }
}

// A registration / comparison target: positions + tree + lazy normals.
export function makeTarget(pos, opts = {}) {
  const tree = new KDTree(pos); return { pos, tree, normals: new NormalCache(tree, opts.normalK || 12) };
}
// Accepts a makeTarget() result, a KDTree, a cloud ({ pos }) or a bare Float32Array of xyz.
export function asTarget(x) {
  if (x.tree && x.normals) return x;
  if (x instanceof KDTree) return { pos: x.pos, tree: x, normals: new NormalCache(x) };
  return makeTarget(x.pos || x);
}

// =====================================================================================================================
// 4. Cloud container and PLY I/O. cloud = { n, rec: Float32Array(n * 17), pos: Float32Array(n * 3), hasScale }
// =====================================================================================================================
const TYPES = { char: [1, "getInt8"], int8: [1, "getInt8"], uchar: [1, "getUint8"], uint8: [1, "getUint8"], short: [2, "getInt16"], int16: [2, "getInt16"],
  ushort: [2, "getUint16"], uint16: [2, "getUint16"], int: [4, "getInt32"], int32: [4, "getInt32"], uint: [4, "getUint32"], uint32: [4, "getUint32"],
  float: [4, "getFloat32"], float32: [4, "getFloat32"], double: [8, "getFloat64"], float64: [8, "getFloat64"] };

export function readPLY(buf, opts = {}) {
  const u8 = new Uint8Array(buf), head = new TextDecoder("latin1").decode(u8.subarray(0, Math.min(u8.length, 65536)));
  const m = /end_header\r?\n/.exec(head); if (!m || !head.startsWith("ply")) throw new Error("Not a PLY file");
  const hdrEnd = m.index + m[0].length, lines = head.slice(0, m.index).split(/\r?\n/);
  const fmt = (lines.find((l) => l.startsWith("format")) || "").split(/\s+/)[1];
  if (fmt !== "binary_little_endian") throw new Error("Only binary little-endian PLY is supported (got " + fmt + ")");
  let n = 0, inVertex = false; const props = [];
  for (const l of lines) {
    const w = l.trim().split(/\s+/);
    if (w[0] === "element") { inVertex = w[1] === "vertex"; if (inVertex) n = parseInt(w[2], 10); else if (props.length) break; }
    else if (w[0] === "property" && inVertex) {
      if (w[1] === "list") throw new Error("List properties in the vertex element are not supported");
      const ty = TYPES[w[1]]; if (!ty) throw new Error("Unknown PLY type " + w[1]);
      props.push({ name: w[2], type: w[1], size: ty[0], get: ty[1] });
    }
  }
  if (!n || !props.length) throw new Error("PLY has no vertices");
  let off = 0; for (const p of props) { p.off = off; off += p.size; }
  const stride = off, by = Object.fromEntries(props.map((p) => [p.name, p]));
  if (!by.x || !by.y || !by.z) throw new Error("PLY has no x/y/z");
  if (u8.length - hdrEnd < n * stride) throw new Error("PLY body is shorter than its header says");
  const dv = new DataView(buf, hdrEnd, n * stride);
  const get = (p, i) => dv[p.get](i * stride + p.off, true);
  const colU8 = (a, b) => by[a] || by[b];
  const cr = colU8("red", "r"), cg = colU8("green", "g"), cb = colU8("blue", "b"), hasDC = !!by.f_dc_0, hasScale = !!by.scale_0;
  const rec = new Float32Array(n * P17), pos = new Float32Array(n * 3), logS = opts.defaultLogScale ?? Math.log(0.01);
  let k = 0;
  for (let i = 0; i < n; i++) {
    const x = get(by.x, i), y = get(by.y, i), z = get(by.z, i);
    if (!Number.isFinite(x + y + z)) continue;
    const o = k * P17;
    rec[o] = x; rec[o + 1] = y; rec[o + 2] = z;
    if (by.nx) { rec[o + 3] = get(by.nx, i); rec[o + 4] = get(by.ny, i); rec[o + 5] = get(by.nz, i); }
    for (let c = 0; c < 3; c++) {
      if (hasDC) rec[o + 6 + c] = get(by["f_dc_" + c], i);
      else if (cr && cg && cb) { const p = [cr, cg, cb][c], v = get(p, i); rec[o + 6 + c] = ((p.size === 1 ? v / 255 : v > 1.5 ? v / 255 : v) - 0.5) / SH_C0; }
    }
    rec[o + 9] = by.opacity ? get(by.opacity, i) : LOGIT09;
    for (let c = 0; c < 3; c++) rec[o + 10 + c] = hasScale ? get(by["scale_" + c], i) : logS;
    if (by.rot_0) for (let c = 0; c < 4; c++) rec[o + 13 + c] = get(by["rot_" + c], i); else rec[o + 13] = 1;
    pos[k * 3] = x; pos[k * 3 + 1] = y; pos[k * 3 + 2] = z; k++;
  }
  return { n: k, rec: k === n ? rec : rec.slice(0, k * P17), pos: k === n ? pos : pos.slice(0, k * 3), hasScale };
}
export function writePLY(cloud) {
  const head = new TextEncoder().encode("ply\nformat binary_little_endian 1.0\nelement vertex " + cloud.n + "\n" + NAMES17.map((p) => "property float " + p + "\n").join("") + "end_header\n");
  const body = new Uint8Array(cloud.rec.buffer, cloud.rec.byteOffset, cloud.n * P17 * 4), out = new Uint8Array(head.length + body.length);
  out.set(head, 0); out.set(body, head.length); return out.buffer;
}
export function cloudFromRecords(rec) { const n = (rec.length / P17) | 0, pos = new Float32Array(n * 3); for (let i = 0; i < n; i++) { pos[i * 3] = rec[i * P17]; pos[i * 3 + 1] = rec[i * P17 + 1]; pos[i * 3 + 2] = rec[i * P17 + 2]; } return { n, rec, pos, hasScale: true }; }
// splat.js parsePLY() output ({ data, count }, 13 floats per splat) -> positions only
export function positionsFromSplats(master) { const { data, count } = master, p = new Float32Array(count * 3); for (let i = 0; i < count; i++) { p[i * 3] = data[i * 13]; p[i * 3 + 1] = data[i * 13 + 1]; p[i * 3 + 2] = data[i * 13 + 2]; } return p; }

export function colorsOf(cloud) {                      // Uint8Array rgb 0..255
  const c = new Uint8Array(cloud.n * 3), r = cloud.rec;
  for (let i = 0; i < cloud.n; i++) for (let k = 0; k < 3; k++) c[i * 3 + k] = Math.max(0, Math.min(255, Math.round((0.5 + SH_C0 * r[i * P17 + 6 + k]) * 255)));
  return c;
}
export function setColors(cloud, rgb) { for (let i = 0; i < cloud.n; i++) for (let k = 0; k < 3; k++) cloud.rec[i * P17 + 6 + k] = (rgb[i * 3 + k] / 255 - 0.5) / SH_C0; }
export function subsetCloud(cloud, indices) {
  const n = indices.length, rec = new Float32Array(n * P17), pos = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) { const s = indices[i]; rec.set(cloud.rec.subarray(s * P17, (s + 1) * P17), i * P17); pos.set(cloud.pos.subarray(s * 3, s * 3 + 3), i * 3); }
  return { n, rec, pos, hasScale: cloud.hasScale };
}
export function concatClouds(a, b) {
  const rec = new Float32Array((a.n + b.n) * P17), pos = new Float32Array((a.n + b.n) * 3);
  rec.set(a.rec, 0); rec.set(b.rec, a.n * P17); pos.set(a.pos, 0); pos.set(b.pos, a.n * 3);
  return { n: a.n + b.n, rec, pos, hasScale: a.hasScale && b.hasScale };
}
// Rigid / similarity transform of a whole cloud: positions, normals, splat orientation and (for scale) splat size.
export function transformCloud(cloud, T) {
  const { n, rec } = cloud, R = T.R, t = T.t, s = T.s || 1, out = new Float32Array(rec.length), pos = new Float32Array(n * 3);
  const qR = quatFromR(R), ls = Math.log(s);
  for (let i = 0; i < n; i++) {
    const o = i * P17; out.set(rec.subarray(o, o + P17), o);
    const x = rec[o], y = rec[o + 1], z = rec[o + 2];
    const X = s * (R[0] * x + R[1] * y + R[2] * z) + t[0], Y = s * (R[3] * x + R[4] * y + R[5] * z) + t[1], Z = s * (R[6] * x + R[7] * y + R[8] * z) + t[2];
    out[o] = X; out[o + 1] = Y; out[o + 2] = Z; pos[i * 3] = X; pos[i * 3 + 1] = Y; pos[i * 3 + 2] = Z;
    const nx = rec[o + 3], ny = rec[o + 4], nz = rec[o + 5];
    out[o + 3] = R[0] * nx + R[1] * ny + R[2] * nz; out[o + 4] = R[3] * nx + R[4] * ny + R[5] * nz; out[o + 5] = R[6] * nx + R[7] * ny + R[8] * nz;
    if (s !== 1) { out[o + 10] += ls; out[o + 11] += ls; out[o + 12] += ls; }
    const w2 = rec[o + 13], x2 = rec[o + 14], y2 = rec[o + 15], z2 = rec[o + 16], [w1, x1, y1, z1] = qR;     // q' = qR (x) q
    const w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, qx = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, qy = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, qz = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2;
    const l = Math.hypot(w, qx, qy, qz) || 1; out[o + 13] = w / l; out[o + 14] = qx / l; out[o + 15] = qy / l; out[o + 16] = qz / l;
  }
  return { n, rec: out, pos, hasScale: cloud.hasScale };
}
// Plain xyz-only imports have no splat size: give them one from the point spacing so they render.
export function fillDefaultScale(cloud, tree) {
  if (cloud.hasScale) return; const ls = Math.log(Math.max(estimateSpacing(tree) * 0.7, 1e-5));
  for (let i = 0; i < cloud.n; i++) cloud.rec[i * P17 + 10] = cloud.rec[i * P17 + 11] = cloud.rec[i * P17 + 12] = ls;
  cloud.hasScale = true;
}

// One index per occupied voxel (the first point seen). Use it to thin a cloud evenly before ICP or for display.
export function voxelIndices(pos, voxel) {
  const n = (pos.length / 3) | 0; if (!n) return new Uint32Array(0);
  let x0 = Infinity, y0 = Infinity, z0 = Infinity, x1 = -Infinity, y1 = -Infinity, z1 = -Infinity;
  for (let i = 0; i < n; i++) { const x = pos[i * 3], y = pos[i * 3 + 1], z = pos[i * 3 + 2]; if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; if (z < z0) z0 = z; if (z > z1) z1 = z; }
  const inv = 1 / voxel, nx = Math.floor((x1 - x0) * inv) + 1, ny = Math.floor((y1 - y0) * inv) + 1, nz = Math.floor((z1 - z0) * inv) + 1;
  if (nx * ny * nz > 2 ** 50) throw new Error("voxel size too small for this cloud");
  const seen = new Set(), out = [];
  for (let i = 0; i < n; i++) {
    const k = (Math.floor((pos[i * 3] - x0) * inv) * ny + Math.floor((pos[i * 3 + 1] - y0) * inv)) * nz + Math.floor((pos[i * 3 + 2] - z0) * inv);
    if (!seen.has(k)) { seen.add(k); out.push(i); }
  }
  return Uint32Array.from(out);
}

// =====================================================================================================================
// 5. Registration: ICP (point-to-plane / point-to-point, trimmed, coarse-to-fine, optional scale)
// =====================================================================================================================
// target: makeTarget() result. src: Float32Array xyz already placed with the initial guess.
// Returns the CORRECTION T (apply to src): { ok, R, t, s, rmse, fitness, iterations, converged, condition, history } or { ok:false, reason }.
//   opts.method       "plane" (default, needs normals; best on surfaces) | "point"
//   opts.stages       [{ maxDist, iters }] capture radius (m) per stage, coarse to fine. Default from the point spacing.
//   opts.trim         fraction of the best correspondences used per iteration (default 0.85)
//   opts.maxPts       source points sampled per iteration (default 10000)
//   opts.estimateScale  also solve a uniform scale (for clouds with unknown relative scale, e.g. two independent photo depths)
//   opts.maxCurv      normals with higher curvature (lambda_min / trace) are ignored (default 0.12)
export function icp(target, src, opts = {}) {
  const o = { method: "plane", maxPts: 10000, trim: 0.85, minPairs: 50, maxCurv: 0.12, estimateScale: false, ...opts };
  const { tree, normals } = target, n = (src.length / 3) | 0;
  if (n < o.minPairs) return { ok: false, reason: "too few points for ICP" };
  const sp = estimateSpacing(tree), stages = o.stages || [{ maxDist: sp * 24, iters: 15 }, { maxDist: sp * 8, iters: 15 }, { maxDist: sp * 3, iters: 15 }];
  const step = Math.max(1, Math.floor(n / o.maxPts)), m = Math.floor(n / step), cur = new Float64Array(m * 3);
  for (let i = 0; i < m; i++) for (let c = 0; c < 3; c++) cur[i * 3 + c] = src[i * step * 3 + c];
  const pi = new Int32Array(m), pj = new Int32Array(m), pd = new Float64Array(m);
  let T = identity(), rmse = Infinity, fitness = 0, iterations = 0, converged = false, condition = 1; const history = [];

  const match = (md2) => {                             // nearest target point for every sampled source point
    let c = 0;
    for (let i = 0; i < m; i++) {
      const j = tree.nearest(cur[i * 3], cur[i * 3 + 1], cur[i * 3 + 2], md2);
      if (j >= 0) { pi[c] = i; pj[c] = j; pd[c] = tree._bd; c++; }
    }
    return c;
  };
  const trimmed = (c) => {                             // indices into the pair list kept after trimming
    const thr = Float64Array.from(pd.subarray(0, c)).sort()[Math.max(0, Math.min(c - 1, Math.floor(o.trim * c) - 1))], keep = [];
    for (let k = 0; k < c; k++) if (pd[k] <= thr) keep.push(k);
    return keep;
  };

  for (const st of stages) {
    const md2 = st.maxDist * st.maxDist; converged = false;
    for (let it = 0; it < st.iters; it++) {
      const c = match(md2);
      if (c < o.minPairs) return { ok: false, reason: "no overlap found: too few points have a neighbour within " + (st.maxDist * 1000).toFixed(0) + " mm" };
      const keep = trimmed(c); let sum = 0; for (const k of keep) sum += pd[k];
      rmse = Math.sqrt(sum / keep.length); history.push(rmse); iterations++;
      let step6 = null;                                // { R, t, s } incremental transform
      if (o.method === "plane") {
        const good = keep.filter((k) => normals.ok(pj[k]) && normals.curv[pj[k]] <= o.maxCurv);
        if (good.length >= Math.max(o.minPairs, 0.2 * keep.length)) step6 = planeStep(cur, pi, pj, good, target, normals, o.estimateScale);
      }
      if (!step6) {                                    // point-to-point (also the fallback when normals are unusable)
        const P = new Float64Array(keep.length * 3), Q = new Float64Array(keep.length * 3);
        keep.forEach((k, r) => { for (let q = 0; q < 3; q++) { P[r * 3 + q] = cur[pi[k] * 3 + q]; Q[r * 3 + q] = target.pos[pj[k] * 3 + q]; } });
        step6 = horn(P, Q, keep.length, o.estimateScale);
      } else condition = Math.min(condition, step6.cond);
      for (let i = 0; i < m; i++) {
        const x = cur[i * 3], y = cur[i * 3 + 1], z = cur[i * 3 + 2], R = step6.R, s = step6.s || 1;
        cur[i * 3] = s * (R[0] * x + R[1] * y + R[2] * z) + step6.t[0]; cur[i * 3 + 1] = s * (R[3] * x + R[4] * y + R[5] * z) + step6.t[1]; cur[i * 3 + 2] = s * (R[6] * x + R[7] * y + R[8] * z) + step6.t[2];
      }
      T = compose(step6, T);
      if (rotationAngle(step6.R) < 1e-5 && Math.hypot(...step6.t) < st.maxDist * 2e-4 && Math.abs((step6.s || 1) - 1) < 1e-6) { converged = true; break; }
    }
  }
  const last = stages[stages.length - 1], c = match(last.maxDist * last.maxDist); let sum = 0;
  for (let k = 0; k < c; k++) sum += pd[k];
  fitness = c / m; rmse = c ? Math.sqrt(sum / c) : Infinity;
  return { ok: true, R: T.R, t: T.t, s: T.s, rmse, fitness, iterations, converged, condition, history };
}
// One linearised point-to-plane step. Solves for a small rotation w, translation tc and (optionally) a scale change ds about the
// centroid c of the used points:  p' = p + w x (p - c) + ds (p - c) + tc,  residual n . (p' - q).
function planeStep(cur, pi, pj, good, target, normals, withScale) {
  let cx = 0, cy = 0, cz = 0; const k = good.length, U = withScale ? 7 : 6;
  for (const g of good) { cx += cur[pi[g] * 3]; cy += cur[pi[g] * 3 + 1]; cz += cur[pi[g] * 3 + 2]; }
  cx /= k; cy /= k; cz /= k;
  const A = new Float64Array(U * U), b = new Float64Array(U), a = new Float64Array(U), N = normals.nrm, Q = target.pos;
  for (const g of good) {
    const i = pi[g] * 3, j = pj[g] * 3, px = cur[i], py = cur[i + 1], pz = cur[i + 2], nx = N[j], ny = N[j + 1], nz = N[j + 2];
    const dx = px - cx, dy = py - cy, dz = pz - cz, r = nx * (px - Q[j]) + ny * (py - Q[j + 1]) + nz * (pz - Q[j + 2]);
    a[0] = dy * nz - dz * ny; a[1] = dz * nx - dx * nz; a[2] = dx * ny - dy * nx; a[3] = nx; a[4] = ny; a[5] = nz;
    if (withScale) a[6] = nx * dx + ny * dy + nz * dz;
    for (let u = 0; u < U; u++) { b[u] -= a[u] * r; for (let v = 0; v <= u; v++) A[u * U + v] += a[u] * a[v]; }
  }
  for (let u = 0; u < U; u++) for (let v = u + 1; v < U; v++) A[u * U + v] = A[v * U + u];
  let tr = 0; for (let u = 0; u < U; u++) tr += A[u * U + u];
  for (let u = 0; u < U; u++) A[u * U + u] += 1e-9 * tr / U + 1e-12;            // tiny damping: planar scenes leave directions undetermined
  const sol = solveSPD(A, b, U);
  if (!sol) return null;
  let wx = sol.x[0], wy = sol.x[1], wz = sol.x[2], tx = sol.x[3], ty = sol.x[4], tz = sol.x[5], ds = withScale ? sol.x[6] : 0;
  const ang = Math.hypot(wx, wy, wz);
  if (ang > 0.3) { const f = 0.3 / ang; wx *= f; wy *= f; wz *= f; tx *= f; ty *= f; tz *= f; }   // limit one step
  ds = Math.max(-0.1, Math.min(0.1, ds));
  const R = rotAxis(wx, wy, wz), sc = 1 + ds, Rc = mulRV(R, [cx, cy, cz]);
  return { R, t: [cx + tx - sc * Rc[0], cy + ty - sc * Rc[1], cz + tz - sc * Rc[2]], s: sc, cond: sol.cond };
}

// Coarse start for clouds with no pose guess: align centroids, then try the 24 axis-flips of the principal axes and keep the
// one a short ICP likes best. Works when the clouds overlap a lot and are not near-symmetric; otherwise supply opts.init.
// Returns { R, t, s } mapping src onto the target.
const PERMS24 = (() => {
  const out = [], P = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]], par = [1, -1, -1, 1, 1, -1];
  P.forEach((p, pi) => { for (let sg = 0; sg < 8; sg++) { const s = [sg & 1 ? -1 : 1, sg & 2 ? -1 : 1, sg & 4 ? -1 : 1], M = new Array(9).fill(0); for (let r = 0; r < 3; r++) M[r * 3 + p[r]] = s[r]; if (par[pi] * s[0] * s[1] * s[2] > 0) out.push(M); } });
  return out;
})();
function principalFrame(pos, sample) {
  const n = (pos.length / 3) | 0, step = Math.max(1, Math.floor(n / sample)); let c = [0, 0, 0], k = 0;
  for (let i = 0; i < n; i += step) { c[0] += pos[i * 3]; c[1] += pos[i * 3 + 1]; c[2] += pos[i * 3 + 2]; k++; }
  c = c.map((v) => v / k);
  const A = new Float64Array(9), V = new Float64Array(9);
  for (let i = 0; i < n; i += step) { const d = [pos[i * 3] - c[0], pos[i * 3 + 1] - c[1], pos[i * 3 + 2] - c[2]]; for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) A[a * 3 + b] += d[a] * d[b] / k; }
  jacobi3(A, V);
  const ord = [0, 1, 2].sort((p, q) => A[q * 4] - A[p * 4]);                    // descending variance
  const E = new Array(9); for (let r = 0; r < 3; r++) for (let q = 0; q < 3; q++) E[r * 3 + q] = V[r * 3 + ord[q]];   // columns = axes
  const det = E[0] * (E[4] * E[8] - E[5] * E[7]) - E[1] * (E[3] * E[8] - E[5] * E[6]) + E[2] * (E[3] * E[7] - E[4] * E[6]);
  if (det < 0) for (let r = 0; r < 3; r++) E[r * 3 + 2] *= -1;
  return { c, E, var: ord.map((q) => A[q * 4]) };
}
// Search: 24 axis assignments x 7 translation seeds, each scored by a short ICP between thinned copies (fast), then the best three
// are refined on the full target and the one with the most matched points wins.
export function coarseAlign(target, srcPos, opts = {}) {
  const fT = principalFrame(target.pos, 4000), fS = principalFrame(srcPos, 4000);
  const s0 = opts.estimateScale ? Math.sqrt((fT.var[0] + fT.var[1] + fT.var[2]) / (fS.var[0] + fS.var[1] + fS.var[2])) : 1;
  const sd = Math.sqrt(fT.var[0]), vox = Math.max(4 * sd / 28, 1e-4);
  const ti = voxelIndices(target.pos, vox), tp = new Float32Array(ti.length * 3); ti.forEach((j, i) => tp.set(target.pos.subarray(j * 3, j * 3 + 3), i * 3));
  const thin = { pos: tp, tree: new KDTree(tp), normals: null };
  const si = voxelIndices(srcPos, vox / Math.max(s0, 1e-3)), stp = Math.max(1, Math.floor(si.length / 500)), sub = new Float32Array(Math.floor(si.length / stp) * 3);
  for (let i = 0; i < sub.length / 3; i++) sub.set(srcPos.subarray(si[i * stp] * 3, si[i * stp] * 3 + 3), i * 3);
  const seeds = [[0, 0, 0]]; for (let a = 0; a < 3; a++) for (const sg of [-1, 1]) { const d = [0, 0, 0]; for (let k = 0; k < 3; k++) d[k] = sg * 0.5 * Math.sqrt(fT.var[a]) * fT.E[k * 3 + a]; seeds.push(d); }
  const cands = [];
  for (const P of PERMS24) {
    const Rc = mulRR(fT.E, mulRR(P, transpose(fS.E))), v = mulRV(Rc, fS.c);
    for (const d of seeds) {
      const T0 = { R: Rc, s: s0, t: [fT.c[0] - s0 * v[0] + d[0], fT.c[1] - s0 * v[1] + d[1], fT.c[2] - s0 * v[2] + d[2]] };
      const r = icp(thin, applyToPositions(T0, sub), { method: "point", maxPts: 500, trim: 0.7, minPairs: 30, stages: [{ maxDist: vox * 8, iters: 5 }, { maxDist: vox * 3, iters: 5 }, { maxDist: vox * 1.2, iters: 3 }] });
      if (r.ok) cands.push({ T: compose(r, T0), fit: r.fitness, rmse: r.rmse });
    }
  }
  cands.sort((p, q) => q.fit - p.fit || p.rmse - q.rmse);
  let best = null; const sp = estimateSpacing(target.tree), stepF = Math.max(1, Math.floor(srcPos.length / 3 / 4000)), subF = new Float32Array(Math.floor(srcPos.length / 3 / stepF) * 3);
  for (let i = 0; i < subF.length / 3; i++) subF.set(srcPos.subarray(i * stepF * 3, i * stepF * 3 + 3), i * 3);
  for (const c of cands.slice(0, 3)) {
    const r = icp(target, applyToPositions(c.T, subF), { method: "point", maxPts: 3000, trim: 0.7, minPairs: 50, stages: [{ maxDist: vox * 2, iters: 8 }, { maxDist: Math.max(sp * 6, vox * 0.5), iters: 8 }, { maxDist: Math.max(sp * 3, vox * 0.2), iters: 6 }] });
    if (r.ok && (!best || r.fitness > best.fitness)) best = { ...compose(r, c.T), fitness: r.fitness, rmse: r.rmse };
  }
  return best ? { ok: true, ...best } : { ok: false, reason: "no starting alignment found" };
}

// Joint pose + depth-scale registration for a photo-derived view (the "Add view" case).
// The photo's depth is relative: z = 1 / (a * dn + b). Fitting (a, b) to the master at a wrong pose and then running a rigid ICP lets the
// two drift apart (a scale error and a shift along the viewing axis look alike), so here the pose AND (a, b) are solved together against
// the master surface (point-to-plane, 6 + 2 unknowns), which pins both down.
//   master: parsePLY() result or a makeTarget(); cam: Float32Array xyz of the view in its own camera frame (from newPointsFromDepth,
//   built with ab0); ab0: { a, b } used to build them; pose0: { R, t } camera-to-master guess.
// Returns { ok, pose, a, b, rmse, frac, iterations } or { ok: false, reason }.
export function icpDepth(master, cam, ab0, pose0, opts = {}) {
  const target = master.tree ? master : targetFromSplats(master), { tree, normals } = target;
  const o = { maxPts: 8000, trim: 0.8, minPairs: 60, maxCurv: 0.12, damp: 1e-2, ...opts }, n = (cam.length / 3) | 0;
  if (n < o.minPairs) return { ok: false, reason: "too few points for ICP" };
  const stages = o.stages ? o.stages.map((s) => ({ maxDist: s.maxDist ?? s.cell, iters: s.iters })) : [{ maxDist: 0.12, iters: 10 }, { maxDist: 0.04, iters: 10 }, { maxDist: 0.015, iters: 10 }];
  const step = Math.max(1, Math.floor(n / o.maxPts)), m = Math.floor(n / step);
  const ray = new Float64Array(m * 2), dn = new Float64Array(m), cur = new Float64Array(m * 3), pi = new Int32Array(m), pj = new Int32Array(m), pd = new Float64Array(m);
  for (let i = 0; i < m; i++) { const z = Math.max(-cam[i * step * 3 + 2], 1e-3); ray[i * 2] = cam[i * step * 3] / z; ray[i * 2 + 1] = cam[i * step * 3 + 1] / z; dn[i] = (1 / z - ab0.b) / ab0.a; }
  let T = { R: pose0.R.slice(), t: pose0.t.slice() }, a = ab0.a, b = ab0.b, iterations = 0;
  const place = () => {
    for (let i = 0; i < m; i++) {
      const z = 1 / Math.max(a * dn[i] + b, 1 / 500), x = ray[i * 2] * z, y = ray[i * 2 + 1] * z, w = -z, R = T.R;
      cur[i * 3] = R[0] * x + R[1] * y + R[2] * w + T.t[0]; cur[i * 3 + 1] = R[3] * x + R[4] * y + R[5] * w + T.t[1]; cur[i * 3 + 2] = R[6] * x + R[7] * y + R[8] * w + T.t[2];
    }
  };
  const match = (md2) => { let c = 0; for (let i = 0; i < m; i++) { const j = tree.nearest(cur[i * 3], cur[i * 3 + 1], cur[i * 3 + 2], md2); if (j >= 0) { pi[c] = i; pj[c] = j; pd[c] = tree._bd; c++; } } return c; };
  for (const st of stages) {
    for (let it = 0; it < st.iters; it++) {
      place(); const c = match(st.maxDist * st.maxDist);
      if (c < o.minPairs) return { ok: false, reason: "no overlap found between the photo and the master" };
      const thr = Float64Array.from(pd.subarray(0, c)).sort()[Math.max(0, Math.floor(o.trim * c) - 1)], good = [];
      for (let k = 0; k < c; k++) if (pd[k] <= thr && normals.ok(pj[k]) && normals.curv[pj[k]] <= o.maxCurv) good.push(k);
      if (good.length < o.minPairs) return { ok: false, reason: "too few reliable surface normals in the overlap" };
      let cx = 0, cy = 0, cz = 0; for (const g of good) { cx += cur[pi[g] * 3]; cy += cur[pi[g] * 3 + 1]; cz += cur[pi[g] * 3 + 2]; }
      cx /= good.length; cy /= good.length; cz /= good.length;
      const U = 8, A = new Float64Array(U * U), rhs = new Float64Array(U), row = new Float64Array(U), N = normals.nrm, Q = target.pos, R = T.R;
      for (const g of good) {
        const i = pi[g], j = pj[g] * 3, px = cur[i * 3], py = cur[i * 3 + 1], pz = cur[i * 3 + 2], nx = N[j], ny = N[j + 1], nz = N[j + 2];
        const dx = px - cx, dy = py - cy, dz = pz - cz, r = nx * (px - Q[j]) + ny * (py - Q[j + 1]) + nz * (pz - Q[j + 2]), zz = 1 / Math.max(a * dn[i] + b, 1 / 500);
        const rx = ray[i * 2], ry = ray[i * 2 + 1], dza = -zz * zz * dn[i], dzb = -zz * zz;          // dz/da, dz/db; P = z * (rx, ry, -1)
        const ux = R[0] * rx + R[1] * ry - R[2], uy = R[3] * rx + R[4] * ry - R[5], uz = R[6] * rx + R[7] * ry - R[8], nu = nx * ux + ny * uy + nz * uz;   // n . R (rx, ry, -1)
        row[0] = dy * nz - dz * ny; row[1] = dz * nx - dx * nz; row[2] = dx * ny - dy * nx; row[3] = nx; row[4] = ny; row[5] = nz; row[6] = nu * dza; row[7] = nu * dzb;
        for (let p = 0; p < U; p++) { rhs[p] -= row[p] * r; for (let q = 0; q <= p; q++) A[p * U + q] += row[p] * row[q]; }
      }
      for (let p = 0; p < U; p++) for (let q = p + 1; q < U; q++) A[p * U + q] = A[q * U + p];
      let tr = 0; for (let p = 0; p < U; p++) tr += A[p * U + p];
      for (let p = 0; p < U; p++) A[p * U + p] += (p >= 6 ? o.damp : 0) * A[p * U + p] + 1e-9 * tr / U + 1e-12;     // damp the two depth-fit unknowns
      const sol = solveSPD(A, rhs, U); if (!sol) return { ok: false, reason: "degenerate geometry (flat or featureless overlap)" };
      let [wx, wy, wz, tx, ty, tz, da, db] = sol.x; const ang = Math.hypot(wx, wy, wz);
      if (ang > 0.2) { const f = 0.2 / ang; wx *= f; wy *= f; wz *= f; tx *= f; ty *= f; tz *= f; }
      da = Math.max(-0.05 * Math.abs(a), Math.min(0.05 * Math.abs(a), da)); db = Math.max(-0.05 * Math.abs(b) - 0.01, Math.min(0.05 * Math.abs(b) + 0.01, db));
      const Rs = rotAxis(wx, wy, wz), Rc = mulRV(Rs, [cx, cy, cz]);
      T = compose({ R: Rs, t: [cx + tx - Rc[0], cy + ty - Rc[1], cz + tz - Rc[2]], s: 1 }, T); T = { R: T.R, t: T.t }; a += da; b += db; iterations++;
      if (rotationAngle(Rs) < 1e-5 && Math.hypot(tx, ty, tz) < st.maxDist * 2e-4 && Math.abs(da) < 1e-5 && Math.abs(db) < 1e-5) break;
    }
  }
  const last = stages[stages.length - 1]; place(); const c = match(last.maxDist * last.maxDist); let sum = 0; for (let k = 0; k < c; k++) sum += pd[k];
  return { ok: true, pose: T, a, b, rmse: c ? Math.sqrt(sum / c) : Infinity, frac: c / m, iterations };
}

// Drop-in for register.js icp(master, src, stages): master = parsePLY() result, src = Float32Array xyz in the master frame.
// Same return shape ({ ok, R, t, rmse, frac }) so registerView() can use it unchanged.
const _splatTargets = new WeakMap();
export function targetFromSplats(master) {
  let t = _splatTargets.get(master); if (!t) { t = makeTarget(positionsFromSplats(master)); _splatTargets.set(master, t); } return t;
}
export function icpPlane(master, src, stages, maxPts = 10000, trim = 0.8) {
  const r = icp(targetFromSplats(master), src, { method: "plane", maxPts, trim, stages: stages && stages.map((s) => ({ maxDist: s.cell, iters: s.iters })) });
  return r.ok ? { ok: true, R: r.R, t: r.t, rmse: r.rmse, frac: r.fitness, converged: r.converged } : { ok: false, reason: r.reason };
}

// =====================================================================================================================
// 6. Comparison: cloud-to-cloud distance, statistics, heat colours
// =====================================================================================================================
// Distance from every point of srcPos to its nearest point in the target. dist = Infinity when nothing is within maxDist.
export function c2c(target, srcPos, opts = {}) {
  const tree = asTarget(target).tree, n = (srcPos.length / 3) | 0, md2 = (opts.maxDist ?? Infinity) ** 2;
  const dist = new Float32Array(n), nn = new Int32Array(n);
  for (let i = 0; i < n; i++) { const j = tree.nearest(srcPos[i * 3], srcPos[i * 3 + 1], srcPos[i * 3 + 2], md2); nn[i] = j; dist[i] = j < 0 ? Infinity : Math.sqrt(tree._bd); }
  return { dist, nn };
}
// Signed distance along the target surface normal: positive = the point lies in front of the target surface (normal side),
// negative = behind it. Normals have arbitrary sign, so pass orient = a viewpoint [x, y, z] to make "front" mean "towards it".
// Unlike the nearest-neighbour distance this does not grow with point spacing: it is the distance to the local surface.
// Where the normal is unreliable (edges, thin structure) the plain nearest-neighbour distance is returned instead.
export function signedC2C(target, srcPos, opts = {}) {
  target = asTarget(target);
  const { dist, nn } = c2c(target, srcPos, opts), n = nn.length, out = new Float32Array(n), N = target.normals, Q = target.pos, ov = opts.orient, maxCurv = opts.maxCurv ?? 0.12;
  for (let i = 0; i < n; i++) {
    const j = nn[i]; if (j < 0) { out[i] = NaN; continue; }
    if (!N.ok(j) || N.curv[j] > maxCurv) { out[i] = dist[i]; continue; }
    let nx = N.nrm[j * 3], ny = N.nrm[j * 3 + 1], nz = N.nrm[j * 3 + 2];
    if (ov && nx * (ov[0] - Q[j * 3]) + ny * (ov[1] - Q[j * 3 + 1]) + nz * (ov[2] - Q[j * 3 + 2]) < 0) { nx = -nx; ny = -ny; nz = -nz; }
    out[i] = nx * (srcPos[i * 3] - Q[j * 3]) + ny * (srcPos[i * 3 + 1] - Q[j * 3 + 1]) + nz * (srcPos[i * 3 + 2] - Q[j * 3 + 2]);
  }
  for (let i = 0; i < n; i++) if (Number.isNaN(out[i])) out[i] = Infinity;      // no neighbour in range
  return out;
}
// Statistics over the finite distances. thresholds (m): fraction of overlapping points within each one.
export function distanceStats(dist, { bins = 50, thresholds = [], cap = null } = {}) {
  const v = []; for (let i = 0; i < dist.length; i++) if (Number.isFinite(dist[i])) v.push(Math.abs(dist[i]));
  const n = v.length, out = { total: dist.length, n, overlap: dist.length ? n / dist.length : 0 };
  if (!n) return { ...out, min: NaN, max: NaN, mean: NaN, std: NaN, rms: NaN, median: NaN, p90: NaN, p95: NaN, p99: NaN, hist: { counts: [], edges: [] }, within: {} };
  const s = Float64Array.from(v).sort(), q = (p) => s[Math.min(n - 1, Math.floor(p * n))];
  let sum = 0, sq = 0; for (let i = 0; i < n; i++) { sum += s[i]; sq += s[i] * s[i]; }
  const mean = sum / n, hi = cap ?? Math.max(q(0.99), 1e-9), counts = new Array(bins).fill(0), edges = Array.from({ length: bins + 1 }, (_, i) => (hi * i) / bins);
  for (let i = 0; i < n; i++) counts[Math.min(bins - 1, Math.floor((s[i] / hi) * bins))]++;
  const within = {}; for (const t of thresholds) { let c = 0; while (c < n && s[c] <= t) c++; within[t] = c / n; }
  return { ...out, min: s[0], max: s[n - 1], mean, std: Math.sqrt(Math.max(sq / n - mean * mean, 0)), rms: Math.sqrt(sq / n), median: q(0.5), p90: q(0.9), p95: q(0.95), p99: q(0.99), hist: { counts, edges }, within };
}
// Symmetric comparison of two clouds (A -> B and B -> A), like CloudCompare's cloud/cloud distance both ways.
export function compareSymmetric(a, b, opts = {}) {
  const ta = asTarget(a), tb = asTarget(b), ab = c2c(tb, a.pos, opts), ba = c2c(ta, b.pos, opts);
  const sa = distanceStats(ab.dist, opts), sb = distanceStats(ba.dist, opts);
  return { ab: sa, ba: sb, chamfer: sa.mean + sb.mean, hausdorff: Math.max(sa.max, sb.max), distAB: ab.dist, distBA: ba.dist };
}
// Heat ramp, t in 0..1: blue -> cyan -> green -> yellow -> red. Bright at both ends, so it reads on the dark viewer background.
const HEAT_STOPS = [[0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0]];
export function heat(t) {
  const u = Math.min(1, Math.max(0, t)) * 4, k = Math.min(3, Math.floor(u)), q = u - k, a = HEAT_STOPS[k], b = HEAT_STOPS[k + 1];
  return [Math.round(a[0] + (b[0] - a[0]) * q), Math.round(a[1] + (b[1] - a[1]) * q), Math.round(a[2] + (b[2] - a[2]) * q)];
}
export const HEAT_CSS = "linear-gradient(90deg,rgb(0,0,255),rgb(0,255,255),rgb(0,255,0),rgb(255,255,0),rgb(255,0,0))";
// Heat colours for a distance array: 0 -> blue, vmax -> red. Points with no neighbour (Infinity / NaN) are grey.
export function heatColors(dist, vmax, opts = {}) {
  const out = new Uint8Array(dist.length * 3), grey = opts.missing || [150, 150, 150];
  for (let i = 0; i < dist.length; i++) {
    const d = dist[i], c = Number.isFinite(d) ? heat(Math.abs(d) / vmax) : grey;
    out[i * 3] = c[0]; out[i * 3 + 1] = c[1]; out[i * 3 + 2] = c[2];
  }
  return out;
}
// Master (dimmed, thinned) + the new cloud coloured by distance, as one cloud ready for SplatViewer.load(writePLY(...)).
export function overlayCloud(master, added, rgb, { maxMaster = 150000, dim = 0.55 } = {}) {
  const step = Math.max(1, Math.floor(master.n / maxMaster)), idx = []; for (let i = 0; i < master.n; i += step) idx.push(i);
  const mm = subsetCloud(master, idx), c = colorsOf(mm);
  for (let i = 0; i < mm.n; i++) { const g = 0.3 * c[i * 3] + 0.59 * c[i * 3 + 1] + 0.11 * c[i * 3 + 2]; for (let k = 0; k < 3; k++) c[i * 3 + k] = Math.round((0.35 * g + 0.65 * c[i * 3 + k]) * dim); }
  setColors(mm, c);
  const aa = { n: added.n, rec: Float32Array.from(added.rec), pos: added.pos, hasScale: added.hasScale }; setColors(aa, rgb);
  return concatClouds(mm, aa);
}
// For the capture-review screen: new view (camera frame, as built by buildPLY) placed with `pose` and compared with the master.
// Returns { stats, overlay (PLY ArrayBuffer), vmax }. Throws on unreadable input; the caller should catch and skip the overlay.
export function compareForReview(masterBuf, newBuf, pose, opts = {}) {
  const tol = opts.tol ?? 0.02, master = readPLY(masterBuf), view = transformCloud(readPLY(newBuf), pose);
  const target = opts.target || makeTarget(master.pos), md = opts.maxDist ?? tol * 10;
  const dist = opts.metric === "nn" ? c2c(target, view.pos, { maxDist: md }).dist : signedC2C(target, view.pos, { maxDist: md });   // plane distance: not inflated by point spacing
  const stats = distanceStats(dist, { thresholds: [tol / 2, tol], bins: 40, cap: tol * 2 }), vmax = opts.vmax ?? tol;
  return { stats, vmax, tol, overlay: writePLY(overlayCloud(master, view, heatColors(dist, vmax), opts)) };
}

// =====================================================================================================================
// 7. Merging
// =====================================================================================================================
// Statistical outlier removal (same rule as the WebPointCloud server): drop points whose mean distance to their k nearest
// neighbours is more than stdRatio standard deviations above the average. Returns Uint8Array keep flags.
// opts.extra: a KDTree of points that already exist (e.g. the master). Neighbours are then looked up in both sets, so new points
// at the seam with existing geometry are not mistaken for outliers just because they have neighbours on one side only.
export function sorMask(pos, { k = 12, stdRatio = 2.5, extra = null } = {}) {
  const n = (pos.length / 3) | 0, keep = new Uint8Array(n).fill(1); if (n < k + 2) return keep;
  const tree = new KDTree(pos), oi = new Int32Array(k + 1), od = new Float64Array(k + 1), ei = new Int32Array(k), ed = new Float64Array(k), md = new Float64Array(n);
  let sum = 0, sq = 0;
  for (let i = 0; i < n; i++) {
    const x = pos[i * 3], y = pos[i * 3 + 1], z = pos[i * 3 + 2], c = tree.knn(x, y, z, k + 1, oi, od), ce = extra ? extra.knn(x, y, z, k, ei, ed) : 0;
    let a = 1, b = 0, s = 0, got = 0;                  // merge the two ascending lists (own list starts at 1: index 0 is the point itself)
    while (got < k && (a < c || b < ce)) { const da = a < c ? od[a] : Infinity, db = b < ce ? ed[b] : Infinity; if (da <= db) { s += Math.sqrt(da); a++; } else { s += Math.sqrt(db); b++; } got++; }
    md[i] = got ? s / got : 0; sum += md[i]; sq += md[i] * md[i];
  }
  const mean = sum / n, thr = mean + stdRatio * Math.sqrt(Math.max(sq / n - mean * mean, 0));
  for (let i = 0; i < n; i++) if (md[i] > thr) keep[i] = 0;
  return keep;
}
// Add `added` (already in the master frame) to `master`.
//   mode "novel" (default): only points the master does not already cover (no master point within tol). Keeps the master intact
//                            and adds just the new geometry. "all": append everything.
//   tol          same-surface distance (m). voxel: thin the added points to one per voxel (m, 0 = off).
//   matchColor   shift the added colours to the master's exposure, using the overlap (gain/offset per channel, clamped)
//   sor          { k, stdRatio } outlier filter on the points about to be added, or null
// Returns { cloud, added, covered, thinned, outliers, colorGain }.
export function mergeClouds(master, added, opts = {}) {
  const o = { mode: "novel", tol: 0.015, voxel: 0.01, matchColor: true, sor: { k: 12, stdRatio: 2.5 }, ...opts };
  const target = o.target || makeTarget(master.pos), { nn } = c2c(target, added.pos, { maxDist: o.tol });
  let colorGain = null, work = added;
  if (o.matchColor) {
    const mc = colorsOf(master), ac = colorsOf(added), pairs = []; for (let i = 0; i < added.n; i++) if (nn[i] >= 0) pairs.push(i);
    if (pairs.length >= 500) {
      const stepP = Math.max(1, Math.floor(pairs.length / 20000)); let gain = [1, 1, 1], off = [0, 0, 0];
      for (let c = 0; c < 3; c++) {
        let ma = 0, mm = 0, k = 0; for (let q = 0; q < pairs.length; q += stepP) { const i = pairs[q]; ma += ac[i * 3 + c]; mm += mc[nn[i] * 3 + c]; k++; }
        ma /= k; mm /= k; let va = 0, vm = 0; for (let q = 0; q < pairs.length; q += stepP) { const i = pairs[q]; va += (ac[i * 3 + c] - ma) ** 2; vm += (mc[nn[i] * 3 + c] - mm) ** 2; }
        gain[c] = Math.min(1.33, Math.max(0.75, Math.sqrt((vm + 1e-9) / (va + 1e-9)))); off[c] = mm - gain[c] * ma;
      }
      const out = new Uint8Array(ac.length); for (let i = 0; i < added.n; i++) for (let c = 0; c < 3; c++) out[i * 3 + c] = Math.max(0, Math.min(255, Math.round(gain[c] * ac[i * 3 + c] + off[c])));
      work = { n: added.n, rec: Float32Array.from(added.rec), pos: added.pos, hasScale: added.hasScale }; setColors(work, out); colorGain = { gain, offset: off, pairs: pairs.length };
    }
  }
  let pick = []; for (let i = 0; i < added.n; i++) if (o.mode === "all" || nn[i] < 0) pick.push(i);
  const covered = added.n - pick.length; let thinned = 0, outliers = 0;
  if (o.voxel > 0 && pick.length) {
    const sub = new Float32Array(pick.length * 3); pick.forEach((s, i) => { sub[i * 3] = added.pos[s * 3]; sub[i * 3 + 1] = added.pos[s * 3 + 1]; sub[i * 3 + 2] = added.pos[s * 3 + 2]; });
    const v = voxelIndices(sub, o.voxel); thinned = pick.length - v.length; pick = Array.from(v, (i) => pick[i]);
  }
  if (o.sor && pick.length > 50) {
    const sub = new Float32Array(pick.length * 3); pick.forEach((s, i) => { sub[i * 3] = added.pos[s * 3]; sub[i * 3 + 1] = added.pos[s * 3 + 1]; sub[i * 3 + 2] = added.pos[s * 3 + 2]; });
    const keep = sorMask(sub, { ...o.sor, extra: target.tree }), before = pick.length; pick = pick.filter((_, i) => keep[i]); outliers = before - pick.length;
  }
  const cloud = pick.length ? concatClouds(master, subsetCloud(work, pick)) : master;
  return { cloud, added: pick.length, covered, thinned, outliers, colorGain };
}

// =====================================================================================================================
// 8. Whole pipeline: register a second cloud to the master and merge it
// =====================================================================================================================
export const REGISTER_DEFAULTS = {
  init: "pca",           // "none" | "centroid" | "pca" | a { R, t } transform guess. Camera-based pose guesses go here.
  method: "plane", estimateScale: false, voxel: 0, tol: 0.015, minFitness: 0.35, maxRmseFrac: 0.6,
  mode: "novel", matchColor: true, sor: { k: 12, stdRatio: 2.5 }, mergeVoxel: 0.01,
};
// masterBuf / newBuf: PLY ArrayBuffers. Returns { ok, merged (PLY), T, reg, before, after, added, covered, ... } or { ok:false, reason }.
export function registerAndMerge(masterBuf, newBuf, opts = {}) {
  const o = { ...REGISTER_DEFAULTS, ...opts }, fail = (reason, extra = {}) => ({ ok: false, reason, ...extra });
  let master, fresh;
  try { master = readPLY(masterBuf); fresh = readPLY(newBuf); } catch (e) { return fail(e.message); }
  if (master.n < 200 || fresh.n < 200) return fail("a cloud has too few points");
  const target = makeTarget(master.pos); fillDefaultScale(master, target.tree);
  fillDefaultScale(fresh, new KDTree(fresh.pos));
  const sp = estimateSpacing(target.tree);
  let thin = fresh.pos; if (o.voxel > 0) thin = subsetCloud(fresh, voxelIndices(fresh.pos, o.voxel)).pos;
  let T0 = identity();
  if (o.init && o.init.R) T0 = { s: 1, ...o.init };
  else if (o.init === "centroid" || o.init === "pca") {
    if (o.init === "pca") { const c = coarseAlign(target, thin, o); if (!c.ok) return fail(c.reason); T0 = { R: c.R, t: c.t, s: c.s }; }
    else { const cm = (p) => { const c = [0, 0, 0], n = p.length / 3; for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) c[k] += p[i * 3 + k] / n; return c; }, a = cm(target.pos), b = cm(thin); T0 = { R: [1, 0, 0, 0, 1, 0, 0, 0, 1], t: [a[0] - b[0], a[1] - b[1], a[2] - b[2]], s: 1 }; }
  }
  const placed0 = applyToPositions(T0, thin), tolStages = o.stages || [{ maxDist: Math.max(sp * 30, o.tol * 8), iters: 15 }, { maxDist: Math.max(sp * 10, o.tol * 3), iters: 15 }, { maxDist: Math.max(sp * 4, o.tol), iters: 20 }];
  const before = distanceStats(signedC2C(target, placed0, { maxDist: tolStages[0].maxDist }), { thresholds: [o.tol] });
  const r = icp(target, placed0, { method: o.method, estimateScale: o.estimateScale, stages: tolStages });
  if (!r.ok) return fail(r.reason, { before });
  const T = compose(r, T0);
  const lastD = tolStages[tolStages.length - 1].maxDist;
  if (r.fitness < o.minFitness) return fail(`the clouds overlap too little after alignment (${(r.fitness * 100).toFixed(0)}% matched)`, { T, reg: r, before });
  if (r.rmse > o.maxRmseFrac * lastD && r.rmse > o.tol) return fail(`alignment residual is high (${(r.rmse * 1000).toFixed(1)} mm)`, { T, reg: r, before });
  const placed = transformCloud(fresh, T), after = distanceStats(signedC2C(target, placed.pos, { maxDist: tolStages[0].maxDist }), { thresholds: [o.tol] });
  const m = mergeClouds(master, placed, { mode: o.mode, tol: o.tol, voxel: o.mergeVoxel, matchColor: o.matchColor, sor: o.sor, target });
  return { ok: true, merged: writePLY(m.cloud), T, reg: { rmse: r.rmse, fitness: r.fitness, iterations: r.iterations, converged: r.converged, condition: r.condition }, before, after,
    added: m.added, covered: m.covered, thinned: m.thinned, outliers: m.outliers, colorGain: m.colorGain, total: m.cloud.n };
}
