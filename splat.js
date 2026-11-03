// Minimal WebGL2 3D Gaussian Splat renderer: parses 3DGS-style binary PLY, sorts splats
// front-to-back, projects each 3D covariance to a 2D ellipse (EWA splatting) and blends.
const VS = `#version 300 es
precision highp float;
in vec2 corner;
in vec3 center; in vec4 color; in vec3 covA; in vec3 covB;
uniform mat4 view, proj; uniform vec2 viewport; uniform vec2 focal;
out vec4 vColor; out vec2 vPos;
void main() {
  vec4 cam = view * vec4(center, 1.0);
  if (cam.z > -0.05) { gl_Position = vec4(0.0, 0.0, 2.0, 1.0); return; }
  vec4 p = proj * cam;
  float zp = -cam.z;
  mat3 J = mat3(focal.x / zp, 0.0, 0.0,
                0.0, focal.y / zp, 0.0,
                focal.x * cam.x / (zp * zp), focal.y * cam.y / (zp * zp), 0.0);
  mat3 T = J * mat3(view);
  mat3 V = mat3(covA.x, covA.y, covA.z, covA.y, covB.x, covB.y, covA.z, covB.y, covB.z);
  mat3 C = T * V * transpose(T);
  float a = C[0][0] + 0.3, b = C[0][1], c = C[1][1] + 0.3;
  float mid = 0.5 * (a + c);
  float r = length(vec2(0.5 * (a - c), b));
  float l1 = mid + r, l2 = max(mid - r, 0.1);
  vec2 dir = normalize(vec2(b, l1 - a));
  vec2 major = min(sqrt(8.0 * l1), 1024.0) * dir;
  vec2 minor = min(sqrt(8.0 * l2), 1024.0) * vec2(dir.y, -dir.x);
  vColor = color; vPos = corner;
  vec2 c0 = p.xy / p.w;
  gl_Position = vec4(c0 + corner.x * major / viewport + corner.y * minor / viewport, 0.0, 1.0);
}`;
const FS = `#version 300 es
precision highp float;
in vec4 vColor; in vec2 vPos; out vec4 o;
void main() {
  float A = -dot(vPos, vPos);
  if (A < -4.0) discard;
  float B = exp(A) * vColor.a;
  o = vec4(B * vColor.rgb, B);
}`;

const SH_C0 = 0.28209479177387814, STRIDE = 13;

function compile(gl, type, src) {
  const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
  return s;
}
function perspective(fovy, asp, n, f) {
  const t = 1 / Math.tan(fovy / 2), m = new Float32Array(16);
  m[0] = t / asp; m[5] = t; m[10] = (f + n) / (n - f); m[11] = -1; m[14] = (2 * f * n) / (n - f);
  return m;
}
function lookAt(e, c, u) {
  let z = [e[0] - c[0], e[1] - c[1], e[2] - c[2]]; let l = Math.hypot(...z); z = z.map((v) => v / l);
  let x = [u[1] * z[2] - u[2] * z[1], u[2] * z[0] - u[0] * z[2], u[0] * z[1] - u[1] * z[0]];
  l = Math.hypot(...x); x = x.map((v) => v / l);
  const y = [z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0]];
  return new Float32Array([x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0,
    -(x[0] * e[0] + x[1] * e[1] + x[2] * e[2]), -(y[0] * e[0] + y[1] * e[1] + y[2] * e[2]),
    -(z[0] * e[0] + z[1] * e[1] + z[2] * e[2]), 1]);
}

export function parsePLY(buf) {
  const head = new TextDecoder().decode(new Uint8Array(buf, 0, Math.min(buf.byteLength, 4096)));
  const end = head.indexOf("end_header\n");
  if (end < 0) throw new Error("Invalid PLY header");
  const lines = head.slice(0, end).split("\n");
  const count = parseInt(lines.find((l) => l.startsWith("element vertex")).split(" ")[2], 10);
  const props = lines.filter((l) => l.startsWith("property")).map((l) => l.split(" ")[2]);
  const idx = Object.fromEntries(props.map((p, i) => [p, i]));
  const f = new Float32Array(buf.slice(end + 11), 0, count * props.length);
  const P = props.length, out = new Float32Array(count * STRIDE);
  const sig = (x) => 1 / (1 + Math.exp(-x));
  for (let i = 0; i < count; i++) {
    const g = (n) => f[i * P + idx[n]], o = i * STRIDE;
    out[o] = g("x"); out[o + 1] = g("y"); out[o + 2] = g("z");
    out[o + 3] = Math.min(Math.max(0.5 + SH_C0 * g("f_dc_0"), 0), 1);
    out[o + 4] = Math.min(Math.max(0.5 + SH_C0 * g("f_dc_1"), 0), 1);
    out[o + 5] = Math.min(Math.max(0.5 + SH_C0 * g("f_dc_2"), 0), 1);
    out[o + 6] = sig(g("opacity"));
    const s = [Math.exp(g("scale_0")), Math.exp(g("scale_1")), Math.exp(g("scale_2"))];
    let q = [g("rot_0"), g("rot_1"), g("rot_2"), g("rot_3")];
    const ql = Math.hypot(...q); q = q.map((v) => v / ql);
    const [w, x, y, z] = q;
    const R = [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
      2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
      2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)];
    const M = R.map((v, k) => v * s[k % 3]);          // M = R * diag(s)
    const cv = (r, c) => M[r * 3] * M[c * 3] + M[r * 3 + 1] * M[c * 3 + 1] + M[r * 3 + 2] * M[c * 3 + 2];
    out[o + 7] = cv(0, 0); out[o + 8] = cv(0, 1); out[o + 9] = cv(0, 2);
    out[o + 10] = cv(1, 1); out[o + 11] = cv(1, 2); out[o + 12] = cv(2, 2);
  }
  return { data: out, count };
}

export class SplatViewer {
  constructor(canvas, { spinSpeed = 0.25, onHover = null } = {}) {
    this.canvas = canvas; this.spinSpeed = spinSpeed;
    this.onHover = onHover; this.pointer = null; this.hovering = false; this.units = "units";
    const gl = canvas.getContext("webgl2", { antialias: false });
    if (!gl) throw new Error("WebGL2 is required");
    this.gl = gl;
    const pr = gl.createProgram();
    gl.attachShader(pr, compile(gl, gl.VERTEX_SHADER, VS));
    gl.attachShader(pr, compile(gl, gl.FRAGMENT_SHADER, FS));
    gl.linkProgram(pr);
    if (!gl.getProgramParameter(pr, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(pr));
    this.pr = pr; gl.useProgram(pr);
    this.u = Object.fromEntries(["view", "proj", "viewport", "focal"].map((n) => [n, gl.getUniformLocation(pr, n)]));

    this.vao = gl.createVertexArray(); gl.bindVertexArray(this.vao);
    const quad = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, quad);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-2, -2, 2, -2, -2, 2, 2, 2]), gl.STATIC_DRAW);
    const lc = gl.getAttribLocation(pr, "corner");
    gl.enableVertexAttribArray(lc); gl.vertexAttribPointer(lc, 2, gl.FLOAT, false, 0, 0);
    this.inst = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, this.inst);
    [["center", 3, 0], ["color", 4, 3], ["covA", 3, 7], ["covB", 3, 10]].forEach(([n, sz, off]) => {
      const l = gl.getAttribLocation(pr, n);
      gl.enableVertexAttribArray(l);
      gl.vertexAttribPointer(l, sz, gl.FLOAT, false, STRIDE * 4, off * 4);
      gl.vertexAttribDivisor(l, 1);
    });
    gl.enable(gl.BLEND);
    gl.blendFuncSeparate(gl.ONE_MINUS_DST_ALPHA, gl.ONE, gl.ONE_MINUS_DST_ALPHA, gl.ONE);
    gl.clearColor(0, 0, 0, 0);

    this.count = 0; this.theta = 0; this.phi = 0.12; this.zoom = 1; this.lastSort = 0; this.last = performance.now();
    this._bindInput(); requestAnimationFrame((t) => this._frame(t));
  }

  // keepView: keep the current camera (used after an edit/undo) instead of re-centring on the cloud.
  load(buf, { units = "units", keepView = false } = {}) {
    this.units = units;                       // "m" when a metric scale has been applied
    const { data, count } = parsePLY(buf);
    this._setData(data, count);
    if (!keepView && count) {
      let cx = 0, cy = 0, cz = 0;
      for (let i = 0; i < count; i++) { cx += data[i * STRIDE]; cy += data[i * STRIDE + 1]; cz += data[i * STRIDE + 2]; }
      cx /= count; cy /= count; cz /= count;
      let r2 = 0;
      for (let i = 0; i < count; i++) r2 += (data[i * STRIDE] - cx) ** 2 + (data[i * STRIDE + 1] - cy) ** 2 + (data[i * STRIDE + 2] - cz) ** 2;
      this.target = [cx, cy, cz];
      this.dist = Math.max(Math.sqrt(r2 / count) * 2.2, 0.5);
      this.theta = 0;
    }
    return count;
  }

  _setData(data, count) {
    this.base = data; this.count = count;
    this.sorted = new Float32Array(data.length);
    this.keys = new Uint16Array(count); this.order = new Uint32Array(count); this.zs = new Float32Array(count);
    this.counts = new Uint32Array(65537);
    this.sel = new Uint8Array(count); this.selCount = 0;
    this.lastSort = 0;
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.inst);
    gl.bufferData(gl.ARRAY_BUFFER, Math.max(this.sorted.byteLength, 4), gl.DYNAMIC_DRAW);
  }

  // ----- selection / deletion -----
  // poly: [{x, y}] in CSS pixels inside the canvas. mode: "add" | "sub" | "set".
  // Selects every point (also hidden ones behind others) whose screen position falls inside the polygon,
  // using the camera of the last drawn frame. Selected points are drawn in orange.
  selectPoly(poly, mode = "add") {
    if (!this.count || !this.lastView || poly.length < 3) return this.selCount;
    const { base, count, sel } = this, v = this.lastView, p = this.lastProj;
    const W = this.lastW, H = this.lastH, dpr = this.lastDpr, fx = p[0], fy = p[5];
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const q of poly) { x0 = Math.min(x0, q.x); y0 = Math.min(y0, q.y); x1 = Math.max(x1, q.x); y1 = Math.max(y1, q.y); }
    if (mode === "set") sel.fill(0);
    const n = poly.length, val = mode === "sub" ? 0 : 1;
    for (let i = 0; i < count; i++) {
      const o = i * STRIDE, X = base[o], Y = base[o + 1], Z = base[o + 2];
      const depth = -(v[2] * X + v[6] * Y + v[10] * Z + v[14]);
      if (depth < 0.05) continue;
      const sx = (0.5 + 0.5 * ((fx * (v[0] * X + v[4] * Y + v[8] * Z + v[12])) / depth)) * W / dpr;
      const sy = (0.5 - 0.5 * ((fy * (v[1] * X + v[5] * Y + v[9] * Z + v[13])) / depth)) * H / dpr;
      if (sx < x0 || sx > x1 || sy < y0 || sy > y1) continue;
      let inside = false;
      for (let a = 0, b = n - 1; a < n; b = a++) {      // even-odd rule
        const A = poly[a], B = poly[b];
        if ((A.y > sy) !== (B.y > sy) && sx < ((B.x - A.x) * (sy - A.y)) / (B.y - A.y) + A.x) inside = !inside;
      }
      if (inside) sel[i] = val;
    }
    let c = 0; for (let i = 0; i < count; i++) c += sel[i];
    this.selCount = c; this.lastSort = 0;                // force a re-sort so the highlight is uploaded
    return c;
  }

  clearSelection() {
    if (!this.sel) return;
    this.sel.fill(0); this.selCount = 0; this.lastSort = 0;
  }

  // Removes the selected points. Returns a copy of the selection mask (indexed like the data before the
  // removal) so the caller can apply the same removal to its PLY, or null if nothing was selected.
  deleteSelected() {
    if (!this.count || !this.selCount) return null;
    const { base, count, sel } = this, del = sel.slice(), keep = count - this.selCount;
    const nb = new Float32Array(keep * STRIDE);
    let j = 0;
    for (let i = 0; i < count; i++) if (!del[i]) { nb.set(base.subarray(i * STRIDE, (i + 1) * STRIDE), j * STRIDE); j++; }
    this._setData(nb, keep);
    return del;
  }

  zoomBy(f) { this.zoom = Math.max(0.2, Math.min(4, this.zoom * f)); }

  _sort(view) {
    const { base, count, zs, keys, counts, order, sorted } = this;
    const a = view[2], b = view[6], c = view[10], d = view[14];
    let mn = Infinity, mx = -Infinity;
    for (let i = 0; i < count; i++) {
      const z = a * base[i * STRIDE] + b * base[i * STRIDE + 1] + c * base[i * STRIDE + 2] + d;
      zs[i] = z; if (z < mn) mn = z; if (z > mx) mx = z;
    }
    const k = 65535 / Math.max(mx - mn, 1e-6);
    counts.fill(0);
    for (let i = 0; i < count; i++) { const q = ((mx - zs[i]) * k) | 0; keys[i] = q; counts[q + 1]++; }
    for (let i = 1; i < counts.length; i++) counts[i] += counts[i - 1];
    for (let i = 0; i < count; i++) order[counts[keys[i]]++] = i;      // nearest first
    const sel = this.sel;
    for (let j = 0; j < count; j++) {
      const s = order[j] * STRIDE, o = j * STRIDE;
      for (let m = 0; m < STRIDE; m++) sorted[o + m] = base[s + m];
      if (sel[order[j]]) { sorted[o + 3] = 1; sorted[o + 4] = 0.3; sorted[o + 5] = 0.1; sorted[o + 6] = 1; }
    }
    const gl = this.gl; gl.bindBuffer(gl.ARRAY_BUFFER, this.inst);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, sorted);
  }

  _setPointer(e) {
    const r = this.canvas.getBoundingClientRect();
    this.pointer = { x: e.clientX - r.left, y: e.clientY - r.top };   // CSS pixels inside the canvas
  }

  // Finds the front-most Gaussian centre within 8 px of the pointer and reports its coordinates.
  // Coordinates are those of the capture camera: X right, Y up, Depth = distance in front of the camera.
  _pick(view, proj, W, H, dpr) {
    if (!this.onHover) return;
    if (!this.pointer || !this.count) {
      if (this.hovering) { this.hovering = false; this.onHover(null); }
      return;
    }
    const px = this.pointer.x * dpr, py = this.pointer.y * dpr, R2 = (8 * dpr) ** 2;
    const { base, count } = this, fx = proj[0], fy = proj[5];
    let best = -1, bestDepth = Infinity, bsx = 0, bsy = 0;
    for (let i = 0; i < count; i++) {
      const o = i * STRIDE, X = base[o], Y = base[o + 1], Z = base[o + 2];
      const depth = -(view[2] * X + view[6] * Y + view[10] * Z + view[14]);
      if (depth < 0.05 || depth >= bestDepth) continue;
      const sx = (0.5 + 0.5 * ((fx * (view[0] * X + view[4] * Y + view[8] * Z + view[12])) / depth)) * W;
      const sy = (0.5 - 0.5 * ((fy * (view[1] * X + view[5] * Y + view[9] * Z + view[13])) / depth)) * H;
      const dx = sx - px, dy = sy - py;
      if (dx * dx + dy * dy <= R2) { best = i; bestDepth = depth; bsx = sx; bsy = sy; }
    }
    if (best < 0) {
      if (this.hovering) { this.hovering = false; this.onHover(null); }
      return;
    }
    const o = best * STRIDE;
    this.hovering = true;
    this.onHover({ x: base[o], y: base[o + 1], z: base[o + 2], depth: -base[o + 2], units: this.units,
      sx: bsx / dpr, sy: bsy / dpr, px: this.pointer.x, py: this.pointer.y });
  }

  _bindInput() {
    const cv = this.canvas; let drag = null, pinch = null;
    cv.style.touchAction = "none";
    cv.addEventListener("pointerdown", (e) => { drag = { x: e.clientX, y: e.clientY }; cv.setPointerCapture(e.pointerId); this._setPointer(e); });
    cv.addEventListener("pointerleave", (e) => { if (e.pointerType === "mouse") this.pointer = null; });
    cv.addEventListener("pointerup", () => (drag = null));
    cv.addEventListener("pointermove", (e) => {
      this._setPointer(e);
      if (drag == null) return;
      this.phi = Math.max(-1.2, Math.min(1.2, this.phi + (e.clientY - drag.y) * 0.005));
      this.theta -= (e.clientX - drag.x) * 0.005;           // horizontal drag turns the model too
      drag = { x: e.clientX, y: e.clientY };
    });
    cv.addEventListener("wheel", (e) => {
      e.preventDefault(); this.zoom = Math.max(0.2, Math.min(4, this.zoom * (e.deltaY > 0 ? 1.08 : 0.92)));
    }, { passive: false });
    cv.addEventListener("touchmove", (e) => {
      if (e.touches.length !== 2) return;
      const d = Math.hypot(e.touches[0].clientX - e.touches[1].clientX, e.touches[0].clientY - e.touches[1].clientY);
      if (pinch) this.zoom = Math.max(0.2, Math.min(4, this.zoom * (pinch / d)));
      pinch = d;
    });
    cv.addEventListener("touchend", () => (pinch = null));
  }

  _frame(now) {
    requestAnimationFrame((t) => this._frame(t));
    const gl = this.gl, cv = this.canvas, dpr = window.devicePixelRatio || 1;
    const W = Math.round(cv.clientWidth * dpr), H = Math.round(cv.clientHeight * dpr);
    if (cv.width !== W || cv.height !== H) { cv.width = W; cv.height = H; }
    gl.viewport(0, 0, W, H); gl.clear(gl.COLOR_BUFFER_BIT);
    const dt = Math.min((now - this.last) / 1000, 0.1); this.last = now;
    if (!this.count || !W || !H) return;

    this.theta += this.spinSpeed * dt;                    // slow automatic spin about the vertical axis
    const D = this.dist * this.zoom, [tx, ty, tz] = this.target;
    const eye = [tx + D * Math.sin(this.theta) * Math.cos(this.phi), ty + D * Math.sin(this.phi),
      tz + D * Math.cos(this.theta) * Math.cos(this.phi)];
    const view = lookAt(eye, this.target, [0, 1, 0]);
    const proj = perspective((50 * Math.PI) / 180, W / H, 0.05, 200);
    this.lastView = view; this.lastProj = proj; this.lastW = W; this.lastH = H; this.lastDpr = dpr;
    if (now - this.lastSort > 80) { this._sort(view); this.lastSort = now; }

    gl.useProgram(this.pr); gl.bindVertexArray(this.vao);
    gl.uniformMatrix4fv(this.u.view, false, view); gl.uniformMatrix4fv(this.u.proj, false, proj);
    gl.uniform2f(this.u.viewport, W, H);
    gl.uniform2f(this.u.focal, (proj[0] * W) / 2, (proj[5] * H) / 2);
    gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, this.count);
    this._pick(view, proj, W, H, dpr);
  }
}
