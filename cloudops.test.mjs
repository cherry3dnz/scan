// Browser: open cloudops.test.html (served next to cloudops.js).  Node 18+ also works: node cloudops.test.mjs
// Optional end-to-end part: if pipeline.js / splat.js / register.js are next to it, it also runs against your real code.
import * as C from "./cloudops.js";

let pass = 0, fail = 0;
const ok = (name, cond, info = "") => { cond ? pass++ : fail++; console.log((cond ? "  PASS " : "  FAIL ") + name + (info ? "  " + info : "")); };
const sec = (t) => console.log("\n" + t);
const ms = (f) => { const t = performance.now(); const r = f(); return [r, Math.round(performance.now() - t)]; };

// ---------- helpers ----------
function rng(seed) { let a = seed >>> 0; return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
const gauss = (r) => Math.sqrt(-2 * Math.log(r() + 1e-12)) * Math.cos(2 * Math.PI * r());
const mulRR = (A, B) => { const o = new Array(9); for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) o[r * 3 + c] = A[r * 3] * B[c] + A[r * 3 + 1] * B[3 + c] + A[r * 3 + 2] * B[6 + c]; return o; };
const tr = (R) => [R[0], R[3], R[6], R[1], R[4], R[7], R[2], R[5], R[8]];
const angDeg = (A, B) => (C.rotationAngle(mulRR(tr(A), B)) * 180) / Math.PI;
const RAD = Math.PI / 180;

// A room corner with a sphere and a box. Surface samples + gaussian noise sigma. Optional x-range to fake partial views.
function scene(n, seed, { sigma = 0, xmin = -9, xmax = 9, color = [1, 1, 1] } = {}) {
  const r = rng(seed), pos = new Float32Array(n * 3), rgb = new Uint8Array(n * 3);
  const S = [["floor", 9], ["back", 6], ["left", 6], ["sphere", 3], ["box", 1.6]], tot = S.reduce((s, q) => s + q[1], 0);
  let k = 0;
  while (k < n) {
    let u = r() * tot, kind = S[0][0]; for (const [nm, w] of S) { if (u < w) { kind = nm; break; } u -= w; }
    let p, c;
    if (kind === "floor") { p = [-1.5 + 3 * r(), 0, -3 * r()]; c = [120, 100, 80]; }
    else if (kind === "back") { p = [-1.5 + 3 * r(), 2 * r(), -3]; c = [200, 190, 170]; }
    else if (kind === "left") { p = [-1.5, 2 * r(), -3 * r()]; c = [170, 180, 200]; }
    else if (kind === "sphere") { const v = [gauss(r), gauss(r), gauss(r)], l = Math.hypot(...v); p = [0.3 + (0.35 * v[0]) / l, 0.35 + (0.35 * v[1]) / l, -1.6 + (0.35 * v[2]) / l]; if (p[1] < 0.35 - 0.34) continue; c = [200, 50, 50]; }
    else { const f = Math.floor(r() * 5), a = r(), b = r(); const x0 = -0.9, x1 = -0.4, y1 = 0.5, z0 = -2.4, z1 = -1.9;
      p = f === 0 ? [x0 + (x1 - x0) * a, y1, z0 + (z1 - z0) * b] : f === 1 ? [x0, y1 * a, z0 + (z1 - z0) * b] : f === 2 ? [x1, y1 * a, z0 + (z1 - z0) * b] : f === 3 ? [x0 + (x1 - x0) * a, y1 * b, z0] : [x0 + (x1 - x0) * a, y1 * b, z1]; c = [60, 90, 200]; }
    if (p[0] < xmin || p[0] > xmax) continue;
    for (let q = 0; q < 3; q++) { pos[k * 3 + q] = p[q] + sigma * gauss(r); rgb[k * 3 + q] = Math.max(0, Math.min(255, (c[q] + 8 * gauss(r)) * color[q])); }
    k++;
  }
  return { pos, rgb };
}
const SH = C.SH_C0;
function cloudOf(pos, rgb) {
  const n = pos.length / 3, rec = new Float32Array(n * 17);
  for (let i = 0; i < n; i++) { const o = i * 17; rec[o] = pos[i * 3]; rec[o + 1] = pos[i * 3 + 1]; rec[o + 2] = pos[i * 3 + 2]; for (let c = 0; c < 3; c++) rec[o + 6 + c] = (rgb[i * 3 + c] / 255 - 0.5) / SH; rec[o + 9] = Math.log(9); rec[o + 10] = rec[o + 11] = rec[o + 12] = Math.log(0.01); rec[o + 13] = 1; }
  return C.cloudFromRecords(rec);
}
// transform error: rotation (deg) and the mean displacement it leaves on a point set
function tErr(T, Tref, pts) {
  const a = C.applyToPositions(T, pts), b = C.applyToPositions(Tref, pts); let s = 0, n = pts.length / 3;
  for (let i = 0; i < n; i++) s += Math.hypot(a[i * 3] - b[i * 3], a[i * 3 + 1] - b[i * 3 + 1], a[i * 3 + 2] - b[i * 3 + 2]);
  return { rot: angDeg(T.R, Tref.R), mm: (s / n) * 1000 };
}
const mk = (axis, deg, t, s = 1) => { const l = Math.hypot(...axis); return { R: C.rotAxis(axis[0] / l * deg * RAD, axis[1] / l * deg * RAD, axis[2] / l * deg * RAD), t, s }; };

// ==========================================================================================
sec("1. KDTree vs brute force");
{
  const r = rng(5), n = 6000, p = new Float32Array(n * 3); for (let i = 0; i < p.length; i++) p[i] = r() * 2 - 1;
  const tree = new C.KDTree(p), oi = new Int32Array(8), od = new Float64Array(8); let bad = 0, badK = 0;
  for (let q = 0; q < 300; q++) {
    const x = r() * 2 - 1, y = r() * 2 - 1, z = r() * 2 - 1, ds = [];
    for (let i = 0; i < n; i++) ds.push([(p[i * 3] - x) ** 2 + (p[i * 3 + 1] - y) ** 2 + (p[i * 3 + 2] - z) ** 2, i]);
    ds.sort((a, b) => a[0] - b[0]);
    if (tree.nearest(x, y, z) !== ds[0][1]) bad++;
    const c = tree.knn(x, y, z, 8, oi, od); if (c !== 8 || [...oi].some((v, i) => v !== ds[i][1])) badK++;
    const lim = ds[0][0] * 0.5; if (tree.nearest(x, y, z, lim) !== -1) bad++;                 // maxD2 respected
  }
  ok("nearest matches brute force (300 queries, with maxD2)", bad === 0, "bad=" + bad);
  ok("knn(8) matches brute force", badK === 0, "bad=" + badK);
  const dup = new Float32Array(300).fill(0.5), t2 = new C.KDTree(dup);
  ok("tree builds on all-identical points", t2.nearest(0.5, 0.5, 0.5) >= 0);
}

sec("2. Transform algebra");
{
  const A = mk([1, 2, 3], 37, [0.1, -0.2, 0.3], 1.3), B = mk([0, 1, 0], -21, [1, 2, 3]);
  const I = C.compose(C.compose(A, B), C.invert(C.compose(A, B))), p = new Float32Array([0.3, -1, 2, 5, 5, 5]), q = C.applyToPositions(I, p);
  ok("T o T^-1 = identity (with scale)", [...q].every((v, i) => Math.abs(v - p[i]) < 1e-5));
  const ab = C.applyToPositions(C.compose(A, B), p), a2 = C.applyToPositions(A, C.applyToPositions(B, p));
  ok("compose(A,B) applies B first", [...ab].every((v, i) => Math.abs(v - a2[i]) < 1e-4));
}

sec("3. Registration: small misalignment (the Add-view case)");
let sceneA, sceneB;
{
  sceneA = scene(60000, 1, { sigma: 0.001 }); sceneB = scene(30000, 2, { sigma: 0.002 });
  const target = C.makeTarget(sceneA.pos), gt = mk([1, 2, 3], 4, [0.03, -0.02, 0.025]);        // how B was displaced
  const src = C.applyToPositions(gt, sceneB.pos), want = C.invert(gt);
  for (const method of ["plane", "point"]) {
    const [r, t] = ms(() => C.icp(target, src, { method }));
    const e = tErr(r, want, src);
    ok(`${method}: recovers the displacement`, r.ok && e.rot < 0.1 && e.mm < 1.5, `rot ${e.rot.toFixed(3)} deg, ${e.mm.toFixed(2)} mm, ${r.iterations} iters, ${t} ms, fitness ${(r.fitness * 100).toFixed(0)}%, rmse ${(r.rmse * 1000).toFixed(2)} mm`);
  }
  const big = mk([0.3, 1, 0.2], 9, [0.12, 0.05, -0.1]), s2 = C.applyToPositions(big, sceneB.pos), r = C.icp(target, s2, {});
  const e = tErr(r, C.invert(big), s2);
  ok("default stages capture a 9 deg / 16 cm error", r.ok && e.rot < 0.2 && e.mm < 2, `rot ${e.rot.toFixed(3)} deg, ${e.mm.toFixed(2)} mm`);
  const far = C.applyToPositions(mk([0, 1, 0], 5, [30, 30, 30]), sceneB.pos), rf = C.icp(target, far, {});
  ok("no overlap is reported, not faked", !rf.ok, rf.reason || "");
}

sec("4. Registration: unknown pose (file import), PCA start");
{
  const A = scene(60000, 11, { sigma: 0.001 }), target = C.makeTarget(A.pos);
  const cases = [
    ["full overlap, 70 deg / 1.1 m", { xmin: -9, xmax: 9 }, mk([0.2, 1, 0.4], 70, [0.6, -0.3, 0.9]), true],
    ["75% overlap (right side missing), 110 deg", { xmin: -9, xmax: 1.0 }, mk([1, 0.3, 0.2], 110, [-0.5, 0.4, 0.7]), true],
    ["80% overlap but a whole wall missing, 70 deg  [hard]", { xmin: -1.1, xmax: 9 }, mk([0.2, 1, 0.4], 70, [0.6, -0.3, 0.9]), false],
  ];
  cases.forEach(([name, range, gt, must], k) => {
    const B = scene(40000, 12 + k, { sigma: 0.002, ...range }), srcPos = C.applyToPositions(gt, B.pos);
    const [c, t] = ms(() => C.coarseAlign(target, srcPos)), r = c.ok ? C.icp(target, C.applyToPositions(c, srcPos), { method: "plane" }) : { ok: false };
    const e = c.ok && r.ok ? tErr(C.compose(r, c), C.invert(gt), srcPos) : { rot: NaN, mm: NaN };
    const good = c.ok && r.ok && e.rot < 0.3 && e.mm < 3;
    if (must) ok("PCA start + ICP: " + name, good, `rot ${e.rot.toFixed(3)} deg, ${e.mm.toFixed(2)} mm, ${t} ms`);
    else console.log(`  INFO  ${name}: ${good ? "recovered" : "NOT recovered"} (${e.rot.toFixed(1)} deg, ${e.mm.toFixed(0)} mm, ${t} ms)`);
  });
}

sec("5. Registration: unknown scale");
{
  const A = scene(50000, 21, { sigma: 0.001 }), B = scene(30000, 22, { sigma: 0.001 }), target = C.makeTarget(A.pos);
  const gt = mk([1, 1, 0], 5, [0.04, 0.02, -0.03], 1.15), src = C.applyToPositions(gt, B.pos), r = C.icp(target, src, { estimateScale: true });
  ok("estimateScale recovers s = 1/1.15", r.ok && Math.abs(r.s - 1 / 1.15) < 0.004 && tErr(r, C.invert(gt), src).mm < 2, `s ${r.s.toFixed(4)} vs ${(1 / 1.15).toFixed(4)}, ${tErr(r, C.invert(gt), src).mm.toFixed(2)} mm`);
}

sec("6. Comparison: C2C, plane distance, statistics, heat");
{
  const g = 200, n = g * g, A = new Float32Array(n * 3), B = new Float32Array(n * 3);
  for (let i = 0; i < g; i++) for (let j = 0; j < g; j++) { const k = i * g + j; A.set([i * 0.01, 0, j * 0.01], k * 3); B.set([i * 0.01, 0.005, j * 0.01], k * 3); }   // grid, plane 5 mm above
  const target = C.makeTarget(A), { dist } = C.c2c(target, B), st = C.distanceStats(dist, { thresholds: [0.004, 0.006, 0.01], bins: 20 });
  ok("planes 5 mm apart, same sampling: every distance is 5 mm", Math.abs(st.median - 0.005) < 1e-4 && Math.abs(st.max - 0.005) < 1e-4, `median ${(st.median * 1000).toFixed(2)} mm`);
  ok("within(6 mm) = 100%, within(4 mm) = 0%", st.within[0.006] > 0.999 && st.within[0.004] < 0.001);
  ok("histogram counts sum to n", st.hist.counts.reduce((a, b) => a + b, 0) === st.n && st.hist.edges.length === 21);
  // independent sampling: nearest-neighbour distance is inflated by the point spacing, plane distance is not
  const r = rng(31), Bq = new Float32Array(n * 3); for (let i = 0; i < n; i++) Bq.set([r() * 2, 0.005, r() * 2], i * 3);
  const nnMed = C.distanceStats(C.c2c(target, Bq).dist).median, plMed = C.distanceStats(C.signedC2C(target, Bq)).median;
  ok("independent sampling: NN median is inflated, plane median stays at 5 mm", nnMed > 0.0058 && Math.abs(plMed - 0.005) < 3e-4, `NN ${(nnMed * 1000).toFixed(2)} mm vs plane ${(plMed * 1000).toFixed(2)} mm`);
  const sg = C.signedC2C(target, B, { orient: [0, 5, 0] });
  ok("signed distance: +5 mm on the viewer side, -5 mm on the other", Math.abs(sg[0] - 0.005) < 3e-4 && Math.abs(C.signedC2C(target, B, { orient: [0, -5, 0] })[0] + 0.005) < 3e-4, `${(sg[0] * 1000).toFixed(2)} mm`);
  const far = new Float32Array([0, 3, 0]), d2 = C.c2c(target, far, { maxDist: 0.1 }); ok("maxDist leaves Infinity and a -1 index", d2.dist[0] === Infinity && d2.nn[0] === -1 && C.signedC2C(target, far, { maxDist: 0.1 })[0] === Infinity);
  const hc = C.heatColors(new Float32Array([0, 0.01, 0.02, Infinity]), 0.02);
  ok("heat: 0 is blue, vmax is red, missing is grey", hc[2] === 255 && hc[0] === 0 && hc[6] === 255 && hc[8] === 0 && hc[9] === 150, `[${[...hc.slice(0, 3)]}] [${[...hc.slice(3, 6)]}] [${[...hc.slice(6, 9)]}]`);
  const sym = C.compareSymmetric(cloudOf(A, new Uint8Array(n * 3)), cloudOf(B, new Uint8Array(n * 3)));
  ok("symmetric compare: chamfer = 10 mm, hausdorff = 5 mm", Math.abs(sym.chamfer - 0.01) < 2e-4 && Math.abs(sym.hausdorff - 0.005) < 1e-4, `chamfer ${(sym.chamfer * 1000).toFixed(2)} mm`);
}

sec("7. Outlier removal (SOR)");
{
  const S = scene(30000, 41, { sigma: 0.002 }), r = rng(42), nOut = 300, all = new Float32Array((30000 + nOut) * 3); all.set(S.pos);
  for (let i = 0; i < nOut; i++) all.set([r() * 4 - 2, r() * 3, -r() * 4], (30000 + i) * 3);
  const keep = C.sorMask(all, { k: 12, stdRatio: 2.5 });
  let outRemoved = 0, inKept = 0; for (let i = 0; i < nOut; i++) if (!keep[30000 + i]) outRemoved++; for (let i = 0; i < 30000; i++) if (keep[i]) inKept++;
  ok("removes >90% of floating points, keeps >97% of the surface", outRemoved / nOut > 0.9 && inKept / 30000 > 0.97, `${outRemoved}/${nOut} removed, ${inKept}/30000 kept`);
}

sec("8. Merging");
{
  const M = cloudOf(...Object.values(scene(50000, 51, { sigma: 0.002, xmax: 0.3 })));
  const Bs = scene(50000, 52, { sigma: 0.002, xmin: -0.6, color: [0.9, 0.9, 0.9] }), nF = 200, r = rng(53);
  const bp = new Float32Array((50000 + nF) * 3), bc = new Uint8Array((50000 + nF) * 3); bp.set(Bs.pos); bc.set(Bs.rgb);
  for (let i = 0; i < nF; i++) { bp.set([r() * 3 - 1.5, r() * 2, -r() * 3], (50000 + i) * 3); bc.fill(255, (50000 + i) * 3, (50000 + i) * 3 + 3); }
  const Bc = cloudOf(bp, bc), tol = 0.015, res = C.mergeClouds(M, Bc, { tol, voxel: 0.01 });
  const tgt = C.makeTarget(M.pos), mine = res.cloud.pos.subarray(M.n * 3), d = C.c2c(tgt, mine).dist;
  let minD = Infinity; for (const v of d) minD = Math.min(minD, v);
  ok("added points are all new (>= tol from every master point)", res.added > 3000 && minD >= tol * 0.999, `added ${res.added}, covered ${res.covered}, thinned ${res.thinned}, outliers ${res.outliers}, nearest ${(minD * 1000).toFixed(1)} mm`);
  ok("SOR removes the floaters but not the seam (outliers < 2% of added)", res.outliers >= nF * 0.8 && res.outliers < 0.02 * (res.added + res.outliers), `${res.outliers} removed`);
  ok("master records are untouched at the front", res.cloud.rec.subarray(0, M.rec.length).every((v, i) => v === M.rec[i]));
  ok("total = master + added", res.cloud.n === M.n + res.added);
  ok("colour match estimates the exposure gain (photo darkened by 0.9 -> gain 1.11)", res.colorGain && Math.abs(res.colorGain.gain[1] - 1 / 0.9) < 0.06, `gain ${res.colorGain && res.colorGain.gain.map((v) => v.toFixed(3))}`);
  const all = C.mergeClouds(M, Bc, { mode: "all", voxel: 0, sor: null, matchColor: false });
  ok("mode 'all' appends everything", all.cloud.n === M.n + Bc.n);
  const px = new Float32Array(3000 * 3); for (let i = 0; i < 3000; i++) px.set([(i % 30) * 0.001, Math.floor(i / 30) * 0.001, 0], i * 3);
  ok("voxelIndices keeps one point per voxel", C.voxelIndices(px, 0.01).length === 30);
}

sec("9. PLY I/O and splat attributes");
{
  const S = scene(2000, 61), cl = cloudOf(S.pos, S.rgb), buf = C.writePLY(cl), back = C.readPLY(buf);
  ok("write -> read round trip is exact", back.n === cl.n && back.rec.every((v, i) => v === cl.rec[i]));
  // plain xyz + uchar rgb PLY, the usual export of other tools
  const n = 500, head = new TextEncoder().encode(`ply\nformat binary_little_endian 1.0\nelement vertex ${n}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n`);
  const body = new ArrayBuffer(n * 15), dv = new DataView(body); for (let i = 0; i < n; i++) { dv.setFloat32(i * 15, i, true); dv.setFloat32(i * 15 + 4, 1, true); dv.setFloat32(i * 15 + 8, 2, true); dv.setUint8(i * 15 + 12, 255); dv.setUint8(i * 15 + 13, 128); dv.setUint8(i * 15 + 14, 0); }
  const f = new Uint8Array(head.length + body.byteLength); f.set(head); f.set(new Uint8Array(body), head.length);
  const x = C.readPLY(f.buffer), col = C.colorsOf(x);
  ok("reads xyz + uchar rgb PLY into the 17-float layout", x.n === n && x.pos[3] === 1 && col[0] === 255 && Math.abs(col[1] - 128) <= 1 && col[2] === 0 && x.hasScale === false);
  let threw = false; try { C.readPLY(new TextEncoder().encode("ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nend_header\n1\n").buffer); } catch (e) { threw = /binary/i.test(e.message); }
  ok("ASCII PLY gives a clear error", threw);
}

// ==========================================================================================
sec("10. Benchmark (this machine's V8; a phone is typically 3-6x slower)");
for (const N of [100000, 300000, 1000000]) {
  const r = rng(71), p = new Float32Array(N * 3); for (let i = 0; i < N; i++) { const a = r() * 6.28, h = r() * 2, rad = 1 + 0.05 * gauss(r); p.set([rad * Math.cos(a), h, rad * Math.sin(a)], i * 3); }
  const [tree, tb] = ms(() => new C.KDTree(p)), tgt = { pos: p, tree, normals: new C.NormalCache(tree) };
  const q = C.applyToPositions(mk([0, 1, 0], 3, [0.02, 0.01, 0.0]), p.subarray(0, 3 * 20000));
  const [res, ti] = ms(() => C.icp(tgt, q, { method: "plane" })), [, tc] = ms(() => C.c2c(tgt, p.subarray(0, 3 * 100000))), [, ts] = ms(() => C.sorMask(p.subarray(0, 3 * 100000)));
  console.log(`  ${String(N).padStart(8)} pts: tree build ${tb} ms | ICP (10k pts, ${res.ok ? res.iterations : "-"} iters) ${ti} ms | C2C of 100k pts ${tc} ms | SOR on 100k ${ts} ms`);
}

// ==========================================================================================
let e2e = false;
try { await import("./pipeline.js"); await import("./splat.js"); await import("./register.js"); e2e = true; } catch (e) { console.log("\n(end-to-end part skipped: " + e.message.split("\n")[0] + ")"); }
if (e2e) {
  sec("11. End to end with the app's own pipeline.js / splat.js / register.js");
  const P = await import("./pipeline.js"), SP = await import("./splat.js"), RG = await import("./register.js");
  const W = 320, H = 240, K = P.intrinsics(W, H);
  const norm = (v) => { const l = Math.hypot(...v); return v.map((a) => a / l); };
  // synthetic camera: ray-cast the room, returns z-depth per pixel + colours. pose = camera-to-world.
  function render(pose, seed) {
    const r = rng(seed), Z = new Float32Array(W * H), img = { width: W, height: H, data: new Uint8ClampedArray(W * H * 4) }, R = pose.R, t = pose.t;
    for (let v = 0; v < H; v++) for (let u = 0; u < W; u++) {
      const dc = [(u - K.cx) / K.fx, -(v - K.cy) / K.fy, -1], d = norm([R[0] * dc[0] + R[1] * dc[1] + R[2] * dc[2], R[3] * dc[0] + R[4] * dc[1] + R[5] * dc[2], R[6] * dc[0] + R[7] * dc[1] + R[8] * dc[2]]);
      let best = Infinity, col = [0, 0, 0];
      const hit = (s, c) => { if (s > 0.05 && s < best) { best = s; col = c; } };
      if (d[1] < -1e-6) { const s = -t[1] / d[1], x = t[0] + s * d[0], z = t[2] + s * d[2]; if (x > -1.5 && x < 1.5 && z > -3 && z < 0.8) hit(s, [120, 100, 80]); }
      if (d[2] < -1e-6) { const s = (-3 - t[2]) / d[2], x = t[0] + s * d[0], y = t[1] + s * d[1]; if (x > -1.5 && x < 1.5 && y > 0 && y < 2) hit(s, [200, 190, 170]); }
      if (d[0] < -1e-6) { const s = (-1.5 - t[0]) / d[0], y = t[1] + s * d[1], z = t[2] + s * d[2]; if (y > 0 && y < 2 && z > -3 && z < 0.8) hit(s, [170, 180, 200]); }
      { const oc = [t[0] - 0.3, t[1] - 0.35, t[2] + 1.6], b = oc[0] * d[0] + oc[1] * d[1] + oc[2] * d[2], c = oc[0] ** 2 + oc[1] ** 2 + oc[2] ** 2 - 0.35 ** 2, disc = b * b - c; if (disc > 0) hit(-b - Math.sqrt(disc), [200, 50, 50]); }
      { const lo = [-0.9, 0, -2.4], hi = [-0.4, 0.5, -1.9]; let t0 = 0, t1 = Infinity;
        for (let a = 0; a < 3; a++) { if (Math.abs(d[a]) < 1e-9) { if (t[a] < lo[a] || t[a] > hi[a]) { t1 = -1; } continue; } let s0 = (lo[a] - t[a]) / d[a], s1 = (hi[a] - t[a]) / d[a]; if (s0 > s1) [s0, s1] = [s1, s0]; t0 = Math.max(t0, s0); t1 = Math.min(t1, s1); }
        if (t0 < t1) hit(t0, [60, 90, 200]); }
      const p = v * W + u, zc = best === Infinity ? 8 : best * -(d[0] * R[2] + d[1] * R[5] + d[2] * R[8]);       // distance along the optical axis
      Z[p] = zc * (1 + 0.004 * gauss(r)); img.data[p * 4] = col[0]; img.data[p * 4 + 1] = col[1]; img.data[p * 4 + 2] = col[2]; img.data[p * 4 + 3] = 255;
    }
    const disp = new Float32Array(W * H); for (let i = 0; i < disp.length; i++) disp[i] = 1 / Z[i];
    return { img, disp };
  }
  const look = (e, tg) => RG.poseFromLookAt(e, tg);
  const pose1 = look([0, 1.0, 0.6], [0, 0.7, -2]), pose2 = look([0.45, 1.15, 0.5], [-0.15, 0.6, -2]);
  const v1 = render(pose1, 1), v2 = render(pose2, 2);
  const fit = (v) => { const { lo, span } = P.normParams(v.disp); return { a: span, b: lo }; };
  // master frame = camera 1 frame. pose of camera 2 in that frame:
  const rel = C.compose(C.invert({ R: pose1.R, t: pose1.t }), { R: pose2.R, t: pose2.t });
  const masterBuf = P.buildPLY(v1.img, v1.disp, fit(v1)), newBuf = P.buildPLY(v2.img, v2.disp, fit(v2));
  const mc = C.readPLY(masterBuf), nc = C.readPLY(newBuf);
  console.log(`  master ${mc.n} pts, new view ${nc.n} pts (320x240, stride 2)`);
  const guess = C.compose({ R: C.rotAxis(0.5 * RAD * 0.3, 2.0 * RAD, -1.0 * RAD), t: [0.03, -0.02, 0.025], s: 1 }, rel);          // the ghost pose is never exact
  const errOf = (T) => tErr(T, rel, nc.pos);
  const g = errOf(guess); console.log(`  start (like an imperfect ghost): ${g.rot.toFixed(2)} deg, ${g.mm.toFixed(1)} mm`);

  // the app's real registerView(), once with the original ICP and once with the plane ICP, same view, same imperfect pose
  const vd = { img: v2.img, disp: v2.disp, w: W, h: H }, RGo = { autoAlign: false };
  const poseErr = (p) => tErr(p, rel, nc.pos);
  const [rDep, tDep] = ms(() => RG.registerView(masterBuf, vd, guess, { ...RGo, icpMethod: "depth" })), [rOld, tOld] = ms(() => RG.registerView(masterBuf, vd, guess, { ...RGo, icpMethod: "point" })), [rNew, tNew] = ms(() => RG.registerView(masterBuf, vd, guess, { ...RGo, icpMethod: "plane" }));
  for (const [nm, r, tt] of [["registerView, original ICP ", rOld, tOld], ["registerView, plane ICP    ", rNew, tNew], ["registerView, depth ICP    ", rDep, tDep]]) {
    if (r.ok) { const e = poseErr(r.pose); console.log(`  ${nm}: ${e.rot.toFixed(3)} deg, ${e.mm.toFixed(2)} mm from truth | residual ${(r.icp.rmse * 1000).toFixed(1)} mm, match ${(r.icp.frac * 100).toFixed(0)}% | +${r.added} pts | ${tt} ms`); }
    else console.log(`  ${nm}: refused - ${r.reason} (${tt} ms)`);
  }
  ok("registerView with the joint depth ICP lands within 5 mm of the truth", rDep.ok && poseErr(rDep.pose).mm < 5 && poseErr(rDep.pose).rot < 0.2, rDep.ok ? `${poseErr(rDep.pose).mm.toFixed(2)} mm, fit a=${rDep.fit.a.toFixed(3)}` : rDep.reason);
  ok("registerView with the plane ICP refines the pose (not the scale)", rNew.ok && poseErr(rNew.pose).mm < g.mm * 0.5, rNew.ok ? "" : rNew.reason);
  if (rOld.ok && rNew.ok) ok("plane ICP is at least as accurate as the original here", poseErr(rNew.pose).mm <= poseErr(rOld.pose).mm * 1.25 + 0.3, `${poseErr(rNew.pose).mm.toFixed(2)} vs ${poseErr(rOld.pose).mm.toFixed(2)} mm`);
  const master13 = SP.parsePLY(masterBuf), probe = C.applyToPositions(guess, nc.pos), ic = C.icpPlane(master13, probe, RG.DEFAULTS.stages);
  ok("icpPlane keeps the icp() return shape", ic.ok && Array.isArray(ic.R) && ic.R.length === 9 && ic.t.length === 3 && typeof ic.rmse === "number" && typeof ic.frac === "number");

  // diagnosis: what is left of the registerView error, expressed in the camera frame, and does a scale-aware ICP remove it?
  if (rNew.ok) {
    const d = C.compose(C.invert(rel), rNew.pose), v = [d.t[0], d.t[1], d.t[2]];     // residual transform in camera-2 frame
    console.log(`  DIAG registerView residual: dt = [${v.map((x) => (x * 1000).toFixed(0)).join(", ")}] mm (camera x, y, z), dRot ${angDeg(d.R, [1, 0, 0, 0, 1, 0, 0, 0, 1]).toFixed(2)} deg, depth fit a=${rNew.fit.a.toFixed(3)} b=${rNew.fit.b.toFixed(3)} vs true a=${fit(v2).a.toFixed(3)} b=${fit(v2).b.toFixed(3)}`);
  }
  {
    const synth = RG.renderSynthetic(SP.parsePLY(masterBuf), guess, W, H), f = RG.fitDepthToMaster(v2.disp, synth.depth, W, H);
    const wrong = P.buildPLY(v2.img, v2.disp, f.ok ? { a: f.a, b: f.b } : null);
    const rs = C.registerAndMerge(masterBuf, wrong, { init: guess, estimateScale: true }), rr = C.registerAndMerge(masterBuf, wrong, { init: guess, estimateScale: false });
    for (const [nm, r] of [["rigid ", rr], ["scale ", rs]]) console.log(`  DIAG cloud ICP on the app-style depth fit, ${nm}: ` + (r.ok ? `${poseErr(r.T).mm.toFixed(1)} mm, ${angDeg(r.T.R, rel.R).toFixed(2)} deg, s=${r.T.s.toFixed(4)}` : "refused: " + r.reason));
  }

  for (const rounds of [2, 4, 6]) for (const m of ["point", "plane", "depth"]) {
    const r = RG.registerView(masterBuf, vd, guess, { ...RGo, icpMethod: m, rounds });
    console.log(`  ROUNDS ${rounds} ${m.padEnd(5)}: ` + (r.ok ? `${poseErr(r.pose).mm.toFixed(1)} mm, ${poseErr(r.pose).rot.toFixed(2)} deg, fit a=${r.fit.a.toFixed(3)}` : "refused " + r.reason));
  }
  {
    const rr = rng(2024), res = { point: [], plane: [], depth: [] }; let refused = { point: 0, plane: 0, depth: 0 };
    for (let k = 0; k < 10; k++) {
      const ax = [gauss(rr), gauss(rr), gauss(rr)], l = Math.hypot(...ax), deg = (1 + 4 * rr()) * RAD, gk = C.compose({ R: C.rotAxis(ax[0] / l * deg, ax[1] / l * deg, ax[2] / l * deg), t: [(rr() - 0.5) * 0.16, (rr() - 0.5) * 0.16, (rr() - 0.5) * 0.16], s: 1 }, rel);
      for (const m of ["point", "plane", "depth"]) { const r = RG.registerView(masterBuf, vd, gk, { ...RGo, icpMethod: m }); if (r.ok) res[m].push(poseErr(r.pose).mm); else refused[m]++; }
    }
    const med = (a) => a.length ? a.slice().sort((x, y) => x - y)[a.length >> 1].toFixed(1) : "-", mx = (a) => a.length ? Math.max(...a).toFixed(1) : "-";
    for (const m of ["point", "plane", "depth"]) console.log(`  SWEEP 10 random guesses (1-5 deg, up to 8 cm), ${m.padEnd(5)}: median ${med(res[m])} mm, worst ${mx(res[m])} mm, refused ${refused[m]}`);
    ok("depth ICP: all 10 random imperfect poses end within 5 mm", res.depth.length === 10 && Math.max(...res.depth) < 5, `worst ${mx(res.depth)} mm`);
  }
  // splat covariance: rotating + scaling a cloud must rotate/scale the covariance parsePLY derives
  {
    const r = rng(99), n = 40, rec = new Float32Array(n * 17);
    for (let i = 0; i < n; i++) { const o = i * 17, q = [gauss(r), gauss(r), gauss(r), gauss(r)], l = Math.hypot(...q); rec.set([r(), r(), r()], o); rec[o + 9] = 1; rec[o + 10] = Math.log(0.01 + 0.03 * r()); rec[o + 11] = Math.log(0.01 + 0.03 * r()); rec[o + 12] = Math.log(0.01 + 0.03 * r()); for (let c = 0; c < 4; c++) rec[o + 13 + c] = q[c] / l; }
    const cl = C.cloudFromRecords(rec), T = mk([1, 2, 3], 50, [0.1, 0.2, 0.3], 1.4), a0 = SP.parsePLY(C.writePLY(cl)), a1 = SP.parsePLY(C.writePLY(C.transformCloud(cl, T))), R = T.R, s2 = T.s * T.s;
    let worst = 0;
    for (let i = 0; i < n; i++) {
      const c = a0.data.subarray(i * 13 + 7, i * 13 + 13), S = [c[0], c[1], c[2], c[1], c[3], c[4], c[2], c[4], c[5]], E = mulRR(mulRR(R, S), tr(R)), d = a1.data.subarray(i * 13 + 7, i * 13 + 13);
      const w = [E[0], E[1], E[2], E[4], E[5], E[8]].map((v) => v * s2); w.forEach((v, k) => { worst = Math.max(worst, Math.abs(v - d[k]) / (Math.abs(v) + 1e-6)); });
    }
    ok("transformCloud rotates/scales splat covariance consistently with parsePLY", worst < 2e-3, `worst relative error ${worst.toExponential(1)}`);
  }

  // comparison overlay for the review screen, wrong pose vs right pose
  const [cmpBad, t1] = ms(() => C.compareForReview(masterBuf, newBuf, guess, { tol: 0.02 })), cmpGood = C.compareForReview(masterBuf, newBuf, rel, { tol: 0.02 });
  console.log(`  review overlay, imperfect pose: median ${(cmpBad.stats.median * 1000).toFixed(1)} mm, ${(cmpBad.stats.within[0.02] * 100).toFixed(0)}% within 20 mm, overlap ${(cmpBad.stats.overlap * 100).toFixed(0)}%  (${t1} ms)`);
  console.log(`  review overlay, true pose     : median ${(cmpGood.stats.median * 1000).toFixed(1)} mm, ${(cmpGood.stats.within[0.02] * 100).toFixed(0)}% within 20 mm`);
  ok("review stats separate a good pose from a bad one", cmpGood.stats.median < cmpBad.stats.median * 0.7 && cmpGood.stats.within[0.02] > cmpBad.stats.within[0.02]);
  const ov = SP.parsePLY(cmpBad.overlay); ok("overlay PLY loads with the app's parsePLY", ov.count > nc.n && ov.data.every(Number.isFinite));

  // full cloud-to-cloud pipeline from the same imperfect guess
  const [rm, tm] = ms(() => C.registerAndMerge(masterBuf, newBuf, { init: guess }));
  ok("registerAndMerge from a camera-pose guess", rm.ok, rm.ok ? "" : rm.reason);
  if (rm.ok) {
    const e = errOf(rm.T);
    console.log(`  registerAndMerge: ${e.rot.toFixed(3)} deg, ${e.mm.toFixed(2)} mm | before: median ${(rm.before.median * 1000).toFixed(1)} mm -> after ${(rm.after.median * 1000).toFixed(1)} mm | +${rm.added} pts, ${rm.covered} already covered | ${tm} ms`);
    ok("pose recovered to < 3 mm and the median distance drops", e.mm < 3 && rm.after.median < rm.before.median);
    const merged = SP.parsePLY(rm.merged); ok("merged PLY is readable by parsePLY and by appendPLY's 17-property check", merged.count === rm.total && merged.count > mc.n);
    const again = RG.appendPLY(masterBuf, new Float32Array(17), 1); ok("appendPLY still accepts the master layout", again.byteLength > masterBuf.byteLength);
  }
  // blind: no pose at all
  const [rb, tb] = ms(() => C.registerAndMerge(masterBuf, newBuf, { init: "pca" }));
  console.log(`  blind PCA start: ${rb.ok ? "registered, " + errOf(rb.T).mm.toFixed(1) + " mm, " + errOf(rb.T).rot.toFixed(2) + " deg" : "not registered (" + rb.reason + ")"}  (${tb} ms)`);
}

console.log(`\n${pass} passed, ${fail} failed`);
if (typeof process !== "undefined") process.exit(fail ? 1 : 0);
