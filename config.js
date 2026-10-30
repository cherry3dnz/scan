// All processing happens in the visitor's browser. No server, no API keys.
window.APP_CONFIG = {
  MODEL_ID: "onnx-community/depth-anything-v2-small", // Apache-2.0, ~100 MB first download, then cached
  MODEL_DTYPE: null,       // null = library default (fp32). Try "q8" or "fp16" if that file exists for the model
  MAX_SIDE: 640,           // working resolution (long side, px)
  POINT_STRIDE: 2,         // 2 => one Gaussian per 2x2 pixels; 1 => 4x denser but slower
  HFOV_DEG: 65,            // horizontal field of view assumed when CALIBRATION is null
  // Optional lens calibration from tools/calibrate.py (paste the printed block here), e.g.
  // CALIBRATION: { width:1280, height:720, fx:1000, fy:1000, cx:640, cy:360, k1:-0.1, k2:0.01, p1:0, p2:0, k3:0 }
  CALIBRATION: null,
  // Reference objects offered in the metric-scale panel. mm: null = the user types the length.
  // Check coin sizes against your own coin; the numbers below are typical values.
  REFERENCE_PRESETS: [
    { label: "Credit card, long edge (85.60 mm)", mm: 85.6 },
    { label: "A4 sheet, long edge (297 mm)", mm: 297 },
    { label: "A4 sheet, short edge (210 mm)", mm: 210 },
    { label: "Coin: US quarter (24.26 mm)", mm: 24.26 },
    { label: "Coin: 1 euro (23.25 mm)", mm: 23.25 },
    { label: "Coin: NZ$1 (23.00 mm)", mm: 23.0 },
    { label: "Coin: other (type diameter)", mm: null },
    { label: "Tape measurement (type length)", mm: null },
  ],
};
