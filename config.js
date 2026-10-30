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
};
