"""Optional, run once on your computer: calibrate YOUR camera with a printed chessboard.
Usage: pip install opencv-python numpy && python calibrate.py ./chessboard_photos 9 6
(9 x 6 = inner corners). Use 15-25 photos from different angles, same camera + resolution
as the app. Paste the printed block into frontend/config.js as CALIBRATION."""
import glob, sys
import cv2, numpy as np

folder, cols, rows = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
objp = np.zeros((rows * cols, 3), np.float32)
objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
objpts, imgpts, size = [], [], None
for p in sorted(glob.glob(f"{folder}/*")):
    g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if g is None:
        continue
    size = g.shape[::-1]
    ok, c = cv2.findChessboardCorners(g, (cols, rows))
    if ok:
        c = cv2.cornerSubPix(g, c, (11, 11), (-1, -1),
                             (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3))
        objpts.append(objp); imgpts.append(c)
print(f"// {len(objpts)} usable images")
rms, K, d, _, _ = cv2.calibrateCamera(objpts, imgpts, size, None, None)
d = d.ravel().tolist() + [0] * 5
print(f"// RMS reprojection error: {rms:.3f} px (aim for < 1.0)")
print(f"CALIBRATION: {{ width:{size[0]}, height:{size[1]}, fx:{K[0,0]:.2f}, fy:{K[1,1]:.2f}, "
      f"cx:{K[0,2]:.2f}, cy:{K[1,2]:.2f}, k1:{d[0]:.5f}, k2:{d[1]:.5f}, p1:{d[2]:.5f}, p2:{d[3]:.5f}, k3:{d[4]:.5f} }},")
