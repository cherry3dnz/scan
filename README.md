# Photo -> Depth -> 3D Gaussian Splat (100% in the browser, free)

Static site for GitHub Pages. Camera capture -> lens undistortion -> Depth Anything V2 Small (Transformers.js,
WebGPU or WASM) -> 3DGS-layout binary .ply -> WebGL2 Gaussian splat renderer that spins about the vertical axis.

Local test:   cd frontend && python -m http.server 8080   (open http://localhost:8080)
Deploy:       push to GitHub, Settings -> Pages -> Source: GitHub Actions. See the PDF guide.
