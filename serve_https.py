#!/usr/bin/env python3
"""HTTPS test server for trying the app on a phone over your Wi-Fi (phones need HTTPS for the camera).
Usage: python3 serve_https.py [port]      then open https://<this computer's LAN IP>:<port> on the phone
and accept the self-signed certificate warning once. Needs openssl on PATH."""
import http.server, os, socket, ssl, subprocess, sys
port = int(sys.argv[1]) if len(sys.argv) > 1 else 8443
here = os.path.dirname(os.path.abspath(__file__)); os.chdir(here)
cert, key = "dev-cert.pem", "dev-key.pem"
if not os.path.exists(cert):
    subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "30", "-subj", "/CN=syn3d-dev",
                    "-keyout", key, "-out", cert], check=True, stderr=subprocess.DEVNULL)
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(("10.255.255.255", 1)); ip = s.getsockname()[0]
except Exception:
    ip = "<your-computer-ip>"
ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER); ctx.load_cert_chain(cert, key)
srv = http.server.ThreadingHTTPServer(("0.0.0.0", port), http.server.SimpleHTTPRequestHandler)
srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
print(f"Serving https://{ip}:{port}  (Ctrl+C to stop)"); srv.serve_forever()
