import json
import os
import re
import sys
import time
import urllib.request
import urllib.error
import base64
from http.server import BaseHTTPRequestHandler, HTTPServer

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    sys.exit("HF_TOKEN environment variable is not set (see kuni-proxy.service)")
HF_MODEL = "black-forest-labs/FLUX.1-schnell"
HF_URL = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL}"

# FLUX doesn't understand Stable-Diffusion-style emphasis syntax like "(term:1.2)" - strip the
# ":weight)" part so it reads as a plain phrase instead of confusing literal punctuation.
_SD_WEIGHT_RE = re.compile(r"\(([^()]+):[0-9.]+\)")


def _clean_prompt(prompt: str) -> str:
    prev = None
    while prev != prompt:
        prev = prompt
        prompt = _SD_WEIGHT_RE.sub(r"\1", prompt)
    return prompt


class SDProxy(BaseHTTPRequestHandler):
    def _read_body(self):
        # kuni's HTTP client (libcurl) sends POST bodies with "Transfer-Encoding: chunked" and no
        # Content-Length header. http.server doesn't dechunk automatically, so reading by
        # Content-Length alone (which defaults to 0) silently returns an empty body - decode that
        # manually instead.
        if self.headers.get('Transfer-Encoding', '').lower() == 'chunked':
            body = b''
            while True:
                size_line = self.rfile.readline().strip()
                chunk_size = int(size_line.split(b';')[0], 16)
                if chunk_size == 0:
                    self.rfile.readline()  # trailing CRLF after last chunk
                    break
                body += self.rfile.read(chunk_size)
                self.rfile.readline()  # CRLF after chunk data
            return body
        content_length = int(self.headers.get('Content-Length', 0))
        return self.rfile.read(content_length)

    def do_POST(self):
        if self.path == "/sdapi/v1/txt2img":
            post_data = self._read_body()
            req_json = json.loads(post_data.decode('utf-8'))
            prompt = _clean_prompt(req_json.get("prompt", "a beautiful anime girl"))

            # round to nearest multiple of 16 (required by most diffusion models); kuni sends random
            # values in [768, 1400].
            width = max(256, min(1536, int(req_json.get("width", 1024)) // 16 * 16))
            height = max(256, min(1536, int(req_json.get("height", 1024)) // 16 * 16))

            print(f"[Proxy] Generating image ({width}x{height}) for prompt: {prompt}")

            payload = json.dumps({
                "inputs": prompt,
                "parameters": {
                    "width": width,
                    "height": height,
                    "num_inference_steps": 4,  # FLUX.1-schnell is distilled for 1-4 steps
                },
            }).encode("utf-8")

            req = urllib.request.Request(
                HF_URL,
                data=payload,
                method="POST",
                headers={
                    "Authorization": f"Bearer {HF_TOKEN}",
                    "Content-Type": "application/json",
                },
            )

            try:
                # The model may need to cold-start on first use ("loading"); retry a few times.
                for attempt in range(3):
                    try:
                        with urllib.request.urlopen(req, timeout=60) as response:
                            image_data = response.read()
                        break
                    except urllib.error.HTTPError as e:
                        body = e.read()
                        if e.code == 503 and attempt < 2:
                            print(f"[Proxy] Model loading, retrying... ({body[:200]})")
                            time.sleep(5)
                            continue
                        raise Exception(f"HuggingFace error {e.code}: {body.decode('utf-8', 'replace')}")

                b64_image = base64.b64encode(image_data).decode('utf-8')

                res_data = {
                    "images": [b64_image],
                    "info": "{}"
                }

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(res_data).encode('utf-8'))
                print("[Proxy] Image successfully returned to Kuni.")
            except Exception as e:
                print(f"[Proxy] Error: {e}")
                self.send_response(500)
                self.end_headers()
                self.wfile.write(str(e).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

if __name__ == "__main__":
    server = HTTPServer(('127.0.0.1', 7860), SDProxy)
    print("Proxy running on http://127.0.0.1:7860/...")
    server.serve_forever()
