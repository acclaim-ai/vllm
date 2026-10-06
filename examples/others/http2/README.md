# Experimental Uvicorn HTTP/2 on vLLM 0.29

This branch retains Uvicorn/FastAPI and adds opt-in HTTP/2 using
`uvicorn==0.54.0` and `zttp==0.0.34`. No Hypercorn, proxy hop, engine rebuild,
model argument change or compile-cache patch is needed. HTTP/2 is experimental:
this is a test image, not a production security or performance recommendation.

The existing upstream proposal
[vllm-project/vllm#44186](https://github.com/vllm-project/vllm/pull/44186)
uses Hypercorn; this experiment uses
[Uvicorn's native HTTP/2 support](https://uvicorn.dev/concepts/http2/).

## Image and certificate

Build on an **existing digest-pinned cached 0.29 image**, not a mutable tag:

```sh
docker build -f docker/Dockerfile.http2-overlay \
  --build-arg VLLM_CACHED_IMAGE="$CACHED_IMAGE_WITH_DIGEST" \
  -t vllm-http2-experiment .
```

The overlay installs only the two pinned frontend packages (`--no-deps`) and
copies four frontend files after checking that their originals match v0.29.0.
Cache files, inference dependencies, inherited environment and cache labels
are unchanged. A runtime cache HIT still needs verification on the actual GPU;
the unchanged cache contract alone is not proof of compatibility.
The registry's schema-6 identity includes the full serve argv and startup
prelude. Adding TLS/parser flags changes the lookup identity even though engine
settings are unchanged. Establish a matching, validated cache identity before
the GPU comparison; do not bypass restore validation to reuse an old bundle.

Generate a short-lived test certificate **outside the image and repository**.
Replace the example IP with the backend's actual reachable IP before generating:

```sh
openssl req -x509 -newkey rsa:2048 -nodes -days 2 \
  -keyout /secure/test-key.pem -out /secure/test-cert.pem \
  -subj /CN=backend.test \
  -addext 'subjectAltName=DNS:backend.test,IP:203.0.113.10'
```

Mount these files into the backend and append to its unchanged model arguments:

```sh
--enable-http2 --ssl-certfile /tls/test-cert.pem --ssl-keyfile /tls/test-key.pem
```

HTTP/1.1 and HTTP/2 then share the same TLS listener, selected by client ALPN.
Trust the certificate explicitly on the test client; never use `verify=False`
or bake private keys into a Docker layer. This is server-authenticated TLS,
**not mTLS or SPIFFE workload authorization**. Restrict experiment access:
vLLM's API key does not protect every administrative endpoint.

## Three separate experiments

Keep these results separate:

1. **Connection-only benchmark:** TCP connect and TLS handshake/ALPN. No HTTP
   requests, model payloads, inference, TTFT or streaming measurements.
2. **CPU protocol checks:** mock SSE tests that validate the implementation.
   Their timings are assertions, not performance results.
3. **Real-image inference benchmark:** representative model requests measuring
   headers/TTFB, TTFT, streaming and concurrency on the actual GPU image.

## Connection-only benchmark

This stdlib-only tool requires no running model: use a dedicated TLS listener
with the same Uvicorn/parser configuration, or the real image's listener without
sending inference requests. Run from the same real client pod planned for GPU
tests. Compare H1/H2/H2/H1 blocks and plaintext TCP separately:

```sh
.venv/bin/python examples/others/http2/benchmark_connections.py \
  --host 203.0.113.10 --port 8000 --server-name backend.test \
  --ca /secure/test-cert.pem --protocol h2 --samples 100 --concurrency 1
```

Repeat with `--protocol h1`; use `--protocol tcp` on the plaintext port for the
TCP-only baseline. Each sample opens a new connection and a new TLS context;
DNS resolution, local certificate loading and connection teardown are excluded
from measured durations. DNS is resolved once and the first address is used for
every sample (no DNS/Happy Eyeballs benchmark). JSONL records TCP, TLS and combined
setup times, TLS version, ALPN and errors; the summary reports successful-sample
nearest-rank p50/p95/p99 and error counts. Failed handshakes are not latency
successes. An H1 listener without ALPN is recorded as `null`; H2 fallback fails.

Do not interpret this as request latency or an H2 multiplexing benchmark. It
does not send even an HTTP health request or measure connection reuse, resumed
TLS sessions, engine queuing or inference throughput. Handshake concurrency is
not model-request concurrency. Repeat with bounded concurrency to assess setup
under connection churn; record client/server CPU and endpoint location alongside
the results. No connection-only performance results have been collected yet.

## Real-image inference comparison

Use one fixed GPU, model, cache bundle and exact inference argv. Stop competing
traffic, verify cache HIT/readiness and warm the model before every phase.
Do not join the experiment to a shared production routing pool.

| Phase | Server | Client | What it isolates |
| --- | --- | --- | --- |
| A | Existing cached image, plaintext HTTP/1.1 | HTTP/1.1 | Deployed baseline |
| B | Overlay, default parser, plaintext HTTP/1.1 | HTTP/1.1 | Uvicorn upgrade overhead |
| C | Overlay, TLS, `--uvicorn-http-protocol zttp` | HTTPS/1.1 | Parser/TLS control |
| D | Overlay, TLS, `--enable-http2` | HTTPS/1.1 and HTTPS/2, interleaved | Primary same-server protocol comparison |

For D, run H1/H2/H2/H1 blocks repeatedly from the same real client pod, not from
inside the engine. Force the negotiated protocol; the probe treats fallback as
an error. Compare both fresh connections and warmed pools. H1 needs a connection
per simultaneous response; H2 multiplexes streams. Test both comparable pool
capacity and a one-connection stress case, reporting connection/pool limits.

Use identical representative payloads (short, normal, long context; 1/128/512
output tokens), deterministic sampling where supported, and matching prefix-cache
state. Include `stream: true` and `stream_options: {"include_usage": true}`.
Do not change request seeds or cache salts between paired runs. Repeat at
concurrency 1, 4, 8, 16, 32; increase further only if engine latency stays usable.
Run separate `--cancel-after 1` and slow-reader tests and verify engine running/
waiting counts return to baseline. Also test certificate rejection and restart.

### Inference probe

Use `uv venv --python 3.12`, then `uv pip install 'httpx[http2]==0.28.1'`.
Set `VLLM_TEST_API_KEY` through the local secret mechanism if required; do not
put credentials or customer payloads into this public fork.

```sh
.venv/bin/python examples/others/http2/benchmark_inference.py \
  --url https://203.0.113.10:8000/v1/chat/completions \
  --ca /secure/test-cert.pem --payload /secure/payload.json \
  --protocol h2 --concurrency 8 --connections 1 --requests 100
```

Repeat with `--protocol h1 --connections 8`; use `--fresh` for fresh-client
connections. The JSONL records negotiated version, TCP/TLS timing when a new
connection is made, response-header time, first response-body byte, first
output-bearing SSE delta (TTFT), E2E duration, output-event gaps, usage tokens,
errors, intentional cancellations and successful requests/s.
These connection timings describe the inference run, including fresh/reused
pool behavior; they do not replace the independent handshake-only dataset.

Role-only deltas/keepalives are not TTFT; SSE events/chunks are not individual
tokens. Event gaps are **not token-level ITL**, and server headers are not first
generated output. Collect engine queue/latency metrics and host CPU/RSS alongside
the probe. This is closed-loop load: it includes pool waiting but does not measure
open-loop overload behavior. A faster local mock is not evidence of a GPU benefit.

Record the base/overlay digests, exact packages/argv, GPU/region, payload hash,
client location, connection mode, p50/p95/p99, all errors and repeated paired
results. Test the real router hop separately only after configuring explicit
certificate trust and verifying backend ALPN; client-to-router HTTP/2 does not
prove router-to-engine HTTP/2.

## CPU-only validation

```sh
uv pip install uvicorn==0.54.0 zttp==0.0.34 \
  'httpx[http2]==0.28.1' pytest pytest-asyncio
.venv/bin/python -m pytest examples/others/http2/test_http2.py \
  --confcutdir=examples/others/http2 -q
```

These checks execute the shared launcher's function with a mock engine watchdog,
using real Uvicorn/TLS sockets. They verify verified/self-signed TLS, H1 keepalive,
H2 multiplexing, H1 fallback, streaming token detection and cancellation isolation.
They do not exercise the full vLLM import graph, CLI, GPU, model or deployment.
Run the existing launcher CLI tests in the full image before GPU testing.
The H11-specific header-limit settings do not establish equivalent limits for
the experimental parser. Keep this listener restricted until those are audited.
