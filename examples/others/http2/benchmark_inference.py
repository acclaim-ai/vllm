# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real-inference HTTP/1.1 vs HTTP/2 streaming probe; writes JSONL to stdout."""

import argparse
import asyncio
import hashlib
import json
import os
import ssl
import time
from pathlib import Path

import httpx


def has_output(event):
    for choice in event.get("choices", []):
        delta = choice.get("delta", {})
        if any(delta.get(key) for key in ("content", "reasoning_content", "reasoning")):
            return True
        if any(
            call.get("function", {}).get("arguments")
            for call in delta.get("tool_calls", [])
        ):
            return True
        if choice.get("text"):
            return True
    return False


async def request(client, args, payload):
    started = time.perf_counter()
    row = {
        "type": "request",
        "http_version": None,
        "status": None,
        "headers_ms": None,
        "first_body_byte_ms": None,
        "ttft_ms": None,
        "output_events": 0,
        "event_gaps_ms": [],
        "completion_tokens": None,
        "done": False,
        "cancelled": False,
        "error": None,
    }
    row.update(tcp_connect_ms=None, tls_handshake_ms=None)
    last_output = None
    data_lines = []
    trace_started = {}

    async def trace(name, info):
        phase, _, state = name.rpartition(".")
        if state == "started":
            trace_started[phase] = time.perf_counter()
        elif state == "complete" and phase in trace_started:
            key = {
                "connection.connect_tcp": "tcp_connect_ms",
                "connection.start_tls": "tls_handshake_ms",
            }.get(phase)
            if key:
                row[key] = (time.perf_counter() - trace_started[phase]) * 1000

    try:
        async with (
            asyncio.timeout(args.timeout),
            client.stream(
                "POST", args.url, json=payload, extensions={"trace": trace}
            ) as response,
        ):
            row["headers_ms"] = (time.perf_counter() - started) * 1000
            row["http_version"] = response.http_version
            row["status"] = response.status_code
            expected = "HTTP/2" if args.protocol == "h2" else "HTTP/1.1"
            if response.http_version != expected:
                raise RuntimeError(f"Expected {expected}, got {response.http_version}")
            response.raise_for_status()

            # aiter_lines handles split/coalesced frames, UTF-8 and CRLF. First
            # body byte is measured independently from complete SSE events.
            async def timed_bytes():
                async for chunk in response.aiter_bytes():
                    if chunk and row["first_body_byte_ms"] is None:
                        row["first_body_byte_ms"] = (
                            time.perf_counter() - started
                        ) * 1000
                    yield chunk

            # Use HTTPX's public decoding path with the original streaming body.
            decoded = httpx.Response(200, stream=ByteStream(timed_bytes()))
            async for line in decoded.aiter_lines():
                if line.startswith("data:"):
                    data_lines.append(line[5:].lstrip(" "))
                elif not line and data_lines:
                    data = "\n".join(data_lines)
                    data_lines.clear()
                    if data == "[DONE]":
                        row["done"] = True
                        continue
                    event = json.loads(data)
                    if event.get("error"):
                        raise RuntimeError("Backend emitted an SSE error event")
                    if event.get("usage"):
                        row["completion_tokens"] = event["usage"].get(
                            "completion_tokens"
                        )
                    if has_output(event):
                        now = time.perf_counter()
                        if row["ttft_ms"] is None:
                            row["ttft_ms"] = (now - started) * 1000
                        if last_output is not None:
                            row["event_gaps_ms"].append((now - last_output) * 1000)
                        last_output = now
                        row["output_events"] += 1
                        if (
                            args.cancel_after
                            and row["output_events"] >= args.cancel_after
                        ):
                            row["cancelled"] = True
                            break
            if not row["done"] and not row["cancelled"]:
                raise RuntimeError("SSE stream ended without DONE")
    except Exception as exc:
        # Do not print server bodies, prompts, credentials or sensitive URLs.
        row["error"] = type(exc).__name__
    row["e2e_ms"] = (time.perf_counter() - started) * 1000
    return row


class ByteStream(httpx.AsyncByteStream):
    def __init__(self, iterator):
        self.iterator = iterator

    async def __aiter__(self):
        async for chunk in self.iterator:
            yield chunk


async def run(args):
    payload_bytes = Path(args.payload).read_bytes()
    payload = json.loads(payload_bytes)
    if payload.get("stream") is not True:
        raise ValueError("Payload must explicitly set stream=true")
    if args.protocol == "h2" and not args.url.startswith("https://"):
        raise ValueError("This probe tests h2 over TLS, not h2c")
    verify = ssl.create_default_context(cafile=args.ca)
    headers = {}
    if token := os.environ.get("VLLM_TEST_API_KEY"):
        headers["Authorization"] = f"Bearer {token}"
    connections = args.connections or args.concurrency

    def client():
        return httpx.AsyncClient(
            http1=args.protocol == "h1",
            http2=args.protocol == "h2",
            verify=verify,
            trust_env=False,
            headers=headers,
            timeout=httpx.Timeout(args.timeout),
            limits=httpx.Limits(
                max_connections=connections, max_keepalive_connections=connections
            ),
        )

    print(
        json.dumps(
            {
                "type": "config",
                "benchmark": "streaming_inference",
                "protocol": args.protocol,
                "concurrency": args.concurrency,
                "connections": connections,
                "fresh": args.fresh,
                "requests": args.requests,
                "payload_sha256": hashlib.sha256(payload_bytes).hexdigest(),
            }
        )
    )
    async with client() as pooled:
        if not args.fresh:
            for _ in range(args.warmup):
                warm = await request(pooled, args, payload)
                if warm["error"]:
                    raise RuntimeError(f"Warmup failed: {warm['error']}")
        sem = asyncio.Semaphore(args.concurrency)

        async def sample(index):
            async with sem:
                if args.fresh:
                    async with client() as fresh:
                        row = await request(fresh, args, payload)
                else:
                    row = await request(pooled, args, payload)
                row["index"] = index
                print(json.dumps(row), flush=True)
                return row

        started = time.perf_counter()
        rows = await asyncio.gather(*(sample(i) for i in range(args.requests)))
        seconds = time.perf_counter() - started
    successful = [row for row in rows if row["done"] and not row["error"]]
    print(
        json.dumps(
            {
                "type": "summary",
                "elapsed_s": seconds,
                "success": len(successful),
                "errors": sum(bool(row["error"]) for row in rows),
                "cancelled": sum(row["cancelled"] for row in rows),
                "successful_rps": len(successful) / seconds,
                "completion_tokens": sum(
                    row["completion_tokens"] or 0 for row in successful
                ),
            }
        )
    )
    if any(row["error"] for row in rows):
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--payload", required=True)
    parser.add_argument("--protocol", choices=["h1", "h2"], required=True)
    parser.add_argument("--ca", help="Trust this experiment certificate/CA")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--connections", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--cancel-after", type=int, default=0)
    args = parser.parse_args()
    if min(args.requests, args.concurrency, args.timeout) <= 0:
        parser.error("requests, concurrency and timeout must be positive")
    if min(args.connections, args.warmup, args.cancel_after) < 0:
        parser.error("connections, warmup and cancel-after cannot be negative")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
