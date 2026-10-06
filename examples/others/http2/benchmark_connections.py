# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fresh TCP/TLS handshake benchmark. Sends no HTTP or inference requests."""

import argparse
import asyncio
import json
import math
import socket
import ssl
import time


async def connection(args, family, address):
    context = None
    if args.protocol != "tcp":
        # A new context per sample prevents TLS session reuse. Context setup is
        # excluded from the measured connection/handshake durations.
        context = ssl.create_default_context(cafile=args.ca)
        context.set_alpn_protocols(["h2" if args.protocol == "h2" else "http/1.1"])
    sock = socket.socket(family, socket.SOCK_STREAM)
    sock.setblocking(False)
    writer = None
    row = dict(
        type="connection",
        tcp_connect_ms=None,
        tls_handshake_ms=None,
        connect_and_handshake_ms=None,
        alpn=None,
        tls_version=None,
        error=None,
    )
    started = time.perf_counter()
    try:
        async with asyncio.timeout(args.timeout):
            await asyncio.get_running_loop().sock_connect(sock, address)
            row["tcp_connect_ms"] = (time.perf_counter() - started) * 1000
            _, writer = await asyncio.open_connection(sock=sock)
            if context:
                tls_started = time.perf_counter()
                await writer.start_tls(
                    context,
                    server_hostname=args.server_name or args.host,
                    ssl_handshake_timeout=args.timeout,
                )
                row["tls_handshake_ms"] = (time.perf_counter() - tls_started) * 1000
                tls = writer.get_extra_info("ssl_object")
                row.update(alpn=tls.selected_alpn_protocol(), tls_version=tls.version())
                expected = "h2" if args.protocol == "h2" else "http/1.1"
                if row["alpn"] != expected and not (
                    args.protocol == "h1" and row["alpn"] is None
                ):
                    raise RuntimeError("Unexpected ALPN; protocol fallback rejected")
        row["connect_and_handshake_ms"] = (time.perf_counter() - started) * 1000
    except Exception as exc:
        row["error"] = type(exc).__name__
    finally:
        if writer:
            writer.close()
            try:
                async with asyncio.timeout(args.timeout):
                    await writer.wait_closed()
            except (OSError, TimeoutError):
                pass
        else:
            sock.close()
    return row


async def run(args):
    # Resolve once, outside sample timing, and use one destination throughout.
    addresses = await asyncio.get_running_loop().getaddrinfo(
        args.host, args.port, type=socket.SOCK_STREAM
    )
    family, _, _, _, address = addresses[0]
    print(
        json.dumps(
            {
                "type": "config",
                "benchmark": "fresh_handshake_only",
                "protocol": args.protocol,
                "host": args.host,
                "address": address,
                "server_name": args.server_name or args.host,
                "samples": args.samples,
                "concurrency": args.concurrency,
                "dns_included": False,
                "tls_session_reuse": False,
                "http_requests": 0,
            }
        )
    )
    sem = asyncio.Semaphore(args.concurrency)

    async def sample(index):
        async with sem:
            row = await connection(args, family, address)
            row["index"] = index
            print(json.dumps(row), flush=True)
            return row

    rows = await asyncio.gather(*(sample(i) for i in range(args.samples)))
    successful = [row for row in rows if not row["error"]]
    summary = dict(
        type="summary", success=len(successful), errors=len(rows) - len(successful)
    )
    for metric in ("tcp_connect_ms", "tls_handshake_ms", "connect_and_handshake_ms"):
        values = sorted(row[metric] for row in successful if row[metric] is not None)
        summary[metric] = {
            f"p{percentile}": values[math.ceil(len(values) * percentile / 100) - 1]
            if values
            else None
            for percentile in (50, 95, 99)
        }
    print(json.dumps(summary))
    if summary["errors"]:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--protocol", choices=["tcp", "h1", "h2"], required=True)
    parser.add_argument(
        "--server-name", help="TLS certificate identity/SNI, if not host"
    )
    parser.add_argument("--ca", help="Trust this experiment certificate/CA")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=10)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    if min(args.samples, args.concurrency, args.timeout) <= 0:
        parser.error("samples, concurrency and timeout must be positive")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
