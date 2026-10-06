# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only transport checks, runnable without importing CUDA/torch/vLLM.

The launcher's actual function body is compiled with test dependencies so that
these transport checks don't require a GPU installation. Full CLI tests remain
in tests/entrypoints/launchers/test_cli_args.py.
"""

import ast
import asyncio
import contextlib
import importlib.util
import json
import socket
import ssl
import subprocess
import types
from pathlib import Path

import httpx
import pytest
import uvicorn
from benchmark_connections import connection
from benchmark_connections import run as run_connections
from benchmark_inference import has_output, request

ROOT = Path(__file__).resolve().parents[3]


def test_overlay_rejects_frontend_drift(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "overlay", Path(__file__).with_name("install_overlay.py")
    )
    overlay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(overlay)
    # Check the known-additions guard against the committed base without copying
    # or importing the inference package.
    for relative in (
        "cli_args.py",
        "launcher.py",
        "api_server/entry.py",
        "render/entry.py",
    ):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(
            subprocess.check_output(
                ["git", "show", f"v0.29.0:vllm/entrypoints/launchers/{relative}"],
                cwd=ROOT,
            )
        )
    monkeypatch.setattr(
        overlay.importlib.metadata,
        "distribution",
        lambda _: types.SimpleNamespace(
            version="0.29.0", locate_file=lambda _: tmp_path
        ),
    )
    original_path = overlay.Path

    def source_path(*parts):
        if parts[0] == "/tmp/http2-overlay":
            filename = parts[1]
            mappings = {
                "api_entry.py": "api_server/entry.py",
                "render_entry.py": "render/entry.py",
            }
            return (
                ROOT / "vllm/entrypoints/launchers" / mappings.get(filename, filename)
            )
        return original_path(*parts)

    monkeypatch.setattr(overlay, "Path", source_path)
    overlay.main()
    assert "enable_http2" in (tmp_path / "launcher.py").read_text()
    with pytest.raises(RuntimeError, match="Base frontend drift"):
        overlay.main()


def test_token_detection_ignores_roles_and_keepalives():
    assert not has_output({"choices": [{"delta": {"role": "assistant"}}]})
    assert not has_output({"choices": []})
    assert has_output({"choices": [{"delta": {"content": "hello"}}]})
    assert has_output({"choices": [{"delta": {"reasoning_content": "think"}}]})
    assert has_output(
        {"choices": [{"delta": {"tool_calls": [{"function": {"arguments": "{"}}]}}]}
    )


@pytest.fixture
def certificate(tmp_path):
    cert, key = tmp_path / "cert.pem", tmp_path / "key.pem"
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-days",
            "1",
            "-keyout",
            str(key),
            "-out",
            str(cert),
            "-subj",
            "/CN=localhost",
            "-addext",
            "subjectAltName=DNS:localhost,IP:127.0.0.1",
        ],
        check=True,
        capture_output=True,
    )
    return cert, key


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
async def test_launcher_tls_negotiation_streaming_and_cancel(
    certificate, enabled, capsys
):
    cert, key = certificate
    clients, active = set(), set()
    peak_active = 0

    async def app(scope, receive, send):
        nonlocal peak_active
        if scope["type"] != "http":
            return
        while (await receive()).get("more_body", False):
            pass
        clients.add(scope["client"])
        task = asyncio.current_task()
        active.add(task)
        peak_active = max(peak_active, len(active))
        try:
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-type", b"text/event-stream")],
                }
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n',
                    "more_body": True,
                }
            )
            await asyncio.sleep(0.03)
            await send(
                {
                    "type": "http.response.body",
                    "body": b'data: {"choices":[{"delta":{"content":"token"}}]}\n\n',
                    "more_body": True,
                }
            )
            await asyncio.sleep(0.03)
            await send(
                {
                    "type": "http.response.body",
                    "body": b"data: [DONE]\n\n",
                    "more_body": False,
                }
            )
        finally:
            active.discard(task)

    class Application:
        routes = []
        state = types.SimpleNamespace(engine_client=None)

        async def __call__(self, scope, receive, send):
            await app(scope, receive, send)

    # Execute the actual shared launcher with a disabled engine watchdog, not a
    # reimplementation of its protocol-selection logic.
    source = ast.parse((ROOT / "vllm/entrypoints/launchers/launcher.py").read_text())
    function = next(
        node
        for node in source.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "serve_http"
    )
    function.returns = None
    for arg in (*function.args.args, *function.args.kwonlyargs):
        arg.annotation = None
    function.args.kwarg.annotation = None

    class Server(uvicorn.Server):
        @contextlib.contextmanager
        def capture_signals(self):
            yield

    async def watchdog(*_):
        await asyncio.Event().wait()

    namespace = dict(
        asyncio=asyncio,
        uvicorn=uvicorn,
        NoSignalServer=Server,
        watchdog_loop=watchdog,
        signal=__import__("signal"),
        logger=__import__("logging").getLogger("test"),
        H11_MAX_INCOMPLETE_EVENT_SIZE_DEFAULT=4194304,
        H11_MAX_HEADER_COUNT_DEFAULT=256,
        find_process_using_port=lambda _: None,
    )
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), "launcher.py", "exec"),
        namespace,
    )
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    application = Application()
    task = asyncio.create_task(
        namespace["serve_http"](
            application,
            sock=sock,
            enable_http2=enabled,
            host="127.0.0.1",
            port=port,
            ssl_certfile=str(cert),
            ssl_keyfile=str(key),
            http="zttp",
            lifespan="off",
            access_log=False,
            log_level="error",
        )
    )
    try:
        async with asyncio.timeout(5):
            while (
                not getattr(application.state, "server", None)
                or not application.state.server.started
            ):
                if task.done():
                    await task
                await asyncio.sleep(0.01)
        # Handshake measurements never enter the HTTP application or run a model.
        connection_args = types.SimpleNamespace(
            host="127.0.0.1",
            server_name="localhost",
            ca=str(cert),
            protocol="tcp",
            timeout=5,
        )
        address = ("127.0.0.1", port)
        tcp = await connection(connection_args, socket.AF_INET, address)
        assert not tcp["error"] and tcp["tcp_connect_ms"] >= 0
        assert tcp["tls_handshake_ms"] is None
        for protocol in ("h1", "h2"):
            connection_args.protocol = protocol
            row = await connection(connection_args, socket.AF_INET, address)
            if protocol == "h2" and not enabled:
                assert row["error"] == "RuntimeError"
                assert row["alpn"] != "h2"
            else:
                assert not row["error"], row
                assert row["tls_handshake_ms"] >= 0
                assert row["connect_and_handshake_ms"] >= (
                    row["tcp_connect_ms"] + row["tls_handshake_ms"]
                )
                expected = (
                    ("h2" if protocol == "h2" else "http/1.1") if enabled else None
                )
                assert row["alpn"] == expected
        connection_args.protocol = "h1"
        await run_connections(
            types.SimpleNamespace(
                **vars(connection_args), port=port, samples=3, concurrency=2
            )
        )
        report = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
        assert report[0]["http_requests"] == 0
        assert report[-1]["success"] == 3 and report[-1]["errors"] == 0
        assert report[-1]["tls_handshake_ms"]["p95"] >= 0
        connection_args.ca = None
        rejected = await connection(connection_args, socket.AF_INET, address)
        assert rejected["error"] == "SSLCertVerificationError"
        assert not clients and not active
        args = types.SimpleNamespace(
            url=f"https://127.0.0.1:{port}/v1/chat/completions",
            protocol="h2" if enabled else "h1",
            cancel_after=0,
            timeout=5,
        )
        context = ssl.create_default_context(cafile=str(cert))
        async with httpx.AsyncClient(
            verify=context,
            http1=not enabled,
            http2=enabled,
            trust_env=False,
            limits=httpx.Limits(max_connections=1),
        ) as client:
            rows = await asyncio.gather(
                *(request(client, args, {"stream": True}) for _ in range(4))
            )
            assert all(row["done"] and not row["error"] for row in rows), rows
            assert all(row["ttft_ms"] > row["first_body_byte_ms"] for row in rows)
            assert len(clients) == 1
            assert peak_active == (4 if enabled else 1)
            cancel_args = types.SimpleNamespace(**vars(args))
            cancel_args.cancel_after = 1
            cancelled, sibling = await asyncio.gather(
                request(client, cancel_args, {"stream": True}),
                request(client, args, {"stream": True}),
            )
            assert cancelled["cancelled"] and not cancelled["error"]
            assert sibling["done"] and not sibling["error"]
        # HTTP/1.1 remains usable with HTTP/2 enabled, on the same TLS listener.
        args.protocol = "h1"
        async with httpx.AsyncClient(
            verify=context, http2=False, trust_env=False
        ) as client:
            assert (await request(client, args, {"stream": True}))["done"]
        async with httpx.AsyncClient(trust_env=False) as client:
            untrusted = await request(client, args, {"stream": True})
            assert untrusted["error"] == "ConnectError"
        async with asyncio.timeout(5):
            while active:
                await asyncio.sleep(0.01)
    finally:
        if getattr(application.state, "server", None):
            application.state.server.should_exit = True
        shutdown = await asyncio.wait_for(task, 5)
        await shutdown
        sock.close()
