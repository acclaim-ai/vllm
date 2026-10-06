# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Install frontend files only; fail if the base differs from vLLM 0.29.0."""

import importlib.metadata
import py_compile
import shutil
from pathlib import Path


def main():
    dist = importlib.metadata.distribution("vllm")
    if dist.version != "0.29.0":
        raise RuntimeError(f"Expected vLLM 0.29.0, found {dist.version}")
    root = Path(dist.locate_file("vllm/entrypoints/launchers"))
    sources = {
        "cli_args.py": "cli_args.py",
        "launcher.py": "launcher.py",
        "api_entry.py": "api_server/entry.py",
        "render_entry.py": "render/entry.py",
    }
    # Only the known additions may differ: never overwrite existing cache patches.
    for source, target in sources.items():
        candidate = Path("/tmp/http2-overlay", source).read_text()
        original = (root / target).read_text()
        if source == "cli_args.py":
            start = candidate.index("    uvicorn_http_protocol:")
            end = candidate.index("    disable_access_log_for_endpoints:", start)
            candidate_base = candidate[:start] + candidate[end:]
        elif source == "launcher.py":
            candidate_base = candidate.replace("    enable_http2: bool = False,\n", "")
            start = candidate_base.index("    if enable_http2:\n")
            end = candidate_base.index("    config = uvicorn.Config", start)
            candidate_base = candidate_base[:start] + candidate_base[end:]
        else:
            candidate_base = candidate.replace(
                "        enable_http2=args.enable_http2,\n", ""
            )
            candidate_base = candidate_base.replace(
                "        http=args.uvicorn_http_protocol,\n", ""
            )
        if candidate_base != original:
            raise RuntimeError(f"Base frontend drift in {target}; refusing overwrite")
    for source, target in sources.items():
        shutil.copyfile(Path("/tmp/http2-overlay", source), root / target)
        py_compile.compile(str(root / target), doraise=True)


if __name__ == "__main__":
    main()
