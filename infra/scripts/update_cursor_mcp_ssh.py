#!/usr/bin/env python3
"""Set --host= in .cursor/mcp.json for tradan SSH MCP servers after Terraform apply."""

from __future__ import annotations

import json
import sys
from pathlib import Path

MCP_SERVERS = {
    "base": "tradan-base-ssh-mcp",
    "train": "tradan-training-ssh-mcp",
}


def _patch_host(args: list, new_ip: str) -> list:
    prefix = "--host="
    out: list = []
    found = False
    for a in args:
        if isinstance(a, str) and a.startswith(prefix):
            out.append(prefix + new_ip)
            found = True
        else:
            out.append(a)
    if not found:
        raise ValueError("no --host= in args")
    return out


def main() -> None:
    if len(sys.argv) != 3:
        print("usage: update_cursor_mcp_ssh.py {base|train} IP", file=sys.stderr)
        sys.exit(2)
    which, ip = sys.argv[1], sys.argv[2].strip()
    if which not in MCP_SERVERS:
        print("usage: update_cursor_mcp_ssh.py {base|train} IP", file=sys.stderr)
        sys.exit(2)
    if not ip:
        print("update_cursor_mcp_ssh: empty IP, skipping", file=sys.stderr)
        return

    repo_root = Path(__file__).resolve().parent.parent.parent
    mcp_path = repo_root / ".cursor" / "mcp.json"
    if not mcp_path.is_file():
        print(f"update_cursor_mcp_ssh: not found: {mcp_path}", file=sys.stderr)
        sys.exit(1)

    data = json.loads(mcp_path.read_text())
    key = MCP_SERVERS[which]
    server = data.get("mcpServers", {}).get(key)
    if not server:
        print(f"update_cursor_mcp_ssh: missing mcpServers.{key}", file=sys.stderr)
        sys.exit(1)
    args = server.get("args")
    if not isinstance(args, list):
        print(f"update_cursor_mcp_ssh: {key}.args is not a list", file=sys.stderr)
        sys.exit(1)
    server["args"] = _patch_host(args, ip)
    mcp_path.write_text(json.dumps(data, indent=4) + "\n")
    print(f"Updated Cursor MCP {key} --host={ip} ({mcp_path})")


if __name__ == "__main__":
    main()
