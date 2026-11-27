#!/usr/bin/env python3
"""Read the JSON lines of `cargo ... --message-format=json-render-diagnostics`.

Cargo's layout of `target/` has changed before and will again; its own messages say where an
artifact is and what a build script set, so nothing here guesses a path.

    cargo_artifacts.py FILE exe NAME          executable path of the bench target NAME
    cargo_artifacts.py FILE pid NAME          package_id of the package that owns bench target NAME
    cargo_artifacts.py FILE cfgs NAME         `msg` once per build-script-executed message of the
                                              package that owns bench target NAME, then
                                              `cfg:<cfg>` for each cfg it set

The `cfgs` array of a build-script-executed message is what the build script's
`cargo:rustc-cfg=` lines set. A package_id is not matched on a name: for a path package it is
`path+file:///dir#0.0.0` when the directory is named like the package. A line that is not JSON is
skipped. Standard library only; `ci/bench/cargo_artifacts.sh` runs the same queries with jq when jq
is installed.
"""
import json
import sys


def messages(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            if isinstance(msg, dict):
                yield msg


def bench_artifacts(path, name):
    for msg in messages(path):
        target = msg.get("target") or {}
        if (
            msg.get("reason") == "compiler-artifact"
            and target.get("name") == name
            and "bench" in (target.get("kind") or [])
        ):
            yield msg


def main(argv):
    if len(argv) != 4 or argv[2] not in ("exe", "pid", "cfgs"):
        print(__doc__, file=sys.stderr)
        return 2
    path, mode, arg = argv[1], argv[2], argv[3]
    if mode == "exe":
        for msg in bench_artifacts(path, arg):
            if msg.get("executable"):
                print(msg["executable"])
    elif mode == "pid":
        for msg in bench_artifacts(path, arg):
            print(msg.get("package_id", ""))
    else:
        owners = [msg.get("package_id") for msg in bench_artifacts(path, arg)]
        if not owners:
            return 0
        for msg in messages(path):
            if msg.get("reason") == "build-script-executed" and msg.get("package_id") == owners[0]:
                print("msg")
                for cfg in msg.get("cfgs") or []:
                    print("cfg:" + cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
