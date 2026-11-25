#!/usr/bin/env python3
"""Per-operation instruction and allocator counts from callgrind outputs of `benches/opcount.rs`.

    callgrind_ops.py --dir DIR [--platform NAME] [--csv FILE] [--ks K1 K2]

DIR holds `callgrind.<case>.<k>.out` files (`<case>` is an opcount case, or `<case>@setup`, the same
loop with only the setup), each a run of `opcount <case> <k>` under `valgrind --tool=callgrind`.
Every number is a per-iteration cost with the process's fixed cost cancelled:
(count at the larger k - count at the smaller k) / (larger k - smaller k), by default 2000 and 4000.

The CSV has one row per case: platform, case, label, op, len, cap, ir_per_op (the whole iteration
loop, setup and the output's drop included), ir_setup_per_op (the `@setup` loop), ir_net_per_op (the
difference), then for each allocator function `<name>`, `<name>_setup` and `<name>_net` calls per
iteration. A function whose name appears in neither run of a case has an empty cell: a count that
the name does not give is not zero. The allocator functions are libc's `malloc`, `free`, `realloc`
and `calloc`, and `__rust_alloc`, `__rust_dealloc` and `__rust_realloc` where they appear by name
(an inlined shim has no symbol).

Stdout is two markdown tables, per (operation, len, capacity) and per label, each the iteration less
its setup: instructions with the ratio against `bytes`, and allocator calls as `malloc/free/realloc`
(`calloc` counted with `malloc`; the `__rust_*` names stand in when libc's do not appear).
Standard library only.
"""
import argparse
import csv
import glob
import os
import re
import sys

LABELS = ["bytes", "n_0", "n_16", "n_24", "n_32"]
OPS = ["from_vec", "try_from_vec", "consume", "warmed_clone", "range"]
CONTROL = "bytes"

ALLOC_NAMES = [
    "malloc",
    "free",
    "realloc",
    "calloc",
    "__rust_alloc",
    "__rust_dealloc",
    "__rust_realloc",
]
# glibc exports some of these under a second name; only one is attached to an address.
ALIASES = {
    "malloc": ("malloc", "__libc_malloc"),
    "free": ("free", "__libc_free", "cfree"),
    "realloc": ("realloc", "__libc_realloc"),
    "calloc": ("calloc", "__libc_calloc"),
    "__rust_alloc": ("__rust_alloc",),
    "__rust_dealloc": ("__rust_dealloc",),
    "__rust_realloc": ("__rust_realloc",),
}
WATCHED = {alias for aliases in ALIASES.values() for alias in aliases}

FILE_RE = re.compile(r"^callgrind\.(?P<case>.+)\.(?P<k>\d+)\.out$")
CASE_RE = re.compile(
    r"^(?P<label>[a-z0-9_]+)-(?P<op>[a-z0-9_]+)-(?P<len>\d+)(?:c(?P<cap>\d+))?$"
)
NAME_RE = re.compile(r"^(cfn|fn)=(.*)$")
ID_RE = re.compile(r"^\((\d+)\)\s*(.*)$")
RECURSION_RE = re.compile(r"'\d+$")


def parse_callgrind(path):
    """(total Ir, {watched function name: calls made to it}) of one callgrind output file.

    The cost line after a `calls=` record is the inclusive cost of those calls, not a self cost; it
    is skipped. `fn=` and `cfn=` share one id space, so a name given once is `(id)` afterwards.
    """
    names = {}
    calls = {}
    totals = None
    summary = None
    callee = None
    skip = False
    with open(path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if skip:
                skip = False
                continue
            if line.startswith("calls="):
                fields = line[len("calls="):].split()
                if callee in WATCHED and fields:
                    calls[callee] = calls.get(callee, 0) + int(fields[0])
                skip = True
                continue
            m = NAME_RE.match(line)
            if m:
                body = m.group(2)
                ident = ID_RE.match(body)
                if ident:
                    if ident.group(2):
                        names[ident.group(1)] = ident.group(2)
                    name = names.get(ident.group(1), "")
                else:
                    name = body
                if m.group(1) == "cfn":
                    callee = RECURSION_RE.sub("", name)
                continue
            if line.startswith("totals:"):
                totals = int(line.split()[1])
            elif line.startswith("summary:"):
                summary = int(line.split()[1])
    total = totals if totals is not None else summary
    if total is None:
        raise ValueError(f"{path}: no totals or summary line")
    return total, calls


def family(calls, name):
    """Calls to one allocator function over its aliases, or None if no alias appears at all."""
    present = [alias for alias in ALIASES[name] if alias in calls]
    if not present:
        return None
    return sum(calls[alias] for alias in present)


def per_iteration(runs, field):
    """(value at the larger k - value at the smaller k) / (larger k - smaller k); None if absent."""
    ks = sorted(runs)
    low, high = runs[ks[0]], runs[ks[-1]]
    a, b = field(low), field(high)
    if a is None and b is None:
        return None
    return ((b or 0) - (a or 0)) / (ks[-1] - ks[0])


def load(directory, wanted):
    """{case: {k: (total, calls)}} for every output under directory, limited to the ks in wanted."""
    runs = {}
    for path in sorted(glob.glob(os.path.join(directory, "callgrind.*.out"))):
        m = FILE_RE.match(os.path.basename(path))
        if not m:
            continue
        k = int(m.group("k"))
        if wanted and k not in wanted:
            continue
        try:
            runs.setdefault(m.group("case"), {})[k] = parse_callgrind(path)
        except (OSError, ValueError) as err:
            print(f"callgrind_ops: skipped {path}: {err}", file=sys.stderr)
    return runs


def sub(a, b):
    return None if a is None and b is None else (a or 0) - (b or 0)


def measure(runs):
    """{case: record} of per-iteration counts, for every case with two k and its @setup."""
    out = {}
    for case, by_k in sorted(runs.items()):
        if case.endswith("@setup"):
            continue
        m = CASE_RE.match(case)
        if not m:
            print(f"callgrind_ops: unrecognised case {case}", file=sys.stderr)
            continue
        setup = runs.get(case + "@setup")
        if len(by_k) < 2 or not setup or len(setup) < 2:
            print(f"callgrind_ops: {case} lacks two runs and its setup", file=sys.stderr)
            continue
        rec = {
            "case": case,
            "label": m.group("label"),
            "op": m.group("op"),
            "len": int(m.group("len")),
            "cap": int(m.group("cap") or m.group("len")),
            "ir": per_iteration(by_k, lambda r: r[0]),
            "ir_setup": per_iteration(setup, lambda r: r[0]),
        }
        rec["ir_net"] = sub(rec["ir"], rec["ir_setup"])
        for name in ALLOC_NAMES:
            rec[name] = per_iteration(by_k, lambda r, n=name: family(r[1], n))
            rec[name + "_setup"] = per_iteration(setup, lambda r, n=name: family(r[1], n))
            rec[name + "_net"] = sub(rec[name], rec[name + "_setup"])
        out[case] = rec
    return out


def number(x, places=3):
    return "" if x is None else f"{x:.{places}f}"


def write_csv(path, platform, records):
    head = ["platform", "case", "label", "op", "len", "cap", "ir_per_op", "ir_setup_per_op"]
    head.append("ir_net_per_op")
    for name in ALLOC_NAMES:
        head += [name, name + "_setup", name + "_net"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(head)
        for rec in records.values():
            row = [platform, rec["case"], rec["label"], rec["op"], rec["len"], rec["cap"]]
            row += [number(rec["ir"]), number(rec["ir_setup"]), number(rec["ir_net"])]
            for name in ALLOC_NAMES:
                row += [number(rec[name]), number(rec[name + "_setup"]), number(rec[name + "_net"])]
            w.writerow(row)


def count(x):
    if x is None:
        return "—"
    if abs(x - round(x)) < 0.005:
        return str(int(round(x)))
    return f"{x:.2f}"


def alloc_calls(rec):
    """(malloc + calloc, free, realloc) net per iteration, libc's names first, then __rust_*."""
    libc = [rec[n + "_net"] for n in ("malloc", "calloc", "free", "realloc")]
    if any(v is not None for v in libc):
        m, c, f, r = libc
        return ((m or 0) + (c or 0), f or 0, r or 0)
    shim = [rec[n + "_net"] for n in ("__rust_alloc", "__rust_dealloc", "__rust_realloc")]
    if any(v is not None for v in shim):
        return tuple(v or 0 for v in shim)
    return None


def row_order(key):
    op, length, cap = key
    return (OPS.index(op) if op in OPS else len(OPS), length, cap)


def tables(platform, records):
    rows = {}
    for rec in records.values():
        rows.setdefault((rec["op"], rec["len"], rec["cap"]), {})[rec["label"]] = rec
    labels = [l for l in LABELS if any(l in cells for cells in rows.values())]
    labels += sorted({r["label"] for r in records.values()} - set(LABELS))
    out = []
    title = f" on {platform}" if platform else ""
    out.append(f"#### instructions per op, op less setup{title}\n")
    out.append("Valgrind callgrind `Ir`; in parentheses, the ratio against `bytes` for the same row.\n")
    head = ["op", "len", "cap"] + labels
    out.append("| " + " | ".join(head) + " |")
    out.append("|" + "|".join(["---", "---:", "---:"] + ["---:"] * len(labels)) + "|")
    for key in sorted(rows, key=row_order):
        cells = rows[key]
        ctl = cells.get(CONTROL)
        line = [f"`{key[0]}`", str(key[1]), str(key[2])]
        for label in labels:
            rec = cells.get(label)
            if rec is None or rec["ir_net"] is None:
                line.append("—")
                continue
            text = f"{rec['ir_net']:.1f}"
            if label != CONTROL and ctl and ctl["ir_net"]:
                text += f" ({rec['ir_net'] / ctl['ir_net']:.2f}×)"
            line.append(text)
        out.append("| " + " | ".join(line) + " |")
    out.append("")
    out.append(f"#### allocator calls per op, op less setup{title}\n")
    out.append("`malloc/free/realloc` (`calloc` counted with `malloc`).\n")
    out.append("| " + " | ".join(head) + " |")
    out.append("|" + "|".join(["---", "---:", "---:"] + ["---:"] * len(labels)) + "|")
    for key in sorted(rows, key=row_order):
        cells = rows[key]
        line = [f"`{key[0]}`", str(key[1]), str(key[2])]
        for label in labels:
            rec = cells.get(label)
            calls = alloc_calls(rec) if rec else None
            line.append("—" if calls is None else "/".join(count(v) for v in calls))
        out.append("| " + " | ".join(line) + " |")
    out.append("")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", required=True, help="directory of callgrind.<case>.<k>.out files")
    ap.add_argument("--platform", default="", help="platform name for the CSV and table titles")
    ap.add_argument("--csv", help="write the per-case CSV here")
    ap.add_argument("--ks", type=int, nargs="*", default=[2000, 4000],
                    help="the k values to read (default 2000 4000)")
    args = ap.parse_args()
    records = measure(load(args.dir, set(args.ks)))
    if args.csv:
        write_csv(args.csv, args.platform, records)
    if not records:
        print(f"callgrind_ops: no complete case under {args.dir}", file=sys.stderr)
        print("no callgrind counts")
        return 1
    print("\n".join(tables(args.platform, records)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
