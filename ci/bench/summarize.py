#!/usr/bin/env python3
"""Summarise criterion output against the same-run `bytes::Bytes` control.

Absolute nanoseconds do not compare across runners; a ratio against the `bytes` label measured in
the SAME run, for the same operation and the same len/cap (or backing), does. Standard library only.

    summarize.py --head DIR [DIR ...] [--base DIR] [--csv FILE] [--title TEXT]

DIR is a criterion output directory (`target/criterion`). Several --head directories are rounds of
the same run: each cell is the median over rounds of the per-round point estimate (and ratio), and
the CSV keeps every round. A bench is read from `<bench>/new/estimates.json` (median point estimate
and confidence interval) with identity from `new/benchmark.json`. The base -> head change comes,
in order of preference, from `change/estimates.json` (criterion's own `--baseline` comparison),
or, with --base DIR, from that directory's `new/estimates.json`, or from an in-tree
`base/estimates.json` (`--save-baseline base`). A missing value prints as `—`, never a guess, and
a bench that the base lacks is simply not compared.

Bench ids understood as a table of (labels x operations):
  budgets_*/{op}/{label}/len_{len}            one table per (group, len)
  stages_*/{op}/{label}/len_{len}_cap_{cap}   one table per (group, len, cap)
  storage/{op}/{backing}/{label}              one table per (group, backing)
Every other bench is listed flat. The control label is `bytes`.
"""
import argparse
import csv
import json
import os
import re
import statistics
import sys

DASH = "—"
CONTROL = "bytes"
LABEL_RE = re.compile(r"^n_(\d+)_handle_(\d+)$")


def load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def median_of(estimates):
    """(point, lo, hi) of the median estimate, or None."""
    if not isinstance(estimates, dict):
        return None
    m = estimates.get("median")
    if not isinstance(m, dict):
        return None
    ci = m.get("confidence_interval") or {}
    try:
        return (float(m["point_estimate"]), float(ci["lower_bound"]), float(ci["upper_bound"]))
    except (KeyError, TypeError, ValueError):
        return None


def walk(root):
    """Yield every benchmark directory (one that has new/estimates.json) under root."""
    for dirpath, dirnames, filenames in os.walk(root):
        if os.path.basename(dirpath) == "new" and "estimates.json" in filenames:
            yield os.path.dirname(dirpath)
            dirnames[:] = []


def read_tree(root):
    """{full_id: {"est": (p, lo, hi), "base": (p,lo,hi)|None, "change": (pct, lo, hi)|None}}"""
    out = {}
    if not root or not os.path.isdir(root):
        return out
    for d in walk(root):
        meta = load_json(os.path.join(d, "new", "benchmark.json"))
        full_id = meta.get("full_id") if isinstance(meta, dict) else None
        if not full_id:
            continue
        est = median_of(load_json(os.path.join(d, "new", "estimates.json")))
        if est is None:
            continue
        base = median_of(load_json(os.path.join(d, "base", "estimates.json")))
        change = median_of(load_json(os.path.join(d, "change", "estimates.json")))
        if change is not None:
            change = tuple(100.0 * v for v in change)  # criterion stores a fraction
        out[full_id] = {"est": est, "base": base, "change": change}
    return out


def parse_id(full_id):
    """-> (group, kind, key, op, label) or None for an unrecognised shape."""
    parts = full_id.split("/")
    group = parts[0]
    if group.startswith("budgets_") and len(parts) >= 4 and parts[-1].startswith("len_"):
        return group, "len", parts[-1], "/".join(parts[1:-2]), parts[-2]
    if group.startswith("stages_") and len(parts) >= 4 and parts[-1].startswith("len_"):
        return group, "len_cap", parts[-1], "/".join(parts[1:-2]), parts[-2]
    if group == "storage" and len(parts) >= 4:
        return group, "backing", parts[-2], "/".join(parts[1:-2]), parts[-1]
    return None


def label_order(label):
    m = LABEL_RE.match(label)
    if m:
        return (0, int(m.group(1)), label)
    if label == CONTROL:
        return (2, 0, label)
    return (1, 0, label)


def num_order(text):
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", text)]


def fmt_ns(v):
    return DASH if v is None else f"{v:.2f}"


def fmt_cell(ns, ratio):
    if ns is None:
        return DASH
    return f"{ns:.2f} ({ratio:.2f}×)" if ratio is not None else f"{ns:.2f} (—)"


def fmt_pct(v):
    return DASH if v is None else f"{v:+.1f}%"


def med(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--head", nargs="+", required=True, help="criterion directories, one per round")
    ap.add_argument("--base", help="criterion directory of a separate base run")
    ap.add_argument("--csv", help="write every bench and round to this CSV file")
    ap.add_argument("--title", default="", help="heading for the markdown")
    args = ap.parse_args(argv)

    rounds = [read_tree(d) for d in args.head]
    base_tree = read_tree(args.base) if args.base else {}
    ids = sorted({i for r in rounds for i in r}, key=num_order)

    # per id: per-round ns, the change and the base ns (median over rounds)
    rows = {}
    for i in ids:
        ns = [r[i]["est"][0] if i in r else None for r in rounds]
        ch = [r[i]["change"][0] if i in r and r[i]["change"] else None for r in rounds]
        if all(c is None for c in ch) and i in base_tree:
            b = base_tree[i]["est"][0]
            ch = [(n - b) / b * 100.0 if n is not None and b else None for n in ns]
        base_ns = [
            (base_tree[i]["est"][0] if i in base_tree else (r[i]["base"][0] if i in r and r[i]["base"] else None))
            for r in rounds
        ]
        rows[i] = {"ns": ns, "change": ch, "base_ns": base_ns}

    # ratios against the same-run control, per round
    def ratio(i, rd):
        p = parse_id(i)
        if not p or p[4] == CONTROL:
            return None
        group, kind, key, op, _ = p
        ctl = f"{group}/{op}/{CONTROL}/{key}" if kind != "backing" else f"{group}/{op}/{key}/{CONTROL}"
        a = rows[i]["ns"][rd]
        c = rows.get(ctl, {}).get("ns", [None] * len(rounds))[rd]
        return a / c if a is not None and c else None

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "round", "median_ns", "ci_low_ns", "ci_high_ns", "ratio_vs_bytes",
                        "base_median_ns", "change_pct"])
            for i in ids:
                for rd, tree in enumerate(rounds):
                    if i not in tree:
                        continue
                    est = tree[i]["est"]
                    r = ratio(i, rd)
                    w.writerow([i, rd + 1, est[0], est[1], est[2], "" if r is None else r,
                                "" if rows[i]["base_ns"][rd] is None else rows[i]["base_ns"][rd],
                                "" if rows[i]["change"][rd] is None else rows[i]["change"][rd]])

    has_base = any(any(c is not None for c in r["change"]) for r in rows.values())
    out = []
    if args.title:
        out.append(f"## {args.title}\n")
    out.append(f"{len(ids)} benchmarks, {len(rounds)} round(s); a cell is `ns (ratio against `{CONTROL}` "
               "in the same run)`, the median over rounds. Absolute ns do not compare across runners; "
               "ratios do.\n")

    tables = {}
    flat = []
    for i in ids:
        p = parse_id(i)
        if p is None:
            flat.append(i)
            continue
        group, kind, key, op, label = p
        tables.setdefault((group, key), {}).setdefault(label, {})[op] = i

    for (group, key) in sorted(tables, key=lambda k: (k[0], num_order(k[1]))):
        labels = tables[(group, key)]
        ops = sorted({op for l in labels.values() for op in l}, key=num_order)
        head = ["label"] + ops + ["max ratio"] + (["largest change vs base"] if has_base else [])
        out.append(f"### {group} · {key}\n")
        out.append("| " + " | ".join(head) + " |")
        out.append("|" + "|".join(["---"] + ["---:"] * (len(head) - 1)) + "|")
        for label in sorted(labels, key=label_order):
            cells, best, best_op, worst_change = [], None, None, None
            for op in ops:
                i = labels[label].get(op)
                if i is None:
                    cells.append(DASH)
                    continue
                n = med(rows[i]["ns"])
                r = med([ratio(i, rd) for rd in range(len(rounds))])
                cells.append(fmt_cell(n, r if label != CONTROL else None))
                if r is not None and (best is None or r > best):
                    best, best_op = r, op
                c = med(rows[i]["change"])
                if c is not None and (worst_change is None or abs(c) > abs(worst_change)):
                    worst_change = c
            line = [f"`{label}`"] + cells + [DASH if best is None else f"{best:.2f}× {best_op}"]
            if has_base:
                line.append(fmt_pct(worst_change))
            out.append("| " + " | ".join(line) + " |")
        out.append("")

    if flat:
        out.append("### other benchmarks (no control)\n")
        head = ["id", "median ns", "CI low", "CI high"] + (["base ns", "change"] if has_base else [])
        out.append("| " + " | ".join(head) + " |")
        out.append("|" + "|".join(["---"] + ["---:"] * (len(head) - 1)) + "|")
        for i in flat:
            tree = next((t for t in rounds if i in t), None)
            est = tree[i]["est"] if tree else (None, None, None)
            line = [f"`{i}`", fmt_ns(med(rows[i]["ns"])), fmt_ns(est[1]), fmt_ns(est[2])]
            if has_base:
                line += [fmt_ns(med(rows[i]["base_ns"])), fmt_pct(med(rows[i]["change"]))]
            out.append("| " + " | ".join(line) + " |")
        out.append("")

    if not ids:
        out.append(f"No benchmark with `new/estimates.json` found under {', '.join(args.head)}.\n")
    sys.stdout.write("\n".join(out) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
