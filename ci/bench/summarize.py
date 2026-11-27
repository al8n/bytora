#!/usr/bin/env python3
"""Summarise criterion output against the same-run `bytes::Bytes` control.

Absolute nanoseconds do not compare across runners; a ratio against the `bytes` label measured in
the SAME run, for the same operation and the same len/cap (or backing), does. Standard library only.

One platform:
    summarize.py --head DIR [DIR ...] [--base DIR] [--csv FILE] [--title TEXT]
Every platform of a dispatched matrix (the downloaded `bench-<runner>` artifacts):
    summarize.py --platforms ARTIFACTS_DIR [--expect RUNNER ...] [--csv FILE]

DIR is a criterion output directory (`target/criterion`). Several --head directories are rounds of
the same run: each cell is the median over rounds of the per-round point estimate (and ratio), and
the CSV keeps every round. A bench is read from `<bench>/new/estimates.json` (median point estimate
and confidence interval) with identity from `new/benchmark.json`. The base -> head change comes,
in order of preference, from `change/estimates.json` (criterion's own `--baseline-lenient`
comparison), or, with --base DIR, from that directory's `new/estimates.json`, or from an in-tree
`base/estimates.json`. A missing value prints as `—`, never a guess, and a bench that the base
lacks is simply not compared.

--platforms reads `ARTIFACTS_DIR/bench-<runner>/{machine.txt,criterion-rounds/round*}`. It prints a
cross-platform overview first (per bench group: rows are handle labels, columns are platforms, a
cell is the label's largest ratio against `bytes` over every operation and len/cap or backing key,
the median over rounds, with the operation and key it came from; with a base, a second overview of
the largest |change| against the base), then each platform's full tables. A runner named by
--expect whose artifact is missing is named in the overview.

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


class Model:
    """One platform: its rounds, per-id medians and the ratio against the control."""

    def __init__(self, head_dirs, base_dir=None):
        self.rounds = [read_tree(d) for d in head_dirs]
        base_tree = read_tree(base_dir) if base_dir else {}
        self.ids = sorted({i for r in self.rounds for i in r}, key=num_order)
        self.rows = {}
        for i in self.ids:
            ns = [r[i]["est"][0] if i in r else None for r in self.rounds]
            ch = [r[i]["change"][0] if i in r and r[i]["change"] else None for r in self.rounds]
            if all(c is None for c in ch) and i in base_tree:
                b = base_tree[i]["est"][0]
                ch = [(n - b) / b * 100.0 if n is not None and b else None for n in ns]
            base_ns = [
                (base_tree[i]["est"][0] if i in base_tree
                 else (r[i]["base"][0] if i in r and r[i]["base"] else None))
                for r in self.rounds
            ]
            self.rows[i] = {"ns": ns, "change": ch, "base_ns": base_ns}
        self.has_base = any(any(c is not None for c in r["change"]) for r in self.rows.values())

    def ratio(self, i, rd):
        p = parse_id(i)
        if not p or p[4] == CONTROL:
            return None
        group, kind, key, op, _ = p
        ctl = f"{group}/{op}/{CONTROL}/{key}" if kind != "backing" else f"{group}/{op}/{key}/{CONTROL}"
        a = self.rows[i]["ns"][rd]
        c = self.rows.get(ctl, {}).get("ns", [None] * len(self.rounds))[rd]
        return a / c if a is not None and c else None

    def med_ratio(self, i):
        return med([self.ratio(i, rd) for rd in range(len(self.rounds))])

    def med_change(self, i):
        return med(self.rows[i]["change"])

    def csv_rows(self, platform):
        for i in self.ids:
            for rd, tree in enumerate(self.rounds):
                if i not in tree:
                    continue
                est = tree[i]["est"]
                r = self.ratio(i, rd)
                b = self.rows[i]["base_ns"][rd]
                c = self.rows[i]["change"][rd]
                yield [platform, i, rd + 1, est[0], est[1], est[2], "" if r is None else r,
                       "" if b is None else b, "" if c is None else c]

    def tables(self, level="###"):
        out = []
        tables = {}
        flat = []
        for i in self.ids:
            p = parse_id(i)
            if p is None:
                flat.append(i)
                continue
            group, kind, key, op, label = p
            tables.setdefault((group, key), {}).setdefault(label, {})[op] = i
        for (group, key) in sorted(tables, key=lambda k: (k[0], num_order(k[1]))):
            labels = tables[(group, key)]
            ops = sorted({op for l in labels.values() for op in l}, key=num_order)
            head = ["label"] + ops + ["max ratio"] + (["largest change vs base"] if self.has_base else [])
            out.append(f"{level} {group} · {key}\n")
            out.append("| " + " | ".join(head) + " |")
            out.append("|" + "|".join(["---"] + ["---:"] * (len(head) - 1)) + "|")
            for label in sorted(labels, key=label_order):
                cells, best, best_op, worst = [], None, None, None
                for op in ops:
                    i = labels[label].get(op)
                    if i is None:
                        cells.append(DASH)
                        continue
                    n = med(self.rows[i]["ns"])
                    r = self.med_ratio(i)
                    cells.append(fmt_cell(n, r if label != CONTROL else None))
                    if r is not None and (best is None or r > best):
                        best, best_op = r, op
                    c = self.med_change(i)
                    if c is not None and (worst is None or abs(c) > abs(worst)):
                        worst = c
                line = [f"`{label}`"] + cells + [DASH if best is None else f"{best:.2f}× {best_op}"]
                if self.has_base:
                    line.append(fmt_pct(worst))
                out.append("| " + " | ".join(line) + " |")
            out.append("")
        if flat:
            out.append(f"{level} other benchmarks (no control)\n")
            head = ["id", "median ns", "CI low", "CI high"] + (["base ns", "change"] if self.has_base else [])
            out.append("| " + " | ".join(head) + " |")
            out.append("|" + "|".join(["---"] + ["---:"] * (len(head) - 1)) + "|")
            for i in flat:
                tree = next((t for t in self.rounds if i in t), None)
                est = tree[i]["est"] if tree else (None, None, None)
                line = [f"`{i}`", fmt_ns(med(self.rows[i]["ns"])), fmt_ns(est[1]), fmt_ns(est[2])]
                if self.has_base:
                    line += [fmt_ns(med(self.rows[i]["base_ns"])), fmt_pct(self.med_change(i))]
                out.append("| " + " | ".join(line) + " |")
            out.append("")
        return out

    def extremes(self):
        """{(group, label): (best ratio, 'op @ key', largest |change|, 'op @ key')}"""
        res = {}
        for i in self.ids:
            p = parse_id(i)
            if p is None or p[4] == CONTROL:
                continue
            group, kind, key, op, label = p
            cur = res.setdefault((group, label), [None, "", None, ""])
            r = self.med_ratio(i)
            if r is not None and (cur[0] is None or r > cur[0]):
                cur[0], cur[1] = r, f"{op} @ {key}"
            c = self.med_change(i)
            if c is not None and (cur[2] is None or abs(c) > abs(cur[2])):
                cur[2], cur[3] = c, f"{op} @ {key}"
        return res


def read_machine(path):
    info = {}
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                k, sep, v = line.rstrip("\r\n").partition("=")
                if sep and re.fullmatch(r"[a-z_]+", k) and k not in info:
                    info[k] = v.strip()
    except OSError:
        pass
    return info


def overview(models, missing, key_idx, title, fmt):
    """One table per bench group: rows = labels, columns = platforms."""
    names = list(models) + list(missing)
    ext = {n: models[n].extremes() for n in models}
    groups = sorted({g for e in ext.values() for (g, _) in e}, key=num_order)
    out = [f"### {title}\n"]
    for group in groups:
        labels = sorted({l for e in ext.values() for (g, l) in e if g == group}, key=label_order)
        out.append(f"**{group}**\n")
        out.append("| label | " + " | ".join(names) + " |")
        out.append("|---|" + "|".join(["---:"] * len(names)) + "|")
        for label in labels:
            cells = []
            for n in names:
                v = ext.get(n, {}).get((group, label))
                cells.append(fmt(v) if v and v[key_idx] is not None else DASH)
            out.append(f"| `{label}` | " + " | ".join(cells) + " |")
        out.append("")
    return out


def platforms_mode(args):
    root = args.platforms
    found = {}
    if os.path.isdir(root):
        for name in sorted(os.listdir(root)):
            d = os.path.join(root, name)
            if name.startswith("bench-") and os.path.isdir(d):
                found[name[len("bench-"):]] = d
    expected = list(args.expect or [])
    names = list(expected) + [n for n in found if n not in expected]
    models, infos, missing, empty = {}, {}, [], []
    for n in names:
        if n not in found:
            missing.append(n)
            continue
        d = found[n]
        infos[n] = read_machine(os.path.join(d, "machine.txt"))
        rounds = sorted(glob_rounds(os.path.join(d, "criterion-rounds")), key=num_order)
        m = Model(rounds)
        if not m.ids:
            empty.append(n)
            missing.append(n)
            continue
        models[n] = m

    out = [f"## Benchmark matrix: {len(models)} of {len(names)} platform(s) with results\n",
           "A cell is the label's largest ratio against `bytes` in the same run, over every operation and "
           "len/cap (or backing) key of the group, the median over rounds, with the operation and key it "
           "came from. Absolute ns do not compare across runners; ratios do.\n"]
    out.append("| platform | CPU | head | base | state |")
    out.append("|---|---|---|---|---|")
    for n in names:
        i = infos.get(n, {})
        state = "ok" if n in models else ("artifact present, no benchmark results" if n in empty
                                           else "**MISSING: no artifact (the job failed or never ran)**")
        out.append(f"| {n} | {i.get('cpu', DASH) or DASH} | `{(i.get('head') or DASH)[:10]}` | "
                   f"`{(i.get('base') or DASH)[:10]}` | {state} |")
    out.append("")
    if missing:
        out.append("Missing platforms (shown as `—` below): " + ", ".join(f"`{n}`" for n in missing) + "\n")
    if models:
        out += overview(models, missing, 0, "Overview: largest ratio against `bytes`",
                        lambda v: f"{v[0]:.2f}× {v[1]}")
        if any(m.has_base for m in models.values()):
            out += overview(models, missing, 2, "Overview: largest change against the base",
                            lambda v: f"{v[2]:+.1f}% {v[3]}")
        for n, m in models.items():
            i = infos[n]
            out.append(f"## {n} · {i.get('cpu', DASH) or DASH}\n")
            out += m.tables("###")
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(CSV_HEAD)
            for n, m in models.items():
                w.writerows(m.csv_rows(n))
    sys.stdout.write("\n".join(out) + "\n")
    return 0


CSV_HEAD = ["platform", "id", "round", "median_ns", "ci_low_ns", "ci_high_ns", "ratio_vs_bytes",
            "base_median_ns", "change_pct"]


def glob_rounds(d):
    if not os.path.isdir(d):
        return []
    return [os.path.join(d, x) for x in os.listdir(d) if os.path.isdir(os.path.join(d, x))]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--head", nargs="+", help="criterion directories of ONE platform, one per round")
    ap.add_argument("--base", help="criterion directory of a separate base run")
    ap.add_argument("--platforms", help="directory of downloaded `bench-<runner>` artifacts")
    ap.add_argument("--expect", nargs="*", help="runner names that must have an artifact")
    ap.add_argument("--csv", help="write every bench and round to this CSV file")
    ap.add_argument("--title", default="", help="heading for the markdown")
    args = ap.parse_args(argv)
    if args.platforms:
        return platforms_mode(args)
    if not args.head:
        ap.error("--head or --platforms is required")

    m = Model(args.head, args.base)
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(CSV_HEAD)
            w.writerows(m.csv_rows(args.title))
    out = []
    if args.title:
        out.append(f"## {args.title}\n")
    out.append(f"{len(m.ids)} benchmarks, {len(m.rounds)} round(s); a cell is `ns (ratio against `{CONTROL}` "
               "in the same run)`, the median over rounds. Absolute ns do not compare across runners; "
               "ratios do.\n")
    out += m.tables("###")
    if not m.ids:
        out.append(f"No benchmark with `new/estimates.json` found under {', '.join(args.head)}.\n")
    sys.stdout.write("\n".join(out) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
