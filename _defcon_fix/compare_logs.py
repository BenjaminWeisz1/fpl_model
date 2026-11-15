"""Parse predict.py run logs into a comparison table."""
import re
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\bdwei\OneDrive\Desktop\Programz\fpl_model")


def parse(path):
    txt = Path(path).read_text(encoding="utf-8", errors="replace")
    blocks = {}
    pat = re.compile(
        r"OUT-OF-TIME VALIDATION: (?P<label>.+?)\n=+\n"
        r"\s*Val: (?P<n>\d+) records \| target rate: (?P<rate>[\d.]+)\n"
        r"\s*Log loss:\s*(?P<ll>[\d.]+)\n"
        r"\s*AUC:\s*(?P<auc>[\d.]+)\n"
        r"\s*Brier:\s*(?P<brier>[\d.]+)\n"
        r"\s*ECE:\s*(?P<ece>[\d.]+)")
    for m in pat.finditer(txt):
        blocks[m.group("label").strip()] = {
            "n": int(m.group("n")), "rate": float(m.group("rate")),
            "ll": float(m.group("ll")), "auc": float(m.group("auc")),
            "brier": float(m.group("brier")), "ece": float(m.group("ece"))}
    extra = {}
    m = re.search(r"Defender: (\d+) prior-carrying validation rows\n"
                  r"\s*mean prediction, priors OFF: ([\d.]+)\n"
                  r"\s*mean prediction, priors ON:\s*([\d.]+)\n"
                  r"\s*observed target rate:\s*([\d.]+)", txt)
    if m:
        extra["prior_rows"] = {"n": int(m.group(1)), "off": float(m.group(2)),
                               "on": float(m.group(3)), "obs": float(m.group(4))}
    m = re.search(r"MatchesPlayed buckets OFF -> ON: (.+)", txt)
    if m:
        extra["buckets"] = m.group(1).strip()
    return blocks, extra


LABELS = [
    "Defender (Season 2025)",
    "Midfielder/Forward (Season 2025)",
    "Midfielder-only (Season 2025)",
    "Forward-only (Season 2025)",
    "Defender (Season 2026, priors ON)",
    "Midfielder/Forward (Season 2026, priors ON)",
    "Defender (Season 2026, priors OFF -- ablation)",
]

runs = [(Path(p).stem, *parse(p)) for p in sys.argv[1:]]
w = max(len(l) for l in LABELS) + 2
for metric in ("ll", "auc", "brier", "ece"):
    print(f"\n=== {metric.upper()} ===")
    head = f"{'block':<{w}}" + "".join(f"{n:>26}" for n, _, _ in runs)
    print(head)
    for lab in LABELS:
        row = f"{lab:<{w}}"
        base = None
        for i, (_, b, _) in enumerate(runs):
            v = b.get(lab, {}).get(metric)
            if v is None:
                row += f"{'-':>26}"
                continue
            if i == 0:
                base = v
                row += f"{v:>26.4f}"
            else:
                row += f"{v:>18.4f} ({v - base:+.4f})"
        print(row)

print(f"\n=== validation n ===")
print(f"{'block':<{w}}" + "".join(f"{n:>14}" for n, _, _ in runs))
for lab in LABELS:
    print(f"{lab:<{w}}" + "".join(f"{b.get(lab, {}).get('n', '-'):>14}" for _, b, _ in runs))

print("\n=== prior-carrying defender rows (Split B) ===")
for n, _, e in runs:
    pr = e.get("prior_rows")
    if pr:
        print(f"  {n:<22} n={pr['n']}  mean pred OFF {pr['off']:.4f} -> ON {pr['on']:.4f}"
              f"   observed {pr['obs']:.4f}")
    if e.get("buckets"):
        print(f"  {'':<22} buckets OFF -> ON: {e['buckets']}")
