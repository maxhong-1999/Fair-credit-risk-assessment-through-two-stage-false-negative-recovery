"""Render Figures C1/C2 from aggregate panel A; no data access or model fitting."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors, cm
from matplotlib.text import Text
import numpy as np

ROOT = Path(__file__).resolve().parent
WIDTH, HEIGHT = 180.0, 84.0
FIGURES = (("C1", "Gv", ("Gv", "Gv_Pv", "Gv_Pt"),
            ("Vanilla G", "Vanilla G + Vanilla P", "Vanilla G + Tuned P")),
           ("C2", "Gt", ("Gt", "Gt_Pv", "Gt_Pt"),
            ("Tuned G", "Tuned G + Vanilla P", "Tuned G + Tuned P")))


def signed(value):
    return ("+" if value >= 0 else "\u2212") + f"{abs(value):,}"


def caption(number, family, negatives, positives):
    return (f"Figure {number}. Pooled held-out confusion matrices for the {family} XGBoost "
            "G-model and its two-stage XGBoost P-model variants. Rows are actual negative "
            "and positive classes; columns are predicted negative and positive classes. "
            "Positive denotes delinquency. Blue intensity encodes the proportion within "
            "each actual-class row, with one shared 0\u20131 scale across all six panels "
            "(colorbar: Row proportion). The blue palette is lightened for black text; "
            "the numeric color scale is unchanged. Cell annotations are raw counts. "
            "Parentheses below two-stage counts show signed raw-count changes relative "
            "to that figure's G baseline; color does not encode those changes. "
            "FN recovery (%) = 100 \u00d7 (baseline FN \u2212 two-stage FN) / baseline FN. "
            "\u0394FP/(\u2212\u0394FN) is the additional FP count per recovered FN. N/A denotes "
            "baseline entries. This ratio is descriptive, not a cost or overall-performance score. "
            "Gv/Gt denote vanilla/tuned G; Pv/Pt denote vanilla/tuned P. "
            f"Each model has {negatives:,} actual negatives and {positives:,} actual positives.")


def build(number, baseline, methods, titles, rows, outdir, cmap):
    family = "vanilla" if baseline == "Gv" else "tuned"
    stem = f"figure_{number}_{family}_G_confusion_matrices"
    norm = colors.Normalize(0, 1)
    fig = plt.figure(figsize=(WIDTH / 25.4, HEIGHT / 25.4), dpi=150)
    panels = []
    def figtext(x, y, text, size=8, **kwargs):
        return fig.text(x / WIDTH, 1 - y / HEIGHT, text, fontsize=size, va="center", **kwargs)
    for method, title, left in zip(methods, titles, (17, 72, 127)):
        figtext(left + 18, 8, title, size=9, ha="center")
        ax = fig.add_axes([left / WIDTH, (HEIGHT - 56) / HEIGHT, 36 / WIDTH, 36 / HEIGHT])
        counts = rows[method]
        raw = np.array([[counts["tn"], counts["fp"]], [counts["fn"], counts["tp"]]])
        proportion = raw / raw.sum(axis=1, keepdims=True)
        ax.pcolormesh([0, 1, 2], [0, 1, 2], proportion, cmap=cmap, norm=norm,
                      edgecolors="white", linewidth=.8, antialiased=False, rasterized=False)
        ax.set(xlim=(0, 2), ylim=(2, 0), xticks=[.5, 1.5], yticks=[.5, 1.5],
               xticklabels=["Predicted\nNegative", "Predicted\nPositive"],
               yticklabels=["Actual\nNegative", "Actual\nPositive"])
        ax.tick_params(axis="both", length=0, pad=3)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_multialignment("center")
        for spine in ax.spines.values():
            spine.set_visible(False)
        base = rows[baseline]
        changes = {key: counts[key] - base[key] for key in counts}
        for row_index, keys in enumerate((("tn", "fp"), ("fn", "tp"))):
            for column, key in enumerate(keys):
                y = row_index + (.5 if method == baseline else .41)
                ax.text(column + .5, y, f"{counts[key]:,}", ha="center", va="center", fontsize=10.5)
                if method != baseline:
                    ax.text(column + .5, row_index + .65, f"({signed(changes[key])})",
                            ha="center", va="center", fontsize=8.2)
        recovery = ratio = None
        recovery_text = ratio_text = "N/A"
        if method != baseline:
            recovered = base["fn"] - counts["fn"]
            if base["fn"]:
                recovery = 100 * recovered / base["fn"]
                recovery_text = f"{recovery:.2f}%"
            if recovered:
                ratio = changes["fp"] / recovered
                ratio_text = f"{ratio:.1f}"
        figtext(left - 2, 72, "FN recovery (%)")
        figtext(left + 36, 72, recovery_text, ha="right", fontweight="bold")
        figtext(left - 2, 78, "\u0394FP / (\u2212\u0394FN)")
        figtext(left + 36, 78, ratio_text, ha="right", fontweight="bold")
        panels.append({"method": method, "title": title, "counts": counts,
                       "row_proportions": proportion.tolist(),
                       "signed_count_changes": None if method == baseline else changes,
                       "FN_recovery_percent": recovery, "additional_FP_per_recovered_FN": ratio})
    cax = fig.add_axes([165 / WIDTH, (HEIGHT - 56) / HEIGHT, 3.3 / WIDTH, 36 / HEIGHT])
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                        ticks=np.linspace(0, 1, 6), format="%.1f")
    cbar.set_label("Row proportion", fontsize=8, labelpad=3)
    cbar.ax.tick_params(labelsize=8, width=.5, length=2, pad=2)
    cbar.outline.set_linewidth(.5)
    cbar.solids.set_rasterized(False)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [item for item in fig.findobj(Text) if item.get_visible() and item.get_text()]
    boxes = []
    for item in texts:
        box = item.get_window_extent(renderer)
        if item.get_fontsize() < 8 or colors.to_rgba(item.get_color()) != (0., 0., 0., 1.):
            raise ValueError("Figure text must be black and at least 8 pt")
        if box.x0 < 0 or box.y0 < 0 or box.x1 > fig.bbox.width or box.y1 > fig.bbox.height:
            raise ValueError("Figure text exceeds canvas bounds")
        for other in boxes:
            if min(box.x1, other.x1) - max(box.x0, other.x0) > 1 and min(box.y1, other.y1) - max(box.y0, other.y0) > 1:
                raise ValueError("Figure text overlaps")
        boxes.append(box)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(outdir / f"{stem}.{ext}", dpi=600, facecolor="white",
                    metadata={"Creator": "Python / matplotlib"} if ext != "png" else None)
    fig.savefig(outdir / f"{stem}_proof.png", dpi=150, facecolor="white")
    plt.close(fig)
    base = rows[baseline]
    return {"figure": number, "stem": stem, "panels": panels, "width_mm": WIDTH,
            "height_mm": HEIGHT, "png_dpi": 600, "minimum_font_pt": 8,
            "text_clipping_count": 0, "text_overlap_count": 0, "all_text_black": True,
            "caption": caption(number, family, base["tn"] + base["fp"], base["fn"] + base["tp"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-a", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("--output-dir must be a new directory")
    with args.panel_a.open(newline="", encoding="utf-8-sig") as f:
        records = list(csv.DictReader(f))
    rows = {r["method"]: {key: int(r[key]) for key in ("tn", "fp", "fn", "tp")} for r in records}
    expected = {method for _, _, methods, _ in FIGURES for method in methods}
    if set(rows) != expected or len(records) != len(expected):
        raise ValueError("All six native methods are required exactly once")
    reference = rows["Gv"]
    class_totals = (reference["tn"] + reference["fp"], reference["tp"] + reference["fn"])
    if min(class_totals) <= 0:
        raise ValueError("Both actual classes are required for row proportions")
    for counts in rows.values():
        if min(counts.values()) < 0 or (counts["tn"] + counts["fp"], counts["tp"] + counts["fn"]) != class_totals:
            raise ValueError("Figure actual-class totals must agree across methods")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.labelsize": 8, "axes.titlesize": 9,
                         "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "figure.facecolor": "white", "axes.facecolor": "white",
                         "savefig.facecolor": "white", "savefig.edgecolor": "white",
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "path",
                         "axes.unicode_minus": True, "text.color": "black",
                         "axes.labelcolor": "black", "xtick.color": "black", "ytick.color": "black"})
    cmap = colors.LinearSegmentedColormap.from_list(
        "Blues_for_black_text", matplotlib.colormaps["Blues"](np.linspace(0, .50, 256)))
    args.output_dir.mkdir(parents=True)
    figures = [build(*spec, rows, args.output_dir, cmap) for spec in FIGURES]
    (args.output_dir / "figure_captions.txt").write_text(
        "\n\n".join(f["caption"] for f in figures) + "\n", encoding="utf-8")
    manifest = {"source_panel_A_sha256": hashlib.sha256(args.panel_a.read_bytes()).hexdigest(),
                "renderer": {"matplotlib": matplotlib.__version__, "numpy": np.__version__},
                "color_quantity": "within-actual-class row proportion", "color_scale": [0, 1],
                "colormap": "Blues sampled from 0.0 to 0.50", "figures": figures}
    (args.output_dir / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"figures": [f["figure"] for f in figures], "text_clipping_count": 0,
                      "text_overlap_count": 0, "model_fits": 0}))


if __name__ == "__main__":
    main()
