"""Draw the source-checked TRACE schematic as vector PDF, SVG/XML and PNG."""
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Rectangle


ROOT = Path(__file__).resolve().parent
MODEL = ROOT.parent / "tread_two/trace_batch/model.py"
WIDTH, HEIGHT = 510, 344
INK = "#1c1c1c"
GRAY = "#5a5a5a"


def build_trace_figure():
    receipt = json.loads((ROOT / "audit/TRACE_ARCHITECTURE.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(MODEL.read_bytes()).hexdigest() == receipt["source_sha256"]
    assert sum(receipt["parameters"].values()) == 71022
    assert receipt["hidden_rgb_invariance_fixed_support"]
    expected = {
        "encoded": [1, 32, 48], "full_context": [1, 32, 48], "context": [1, 32, 48],
        "visible_input": [1, 32, 4, 64, 64], "local_prediction": [1, 32, 6, 4, 4],
        "masked_prediction": [1, 32, 6, 4, 4], "pooled": [1, 240], "projected": [1, 6],
    }
    assert all(receipt["outputs"][name] == shape for name, shape in expected.items())
    reuse = {name: len(receipt["calls"][name]) for name in ("encoder", "decoder", "context", "metadata")}
    assert reuse == {"encoder": 2, "decoder": 2, "context": 2, "metadata": 1}

    fig = plt.figure(figsize=(WIDTH / 72, HEIGHT / 72), dpi=144)
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, WIDTH), ylim=(HEIGHT, 0))
    ax.set_axis_off()
    bounds = []

    def text(x, y, value, size=8.2, *, align="center", italic=False, color=INK):
        return ax.text(x, y, value, fontsize=size, fontfamily="STIXGeneral", ha=align, va="center",
                       fontstyle="italic" if italic else "normal", math_fontfamily="stix",
                       color=color, zorder=5)

    def node(x, top, width, height, title, detail=None, *, learned=False):
        left, right = x - width / 2, x + width / 2
        ax.add_patch(Rectangle((left, top), width, height, fc="#ededed" if learned else "white",
                               ec=INK, lw=.68, zorder=2))
        center = top + height / 2
        labels = [text(x, center, title)] if detail is None else [
            text(x, center - 4.65, title), text(x, center + 4.65, detail, 7.55)]
        bounds.extend((label, (left + 2.5, top + 1, right - 2.5, top + height - 1)) for label in labels)

    def route(*points, arrow=True, dashed=False):
        xs, ys = zip(*points)
        color = GRAY if dashed else INK
        ax.plot(xs, ys, color=color, lw=.68, ls=(0, (2.5, 2.1)) if dashed else "-",
                solid_capstyle="butt", solid_joinstyle="miter", zorder=3)
        if arrow:
            x, y = points[-1]
            px, py = points[-2]
            length = abs(x - px) + abs(y - py)
            assert length > 0 and (x == px or y == py)
            dx, dy = (x - px) / length, (y - py) / length
            tip = [(x, y), (x - 3.7 * dx + 1.6 * dy, y - 3.7 * dy - 1.6 * dx),
                   (x - 3.7 * dx - 1.6 * dy, y - 3.7 * dy + 1.6 * dx)]
            ax.add_patch(Polygon(tip, closed=True, fc=color, ec="none", zorder=4))

    def fork(x, y):
        ax.add_patch(Circle((x, y), 1.45, fc=INK, ec="none", zorder=4))

    def operator(x, y, symbol):
        ax.add_patch(Circle((x, y), 5.25, fc="white", ec=INK, lw=.72, zorder=4))
        text(x, y + .2, symbol, 8.7)

    full, masked, meta, local = 128, 304, 216, 34
    # (a) Shared input, two encoder calls, and three readouts. All flow is upward.
    node(meta, 299, 154, 24, "Image and frozen geometry", "RGB  3 × 512 × 384")
    node(meta, 266, 154, 24, "Canonicalize and tile", "512 × 256 atlas, split into 8 × 4 tiles")
    route((meta, 299), (meta, 290))
    route((meta, 266), (meta, 250), arrow=False)
    for x in (full, masked):
        route((meta, 250), (x, 250), (x, 242))
    fork(meta, 250)
    node(full, 220, 110, 22, "Tile-local contrast", "32 × 4 × 64 × 64")
    node(masked, 220, 110, 22, "Hide raw RGB and support", "Then compute tile-local contrast")
    for x in (full, masked):
        node(x, 188, 110, 24, "Tile encoder E", "4 → 16 → 24 → 32 → 48 → mean", learned=True)
        route((x, 220), (x, 212))
        route((x, 188), (x, 173))
    node(full, 149, 110, 24, "Token replacement", "Unsupported → e")
    node(masked, 149, 110, 24, "Token replacement", "Hidden or unsupported → e")
    node(meta, 149, 58, 24, "Metadata M", "Row, col., cov.", learned=True)
    route((meta, 250), (meta, 173), dashed=True)
    text(meta + 5, 200, "Original", 7.6, align="left", color=GRAY)
    text(meta + 5, 209, "support", 7.6, align="left", color=GRAY)
    route((meta, 149), (meta, 134), arrow=False, dashed=True)
    route((meta, 134), (full + 5.25, 134), dashed=True)
    route((meta, 134), (masked - 5.25, 134), dashed=True)
    fork(meta, 134)
    for x in (full, masked):
        operator(x, 134, "+")
        route((x, 149), (x, 139.25))
        node(x, 99, 110, 22, "Axial stack A", "Two residual blocks", learned=True)
        route((x, 128.75), (x, 121))

    route((full, 181), (local, 181), (local, 76))
    fork(full, 181)
    text(37.5, 173, "Z  32 × 48", 7.45, align="left")
    for x in (local, masked):
        node(x, 51, 58, 25, "Decoder D", "48 → 96 → 96", learned=True)
        route((x, 51), (x, 38))
    route((masked, 99), (masked, 76))
    text(masked + 5, 91, "C  32 × 48", 7.9, align="left")
    for x, name in ((local, "Local fields"), (masked, "Masked fields")):
        text(x, 18, name, 9)
        text(x, 29, "32 × 6 × 4 × 4", 7.9)

    node(full, 72, 74, 15, "Five moments (240)")
    node(full, 47, 74, 18, "Projector P → 6", learned=True)
    route((full, 99), (full, 87))
    text(full + 5, 94, "H  32 × 48", 7.9, align="left")
    route((full, 72), (full, 65))
    route((full, 47), (full, 38))
    text(full, 18, "Global projection", 9)
    text(full, 29, "Paired-view objective", 7.9)
    text(180, 336, "(a) TRACE forward graph", 9.35)
    ax.plot([367, 367], [12, 319], color="#b2b2b2", lw=.55)

    # (b) One of the two axial blocks. The side wire is the identity skip.
    mid, row, col, skip = 437, 404, 470, 503
    text(mid, 283, "X  (48 × 8 × 4)", 8.5)
    route((mid, 273), (mid, 241), arrow=False)
    route((mid, 241), (row, 241), (row, 224))
    route((mid, 241), (col, 241), (col, 224))
    fork(mid, 241)
    for x, kernel in ((row, "3 × 1"), (col, "1 × 3")):
        node(x, 199, 53, 25, "Depthwise", f"{kernel} conv", learned=True)
        route((x, 199), (x, 187), (mid, 187), arrow=False)
    route((mid, 187), (mid, 180.25))
    operator(mid, 175, "+")
    fork(mid, 187)

    node(mid, 146, 83, 15, "GroupNorm (6)", learned=True)
    node(mid, 119, 83, 15, "1 × 1 conv   48 → 96", learned=True)
    node(mid, 96, 83, 13, "SiLU")
    node(mid, 70, 83, 15, "1 × 1 conv   96 → 48", learned=True)
    for start, end in ((169.75, 161), (146, 134), (119, 109), (96, 85), (70, 58.25)):
        route((mid, start), (mid, end))
    operator(mid, 53, "×")
    text(mid + 16, 53, "γ", 9.4, italic=True)
    route((mid + 13, 53), (mid + 5.25, 53))
    operator(mid, 29, "+")
    route((mid, 47.75), (mid, 34.25))
    route((mid, 257), (skip, 257), (skip, 29), (mid + 5.25, 29))
    fork(mid, 257)
    text(skip - 5, 174, "Identity", 7.75, align="right")
    route((mid, 23.75), (mid, 17))
    text(mid, 10, r"$A_j(X)$", 9.4)
    text(mid, 336, "(b) Axial residual block", 9.35)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for label, (left, top, right, bottom) in bounds:
        (x0, y0), (x1, y1) = label.get_window_extent(renderer).get_points()
        a, b = ax.transData.inverted().transform([(x0, y1), (x1, y0)])
        assert left <= a[0] <= b[0] <= right and top <= a[1] <= b[1] <= bottom, label.get_text()
    for label in ax.texts:
        box = label.get_window_extent(renderer)
        assert fig.bbox.contains(box.x0, box.y0) and fig.bbox.contains(box.x1, box.y1), label.get_text()

    with matplotlib.rc_context({"pdf.fonttype": 42, "svg.fonttype": "none", "svg.hashsalt": "TRACE"}):
        for ext in ("pdf", "svg", "png"):
            fig.savefig(ROOT / "figures" / f"architecture.{ext}", facecolor="white", dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    build_trace_figure()
