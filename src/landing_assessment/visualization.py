"""Four-panel landing assessment visualization dashboard."""

import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch
from PIL import Image

REC_COLOR = {"SAFE": "#16A34A", "CAUTION": "#EA8A1A", "UNSAFE": "#DC2626"}
REC_TINT = {"SAFE": "#DCFCE7", "CAUTION": "#FEF3C7", "UNSAFE": "#FEE2E2"}
INK = "#0F172A"
INK_MID = "#475569"
INK_LIGHT = "#94A3B8"
RULE = "#E2E8F0"


def _wrap(text: str, width: int = 44) -> list:
    out, cur = [], ""
    for w in text.split():
        if len(cur) + len(w) + 1 <= width:
            cur = (cur + " " + w).strip()
        else:
            out.append(cur)
            cur = w
    if cur:
        out.append(cur)
    return out


def _bar(ax, x, y, w, h, value, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.001",
                 facecolor="#E2E8F0", edgecolor="none", transform=ax.transAxes, zorder=2))
    if value > 0.001:
        ax.add_patch(FancyBboxPatch((x, y), max(w * value, 0.005), h, boxstyle="round,pad=0.001",
                     facecolor=color, edgecolor="none", transform=ax.transAxes, zorder=3))


def _sub_color(v: float) -> str:
    if v >= 0.65:
        return REC_COLOR["SAFE"]
    if v >= 0.30:
        return REC_COLOR["CAUTION"]
    return REC_COLOR["UNSAFE"]


def _frame(ax, color=RULE, lw=1.0):
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor(color)
        sp.set_linewidth(lw)


def visualize_landing_assessment(image: Image.Image, detections: list, assessment: dict,
                                  weights: dict, factor_of_safety: float = 1.2,
                                  title: str = "Landing site assessment",
                                  figsize=(20, 7.2), save_path: str = None):
    rec = assessment["recommendation"]
    safety = assessment["safety_score"]
    risk = assessment["risk_score"]
    sub = assessment["sub_scores"]
    geo = assessment["geometry"]
    nearby = assessment["nearby_hazards"]
    rationale = assessment["rationale"]
    W, H = image.size
    veh_r = geo["vehicle_radius_required_px"]
    veh_r_fos = int(veh_r * factor_of_safety)

    plt.rcParams.update({"font.family": "DejaVu Sans"})

    fig = plt.figure(figsize=figsize, facecolor="white")
    fig.patch.set_facecolor("white")

    gs = gridspec.GridSpec(
        2, 4, figure=fig,
        height_ratios=[0.32, 6.5],
        width_ratios=[4, 4, 4, 2.8],
        hspace=0.04, wspace=0.025,
        left=0.006, right=0.994,
        top=0.985, bottom=0.012,
    )

    ax_hdr = fig.add_subplot(gs[0, :])
    ax_o = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_g = fig.add_subplot(gs[1, 2])
    ax_i = fig.add_subplot(gs[1, 3])

    ax_hdr.set_facecolor(REC_COLOR[rec])
    ax_hdr.set_xticks([])
    ax_hdr.set_yticks([])
    for sp in ax_hdr.spines.values():
        sp.set_visible(False)

    ax_hdr.text(0.010, 0.50, title, transform=ax_hdr.transAxes,
                fontsize=14, fontweight="bold", color="white", va="center")
    ax_hdr.text(0.50, 0.50,
                "Quadcopter Landing Site Risk Assessment  ·  Foundation-Model Perception Pipeline",
                transform=ax_hdr.transAxes, fontsize=8.5, color="white", alpha=0.92,
                ha="center", va="center")
    ax_hdr.text(0.990, 0.50, f"●  {rec}", transform=ax_hdr.transAxes,
                fontsize=12, fontweight="bold", color="white", ha="right", va="center",
                bbox=dict(facecolor="white", alpha=0.20, edgecolor="white", boxstyle="round,pad=0.35"))

    # Panel 1: original
    ax_o.imshow(image)
    _frame(ax_o)
    ax_o.set_title("①  Original Image", fontsize=10.5, fontweight="bold", color=INK, loc="left", pad=6)

    # Panel 2: detections + masks
    ax_d.imshow(image, alpha=0.72)
    cmap20 = plt.cm.tab20(np.linspace(0, 1, max(1, len(detections))))
    for det, c in zip(detections, cmap20):
        if "mask" in det:
            ov = np.zeros((H, W, 4))
            ov[det["mask"]] = [*c[:3], 0.45]
            ax_d.imshow(ov)
        x1, y1, x2, y2 = det["box"]
        ax_d.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                       linewidth=1.4, edgecolor=c, facecolor="none", zorder=3))
        ax_d.text(x1 + 2, max(y1 + 11, 10), f"{det['label']} {det['score']:.2f}",
                  color="white", fontsize=7, fontweight="bold",
                  bbox=dict(facecolor=c, edgecolor="none", alpha=0.88, pad=1.2,
                            boxstyle="round,pad=0.18"), zorder=4)
    _frame(ax_d)
    ax_d.set_title("②  Obstructions Detected & Segmented", fontsize=10.5, fontweight="bold",
                   color=INK, loc="left", pad=6)

    # Panel 3: landing geometry
    ax_g.imshow(image, alpha=0.55)
    ov_obs = np.zeros((H, W, 4))
    ov_obs[assessment["obstruction_mask"]] = [0.85, 0.18, 0.13, 0.40]
    ax_g.imshow(ov_obs, zorder=2)

    dmap = assessment["distance_map"]
    masked = np.ma.masked_where(dmap < 1, dmap)
    ax_g.imshow(masked, cmap="RdYlGn", alpha=0.32, zorder=1)

    caption = "Heatmap = distance to nearest obstacle  ·  green = far/safer    red = near/unsafe"
    ax_g.text(8, 18, caption, fontsize=8, color=INK, fontweight="bold", va="top", ha="left", zorder=10,
              bbox=dict(facecolor="white", alpha=0.92, edgecolor=RULE, boxstyle="round,pad=0.35", linewidth=0.8))

    for s in assessment.get("all_candidates_scored", []):
        cx, cy, cr = s["candidate"]
        is_best = (assessment["best_candidate"] is not None
                   and (cx, cy) == assessment["best_candidate"][:2])
        if not is_best:
            ax_g.add_patch(Circle((cx, cy), max(cr, 8), linewidth=1.0, edgecolor="white",
                           facecolor="none", alpha=0.40, linestyle="--", zorder=3))

    if assessment["best_candidate"] is not None:
        bx, by, max_r = assessment["best_candidate"]
        dc = REC_COLOR[rec]
        ax_g.add_patch(Circle((bx, by), max_r, linewidth=0, facecolor=dc, alpha=0.16, zorder=4))
        ax_g.add_patch(Circle((bx, by), max_r, linewidth=2.6, edgecolor=dc, facecolor="none", zorder=5))
        ax_g.add_patch(Circle((bx, by), veh_r, linewidth=1.6, edgecolor="white", facecolor="none",
                       linestyle="--", alpha=0.95, zorder=6))
        ax_g.add_patch(Circle((bx, by), veh_r_fos, linewidth=1.6, edgecolor="#FACC15", facecolor="none",
                       linestyle=":", alpha=0.95, zorder=6))
        ax_g.plot(bx, by, "+", markersize=22, mew=2.6, color=dc, zorder=7)
        title3 = (f"③  Optimal Landing  ·  free-disk = {max_r:.0f} px  "
                  f"·  needs {veh_r_fos} px (FoS {factor_of_safety}×)")
    else:
        title3 = "③  No Valid Landing Site Found"

    _frame(ax_g)
    ax_g.set_title(title3, fontsize=10.5, fontweight="bold", color=INK, loc="left", pad=6)

    # Panel 4: info column
    ax_i.set_facecolor("white")
    ax_i.set_xlim(0, 1)
    ax_i.set_ylim(0, 1)
    ax_i.set_xticks([])
    ax_i.set_yticks([])
    for sp in ax_i.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor(RULE)
        sp.set_linewidth(1.0)
    t = ax_i.transAxes

    ax_i.add_patch(FancyBboxPatch((0.05, 0.905), 0.90, 0.075, boxstyle="round,pad=0.01",
                   facecolor=REC_TINT[rec], edgecolor=REC_COLOR[rec], linewidth=2.0, transform=t, zorder=2))
    ax_i.text(0.50, 0.943, f"●  {rec}", transform=t, fontsize=16, fontweight="bold",
              ha="center", va="center", color=REC_COLOR[rec], zorder=3)

    ax_i.text(0.50, 0.870, "SAFETY  SCORE", transform=t, fontsize=7, ha="center",
              color=INK_LIGHT, fontweight="bold")
    ax_i.text(0.50, 0.821, f"{safety:.0f}", transform=t, fontsize=32, fontweight="bold",
              ha="center", va="center", color=INK)
    ax_i.text(0.50, 0.785, f"out of 100   ·   risk {risk:.0f}/100", transform=t, fontsize=8,
              ha="center", color=INK_MID)

    ax_i.plot([0.05, 0.95], [0.770, 0.770], color=RULE, lw=1.0, transform=t)

    ax_i.add_patch(FancyBboxPatch((0.08, 0.736), 0.84, 0.026, boxstyle="round,pad=0.005",
                   facecolor="#FEF9C3", edgecolor="#FACC15", linewidth=1.0, transform=t, zorder=2))
    ax_i.text(0.50, 0.749, f"⚙  Factor of Safety  {factor_of_safety}×    effective req. = {veh_r_fos} px",
              transform=t, fontsize=7.8, ha="center", va="center", color="#713F12", fontweight="bold")

    ax_i.plot([0.05, 0.95], [0.722, 0.722], color=RULE, lw=1.0, transform=t)

    ax_i.text(0.06, 0.700, "SUB-SCORES", transform=t, fontsize=7.5, fontweight="bold", color=INK_LIGHT)

    sub_items = [
        ("Free Space", "free_space_ratio", weights["freespace"]),
        ("Free Disk", "free_disk_adequacy", weights["freedisk"]),
        ("Hazard Safety", "hazard_safety", weights["hazard"]),
        ("Surface", "surface_suitability", weights["surface"]),
    ]
    y = 0.674
    for label, key, wt in sub_items:
        v = sub[key]
        bc = _sub_color(v)
        ax_i.text(0.06, y + 0.005, label, transform=t, fontsize=8.5, color=INK)
        ax_i.text(0.94, y + 0.005, f"{v:.2f}", transform=t, fontsize=8.5, ha="right",
                  fontweight="bold", color=bc)
        _bar(ax_i, 0.06, y - 0.014, 0.88, 0.013, v, bc)
        ax_i.text(0.94, y - 0.030, f"weight {wt:.2f}", transform=t, fontsize=6.5, ha="right", color=INK_LIGHT)
        y -= 0.060

    ax_i.plot([0.05, 0.95], [y + 0.020, y + 0.020], color=RULE, lw=1.0, transform=t)

    y_r = y + 0.005
    ax_i.text(0.06, y_r, "RATIONALE", transform=t, fontsize=7.5, fontweight="bold", color=INK_LIGHT)
    rationale_lines = _wrap(rationale, width=44)[:4]
    for i, ln in enumerate(rationale_lines):
        ax_i.text(0.06, y_r - 0.026 - i * 0.024, ln, transform=t, fontsize=7.8, color=INK_MID)

    y_h = y_r - 0.034 - len(rationale_lines) * 0.024
    ax_i.plot([0.05, 0.95], [y_h + 0.004, y_h + 0.004], color=RULE, lw=1.0, transform=t)

    y_h2 = y_h - 0.013
    ax_i.text(0.06, y_h2, f"NEARBY  HAZARDS  ({len(nearby)})", transform=t, fontsize=7.5,
              fontweight="bold", color=INK_LIGHT)

    def _sc(s):
        if s >= 0.85:
            return REC_COLOR["UNSAFE"]
        if s >= 0.55:
            return REC_COLOR["CAUTION"]
        return INK_LIGHT

    for i, h in enumerate(nearby[:5]):
        yh = y_h2 - 0.026 - i * 0.030
        if yh < 0.018:
            break
        sc = _sc(h["severity"])
        ax_i.text(0.06, yh, f"▸ {h['label']}", transform=t, fontsize=8, color=INK)
        _bar(ax_i, 0.55, yh - 0.005, 0.27, 0.011, h["severity"], sc)
        ax_i.text(0.94, yh, f"{h['severity']:.2f}", transform=t, fontsize=8, ha="right",
                  fontweight="bold", color=sc)

    if save_path:
        fig.savefig(save_path, dpi=160, bbox_inches="tight", facecolor="white", pad_inches=0.05)
        print(f"Saved: {save_path}")
    plt.show()
    plt.close(fig)
