"""Generates grating_diagram.png — angle sign convention illustration."""

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

BG = "#0e1117"
TEXT = "#e5e2e1"
MUTED = "#bbcac1"
ACCENT = "#42e5b0"
YELLOW = "#FFD10F"
ORANGE = "#FF8C42"
PURPLE = "#c084fc"

fig, ax = plt.subplots(figsize=(14, 8), facecolor=BG)
ax.set_facecolor(BG)
ax.set_xlim(-5.5, 5.5)
ax.set_ylim(-1.3, 5.3)
ax.set_aspect("equal")
ax.axis("off")

alpha_deg = 80
alpha_rad = np.radians(alpha_deg)
L = 3.6  # ray length

# ── Grating surface ──────────────────────────────────────────────────────────
ax.plot(
    [-5, 5], [0, 0], color=ACCENT, lw=3.5, solid_capstyle="round", zorder=5
)
for x in np.linspace(-4.8, 4.8, 40):
    ax.plot([x, x - 0.06], [0, -0.22], color=ACCENT, lw=0.8, alpha=0.4)
ax.text(4.9, -0.75, "grating", color=ACCENT, fontsize=11, ha="right")

# ── Normal ───────────────────────────────────────────────────────────────────
ax.plot([0, 0], [-0.8, 4.9], color=MUTED, lw=1.2, linestyle="--", alpha=0.6)
ax.annotate(
    "",
    xy=(0, 4.9),
    xytext=(0, 4.5),
    arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2),
)
ax.text(0.13, 4.75, "normal", color=MUTED, fontsize=11)

# ── Incident beam ─────────────────────────────────────────────────────────────
sx, sy = -np.sin(alpha_rad) * L, np.cos(alpha_rad) * L
ax.annotate(
    "",
    xy=(0, 0),
    xytext=(sx, sy),
    arrowprops=dict(arrowstyle="->", color=TEXT, lw=2.5, mutation_scale=18),
)
ax.text(
    sx - 0.2,
    sy + 0.15,
    "incident\nbeam",
    color=TEXT,
    fontsize=12,
    ha="right",
    fontweight="bold",
)

# ── α arc ────────────────────────────────────────────────────────────────────
r_a = 1.0
ax.add_patch(
    patches.Arc(
        (0, 0),
        r_a * 2,
        r_a * 2,
        angle=0,
        theta1=90,
        theta2=90 + alpha_deg,
        color=TEXT,
        lw=1.8,
    )
)
mid_a = np.radians(90 + alpha_deg / 2)
ax.text(
    np.cos(mid_a) * (r_a + 0.3),
    np.sin(mid_a) * (r_a + 0.3),
    "α = 45°",
    color=TEXT,
    fontsize=13,
    fontweight="bold",
    ha="center",
    va="center",
)

# ── Diffracted orders ─────────────────────────────────────────────────────────
# 2400 l/mm, λ = 650 nm, α = 45°  →  d = 416.7 nm
# Beam endpoint: (−sin β · L,  cos β · L)
#   β > 0  →  x < 0  (LEFT,  same side as incident)
#   β < 0  →  x > 0  (RIGHT, opposite side)
#
# m =  0 :  β = −45.0°  (specular)
# m = +1 :  β = +58.6°  (only valid diffracted order)
# m = −1, m = +2 : evanescent  (|sin β| > 1)
orders = [
    (0, -45.0, MUTED, "right", "m = 0"),
    (+1, 58.6, ORANGE, "left", "m = +1"),
]

for m, beta_deg, color, side, label in orders:
    beta_rad = np.radians(beta_deg)
    ex, ey = -np.sin(beta_rad) * L, np.cos(beta_rad) * L
    ax.annotate(
        "",
        xy=(ex, ey),
        xytext=(0, 0),
        arrowprops=dict(
            arrowstyle="->", color=color, lw=2.2, mutation_scale=16
        ),
    )
    dx = 0.2 if side == "right" else -0.2
    ha = "left" if side == "right" else "right"
    ax.text(
        ex + dx,
        ey + 0.13,
        label,
        color=color,
        fontsize=12,
        ha=ha,
        fontweight="bold",
    )
    if m == 0:
        ax.text(
            ex + dx,
            ey - 0.32,
            "specular  (β = −α)",
            color=color,
            fontsize=9,
            ha=ha,
            alpha=0.85,
        )
    if m == 1:
        ax.text(
            ex + dx,
            ey - 0.32,
            "β = +58.6°",
            color=color,
            fontsize=9,
            ha=ha,
            alpha=0.85,
        )

# ── evanescent order note ─────────────────────────────────────────────────────
ax.text(
    4.8,
    1.8,
    "m = −1\nm = +2\n(evanescent)",
    color=MUTED,
    fontsize=9,
    ha="right",
    alpha=0.6,
    style="italic",
)

# ── β arc for m = +1  (positive, left side) ──────────────────────────────────
r_b = 1.65
beta_p = 58.6
ax.add_patch(
    patches.Arc(
        (0, 0),
        r_b * 2,
        r_b * 2,
        angle=0,
        theta1=90,
        theta2=90 + beta_p,
        color=ORANGE,
        lw=1.7,
    )
)
mid_bp = np.radians(90 + beta_p / 2)
ax.text(
    np.cos(mid_bp) * (r_b + 0.35),
    np.sin(mid_bp) * (r_b + 0.35),
    "β",
    color=ORANGE,
    fontsize=17,
    fontweight="bold",
    ha="center",
    va="center",
)

# ── β arc for m = 0  (negative, right side) ──────────────────────────────────
beta_n = -45.0
ax.add_patch(
    patches.Arc(
        (0, 0),
        r_b * 2,
        r_b * 2,
        angle=0,
        theta1=90 + beta_n,
        theta2=90,
        color=MUTED,
        lw=1.7,
    )
)
mid_bn = np.radians(90 + beta_n / 2)
ax.text(
    np.cos(mid_bn) * (r_b + 0.35),
    np.sin(mid_bn) * (r_b + 0.35),
    "β",
    color=MUTED,
    fontsize=17,
    fontweight="bold",
    ha="center",
    va="center",
)

# ── Side legend ──────────────────────────────────────────────────────────────
ax.text(-4.8, 4.85, "β > 0", color=ORANGE, fontsize=12, fontweight="bold")
ax.text(-4.8, 4.45, "same side as incident", color=MUTED, fontsize=9)
ax.text(
    4.8, 4.85, "β < 0", color=MUTED, fontsize=12, fontweight="bold", ha="right"
)
ax.text(4.8, 4.45, "opposite side", color=MUTED, fontsize=9, ha="right")

# ── Equation ─────────────────────────────────────────────────────────────────
ax.text(
    0,
    -1.0,
    r"$m\lambda = d\,(\sin\alpha + \sin\beta)$"
    r"          2400 gr/mm · λ = 650 nm · α = 45°",
    color=MUTED,
    fontsize=11,
    ha="center",
)

# ── Title ─────────────────────────────────────────────────────────────────────
ax.text(
    0,
    5.15,
    "Reflection Grating — Angle Sign Convention",
    color=TEXT,
    fontsize=14,
    fontweight="bold",
    ha="center",
)

plt.tight_layout(pad=0.5)
plt.savefig("grating_diagram.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("Saved grating_diagram.png")
