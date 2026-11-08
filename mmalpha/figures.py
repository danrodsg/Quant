"""Gráficos do README (figures/)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mmalpha.alpha import LOW_VOL, RESULTS, load

FIG = Path("figures")


def br(x: float, fmt: str = ".1f") -> str:
    return format(x, fmt).replace(".", ",")

BLUE, ORANGE = "#2a78d6", "#eb6834"
GRAY, GRAY_FILL = "#8a8984", "#e4e3df"
INK, INK_2 = "#0b0b0b", "#52514e"
BG = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
    "font.size": 10,
    "axes.edgecolor": "#c9c8c3",
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": "#ebeae6",
    "grid.linewidth": 0.8,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "legend.frameon": False,
    "lines.linewidth": 2,
})


def luck_vs_skill() -> None:
    lvs = pd.read_csv(RESULTS / "luck_vs_skill.csv")
    panels = [("directional", "Fundos direcionais"), ("cdi_plus", f"Fundos CDI+ (vol < {LOW_VOL:g}% a.a.)")]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, (group, title) in zip(axes, panels):
        d = lvs[lvs["group"] == group]
        ax.fill_between(d["percentile"], d["t_luck_p5"], d["t_luck_p95"],
                        color=GRAY_FILL, lw=0, label="só sorte (5%–95% das simulações)")
        ax.plot(d["percentile"], d["t_luck_median"], color=GRAY, lw=1.5, ls="--",
                label="só sorte (mediana)")
        ax.plot(d["percentile"], d["t_actual"], color=BLUE, label="real")
        ax.axhline(0, color="#c9c8c3", lw=0.8)
        ax.set_title(title)
        ax.set_xlabel("percentil dos fundos")
        ax.set_xticks([1, 25, 50, 75, 99])
        ax.set_xlim(1, 99)

        top = d.iloc[-1]
        ax.annotate(f"top 1%: t = {br(top['t_actual'])}", (99, top["t_actual"]),
                    xytext=(-6, 0), textcoords="offset points", ha="right", va="center",
                    color=INK, fontsize=9)

    axes[0].set_ylabel("t-stat do alpha")
    axes[0].legend(loc="upper left", fontsize=9)
    fig.suptitle("Os melhores fundos direcionais estão dentro do que a sorte explica",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "luck_vs_skill.png", dpi=200)
    plt.close(fig)


def survivorship() -> None:
    excess, _, funds = load()
    alive = funds.index[funds["alive"]].intersection(excess.columns)
    series = {
        "todos os fundos": (excess.mean(axis=1), BLUE),
        "só os que sobreviveram": (excess[alive].mean(axis=1), ORANGE),
    }

    fig, ax = plt.subplots(figsize=(8, 4.2))
    for label, (r, color) in series.items():
        cum = ((1 + r.fillna(0)).cumprod() - 1) * 100
        x = cum.index.to_timestamp(how="end")
        ax.plot(x, cum, color=color, label=label)
        ax.annotate(f"{label}\n{br(cum.iloc[-1], '+.1f')}%", (x[-1], cum.iloc[-1]),
                    xytext=(6, 0), textcoords="offset points", va="center",
                    color=INK, fontsize=9)

    ax.axhline(0, color="#c9c8c3", lw=0.8)
    ax.set_ylabel("retorno acumulado acima do CDI (%)")
    ax.set_title("Ignorar os fundos que fecharam infla o retorno em ~0,6 p.p. ao ano")
    ax.legend(loc="upper left", fontsize=9)
    ax.margins(x=0)
    fig.tight_layout()
    fig.subplots_adjust(right=0.80)
    fig.savefig(FIG / "survivorship.png", dpi=200)
    plt.close(fig)


def tstat_vs_vol() -> None:
    a = pd.read_csv(RESULTS / "alphas.csv")
    groups = [("directional", "direcionais", BLUE), ("cdi_plus", "CDI+", ORANGE)]

    fig, ax = plt.subplots(figsize=(8, 4.6))
    for group, label, color in groups:
        d = a[a["group"] == group]
        ax.scatter(d["vol_pct_year"], d["t_alpha"], s=12, color=color, alpha=0.6,
                   edgecolors=BG, linewidths=0.4, label=f"{label} ({len(d)})")

    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 2, 5, 10, 20, 50])
    ax.set_xticklabels(["0,5", "1", "2", "5", "10", "20", "50"])
    for t in (-2, 2):
        ax.axhline(t, color=GRAY, lw=0.8, ls="--")
    ax.axvline(LOW_VOL, color=GRAY, lw=0.8)
    ax.text(LOW_VOL * 1.05, ax.get_ylim()[1] * 0.95, f"{LOW_VOL:g}% a.a.",
            color=INK_2, fontsize=9, va="top")

    ax.set_xlabel("volatilidade do retorno acima do CDI (% a.a., escala log)")
    ax.set_ylabel("t-stat do alpha")
    ax.set_title("Os t-stats altos vêm quase só dos fundos de baixa volatilidade")
    ax.legend(loc="upper right", fontsize=9, markerscale=1.5)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "tstat_vs_vol.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    luck_vs_skill()
    survivorship()
    tstat_vs_vol()
