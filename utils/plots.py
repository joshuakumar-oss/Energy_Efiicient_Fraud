import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    fig.patch.set_facecolor("#fff9f0")
    ax.set_facecolor("#fffdf9")

    heatmap = ax.imshow(cm, cmap="Oranges")

    for i in range(len(cm)):
        for j in range(len(cm)):
            color = "#14213d" if cm[i, j] < np.max(cm) * 0.6 else "white"
            ax.text(
                j,
                i,
                f"{cm[i, j]}",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color=color,
            )

    ax.set_xticks([0, 1], ["Legit", "Fraud"])
    ax.set_yticks([0, 1], ["Legit", "Fraud"])
    ax.set_xlabel("Predicted Label", fontsize=11, color="#264653")
    ax.set_ylabel("Actual Label", fontsize=11, color="#264653")
    ax.set_title("Fraud Detection Confusion Matrix", fontsize=14, weight="bold", color="#14213d", pad=14)

    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(colors="#5f6b7a")

    fig.tight_layout()
    return fig
