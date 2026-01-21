#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import subprocess
import csv
import time
import json
import os
import logging
import sys
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

fm.fontManager.addfont('./times.ttf')

ts = time.strftime("%Y-%m-%d-%H-%M-%S")
SCRIPT = "comm_cifar100_resnet34_split_new.py"
EPOCHS = 100
BATCH = 128
NC_LIST = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]  # list(range(1, 11))
SNR_LIST = [0, 3, 5]
LOG_PATH = f"plot_{ts}.log"
# RESULT_CSV = f"results_{ts}.csv"
RESULT_CSV = "results_2025-07-28-12-06-01.csv"
FIG_PATH = f"cifar10_resnet18_split.pdf"


def plot_results():
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "mathtext.fontset": "cm",
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })

    df = pd.read_csv(RESULT_CSV)
    # fig, ax = plt.subplots(figsize=(5, 4.7))
    fig, ax = plt.subplots(figsize=(5, 5.3))

    markers = {0: "s", 3: "^", 5: "o"}
    colors = {0: "#F5C267", 3: "#A6D6A6", 5: "#5AB2FF"}

    for snr in SNR_LIST:
        sub = df[df.snr_db == snr].sort_values("nc")
        ax.plot(sub.nc, sub.best_acc,
                marker=markers[snr], color=colors[snr],
                label=f"SNR = {snr} dB",
                solid_capstyle='round')

    ax.set_xlabel(r"Dimensionality of Semantic Representation $z$")
    ax.set_ylabel(r"Top-1 Accuracy")

    ax.text(
        0.5, -0.16,
        "(a) CIFAR-10 / ResNet-18 (conv2_x-conv3_x)",
        transform=ax.transAxes,
        ha='center', va='top',
        fontsize=15,
        fontweight='normal',
        fontfamily='Times New Roman'
    )

    # ax.set_ylim(0, 0.8)
    ax.set_ylim(0.05, 0.95)
    ax.set_xscale("log", base=2)
    ax.set_xticks(NC_LIST)
    # ax.set_xlim(0.5, 10.5)
    ax.set_xlim(min(NC_LIST) * 0.8, max(NC_LIST) * 1.2)

    ax.tick_params(
        which='both', direction='in',
        top=False, bottom=False,
        left=False, right=False,
        color='gray', pad=6
    )

    ax.grid(True,
            which="both",
            linestyle=":",
            color="gray"
            )

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=3,
        frameon=False,
        columnspacing=1,
        # borderaxespad=0.08
        borderaxespad=0.3
    )

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    fig.tight_layout()
    # fig.savefig(FIG_PATH, dpi=600)
    fig.savefig(FIG_PATH, format="pdf", bbox_inches="tight")


def main():
    plot_results()


if __name__ == "__main__":
    main()
