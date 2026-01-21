#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
This script plots the relationship between SNR (dB) and the normalized communication time
and saves the figure in both PDF and PNG formats.

The plot uses Times New Roman font with custom sizing for titles, labels, ticks, and legend.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import math
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

fm.fontManager.addfont('./times.ttf')

plt.rcParams.update({
    "font.family": "Times New Roman",
    "mathtext.fontset": "cm",
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 19,
    "legend.fontsize": 17,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
})

# -------- Calculate Normalized Communication Time --------
result_snr0db_sp2 = 1024 / (3 * 32 * 32)
result_snr0db_sp3 = 64 / (3 * 32 * 32)
result_snr0db_sp4 = 32 / (3 * 32 * 32)

result_snr3db_sp2 = 512 / (3 * 32 * 32)
result_snr3db_sp3 = 32 / (3 * 32 * 32)
result_snr3db_sp4 = 32 / (3 * 32 * 32)

result_snr5db_sp2 = 512 / (3 * 32 * 32)
result_snr5db_sp3 = 32 / (3 * 32 * 32)
result_snr5db_sp4 = 16 / (3 * 32 * 32)

snr_db = np.array([0, 3, 5])

sp2_values = np.array([result_snr0db_sp2, result_snr3db_sp2, result_snr5db_sp2])
sp3_values = np.array([result_snr0db_sp3, result_snr3db_sp3, result_snr5db_sp3])
sp4_values = np.array([result_snr0db_sp4, result_snr3db_sp4, result_snr5db_sp4])

fig, ax = plt.subplots(figsize=(5, 5))

# SP-2
ax.plot(snr_db, sp2_values, linestyle='--', color='#F5C267', linewidth=2)
ax.scatter(snr_db, sp2_values, color='#F5C267', s=80, marker='o', edgecolor='black', label='SP-2', zorder=5)

# SP-3
ax.plot(snr_db, sp3_values, linestyle='--', color='#A6D6A6', linewidth=2)
ax.scatter(snr_db, sp3_values, color='#A6D6A6', s=80, marker='^', edgecolor='black', label='SP-3', zorder=5)

# SP-4
ax.plot(snr_db, sp4_values, linestyle='--', color='#5AB2FF', linewidth=2)
ax.scatter(snr_db, sp4_values, color='#5AB2FF', s=80, marker='s', edgecolor='black', label='SP-4', zorder=5)

ax.set_xticks(snr_db)
ax.set_xlabel('SNR (dB)')
# ax.set_ylabel('Communication Time (ms)')  # 1.
ax.set_ylabel('Normalized Communication Time')  # 2.

for spine in ax.spines.values():
    spine.set_linewidth(1.2)
ax.tick_params(which='both', direction='in', color='gray', pad=6)
ax.grid(True, linestyle=":", color="gray", alpha=0.7)

ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1.12),
    ncol=3,
    frameon=False,
    columnspacing=1,
    borderaxespad=0.3,
    handlelength=1.5
)

fig.tight_layout()

fig.savefig("tcomm_vs_snrdb.pdf", format="pdf", bbox_inches="tight")
fig.savefig("tcomm_vs_snrdb.png", dpi=600, bbox_inches="tight")
