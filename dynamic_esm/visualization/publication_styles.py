"""
Nature / Science Publication Styling Configuration.
Ensures uniform typographic and aesthetic standards for 300 DPI raster and vector PDF exports.
"""

import matplotlib.pyplot as plt


def set_publication_style():
    """Apply Nature / Science journal graphic standards to matplotlib."""
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    plt.rcParams["font.size"] = 9
    plt.rcParams["axes.titlesize"] = 10
    plt.rcParams["axes.labelsize"] = 9.5
    plt.rcParams["xtick.labelsize"] = 8.5
    plt.rcParams["ytick.labelsize"] = 8.5
    plt.rcParams["legend.fontsize"] = 8.5
    plt.rcParams["figure.titlesize"] = 11

    plt.rcParams["axes.linewidth"] = 1.0
    plt.rcParams["lines.linewidth"] = 1.5
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["xtick.major.size"] = 4
    plt.rcParams["ytick.major.size"] = 4
    plt.rcParams["xtick.minor.visible"] = False
    plt.rcParams["ytick.minor.visible"] = False

    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["savefig.bbox"] = "tight"
