"""
Visualization and publication-grade figure generation subpackage.
"""

from dynamic_esm.visualization.publication_styles import set_publication_style
from dynamic_esm.visualization.plot_fig1 import generate_figure_1
from dynamic_esm.visualization.plot_fig2_fig3 import plot_figure_2, plot_figure_3
from dynamic_esm.visualization.plot_fig4 import plot_figure_4
from dynamic_esm.visualization.plot_fig5 import plot_figure_5

__all__ = [
    "set_publication_style",
    "generate_figure_1",
    "plot_figure_2",
    "plot_figure_3",
    "plot_figure_4",
    "plot_figure_5",
]
