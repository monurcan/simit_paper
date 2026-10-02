import os as _os
from pathlib import Path as _Path
SIMIT_ROOT = _os.environ.get("SIMIT_ROOT") or str(_Path(__file__).resolve().parents[1])
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import copy as cp

# old paper values:
# color_palette_primary = [
#     "#3B6BA5",  # Primary Blue
#     "#5DA5DA",  # Soft Sky Blue
#     "#60BD68",  # Primary Green
#     "#B5BD61",  # Lime-Yellow
#     "#FAA43A",  # Primary Orange
#     "#FDD262",  # Golden Yellow
#     "#B276B2",  # Violet
#     "#9C9EDE",  # Soft Indigo
#     "#F17CB0",  # Magenta
#     "#8C8C8C"   # Neutral Gray
# ]

color_palette_primary = {
    "Color 1": "#277DA1",
    "Color 1+": "#196382",
    "Color 2": "#577590",
    "Color 2+": "#425E78",
    "Color 3": "#4D908E",
    "Color 3+": "#377472",
    "Color 4": "#43AA8B",
    "Color 4+": "#2C8D70",
    "Color 5": "#90BE6D",
    "Color 5+": "#789F5A",
    "Color 6": "#F9C74F",
    "Color 6+": "#B48B28",
    "Color 7": "#F9844A",
    "Color 7+": "#AF5F37",
    "Color 8": "#F8961E",
    "Color 8+": "#B76D11",
    "Color 9": "#F3722C",
    "Color 9+": "#C56029",
    "Color 10": "#F94144",
    "Color 10+": "#BD2E30",
    "Color 11": "#757575",
}

global_alpha = 0.8
global_grid_alpha = 0.3

global_font = "Fira Sans"
rcParams['font.family'] = global_font

global_fontsize = 18
global_fontsize_small = 15

save_path = SIMIT_ROOT + "/visualization_scripts/figures/"

