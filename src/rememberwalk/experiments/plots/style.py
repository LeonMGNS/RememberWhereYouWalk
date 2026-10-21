r"""Shared figure style for the experiment plots.

Sized for the ieeeconf.cls / IEEEtran.cls two-column layout:
    \textwidth = 7.0in, \columnsep = 0.2in  ->  column width = 3.4in
Font sizes taken from the 10pt branch of the class file:
    \normalsize 10pt | \small 9pt | \footnotesize 8pt (captions) | \scriptsize 7pt

Insert with \includegraphics{fig.pdf} -- NO width= scaling, so the point
sizes below are exactly the point sizes on the printed page.

Approach colours are not defined here: every figure takes them from the suite file,
so all figures and the video use the same colour per approach.
"""

# --------------------------------------------------------------------------
# Page geometry from ieeeconf.cls
# --------------------------------------------------------------------------
TEXTWIDTH_IN = 7.0                                  # \textwidth
COLUMNSEP_IN = 0.2                                  # \columnsep
COLWIDTH_IN = (TEXTWIDTH_IN - COLUMNSEP_IN) / 2.0   # = 3.4in, single column
FIGWIDTH_IN = COLWIDTH_IN - 0.02                    # hair under, avoids overfull hbox

# Font sizes (pt) from the class file's 10pt branch
FS_NORMAL, FS_SMALL, FS_FOOTNOTE, FS_SCRIPT = 10, 9,8,8

# Control frequency of the sim: one step = CONTROL_DT seconds
CONTROL_FREQ_HZ = 50
CONTROL_DT = 1.0 / CONTROL_FREQ_HZ

# Policies within an approach are separated by dash pattern only.
POLICY_LINESTYLES = [
    (0, ()),            # solid
    (0, (4, 1.5)),      # dashed
    (0, (1, 1.2)),      # dotted
]

RC = {
    "font.family": "serif",
    # IEEE uses Times (ptm). Falls back to DejaVu Serif if Times is missing.
    "font.serif": ["Times New Roman", "Nimbus Roman No9 L", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": FS_FOOTNOTE,
    "axes.labelsize": FS_FOOTNOTE,
    "axes.titlesize": FS_FOOTNOTE,
    "xtick.labelsize": FS_SCRIPT,
    "ytick.labelsize": FS_SCRIPT,
    "legend.fontsize": FS_SCRIPT,
    "axes.linewidth": 0.5,
    "grid.linewidth": 0.4,
    "lines.linewidth": 0.9,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.0,
    "ytick.major.size": 2.0,
    "xtick.major.pad": 1.5,
    "ytick.major.pad": 1.5,
    "axes.labelpad": 1.5,
    "legend.frameon": False,
    "legend.handlelength": 1.8,
    "legend.handletextpad": 0.4,
    "legend.columnspacing": 1.0,
    "legend.borderaxespad": 0.0,
    "pdf.fonttype": 42,   # embed TrueType; keeps text selectable/searchable
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.005,
}


def png_path(out_path) -> str:
    """Every figure is written as .pdf and as a .png preview next to it."""
    return str(out_path).replace(".pdf", ".png")
