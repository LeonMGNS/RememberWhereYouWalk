r"""
Colour the maze at runtime, with no texture file on disk.

MuJoCo has no per-vertex or height-dependent colouring for heightfields: a geom
carries one material. So the colour comes from a 2D texture mapped across the
hfield's XY extent, whose bytes are filled in memory from model.hfield_data.

The texture is declared at exactly the heightfield resolution and written in the
same flat order as hfield_data, texel i taking its colour from height i. No
reshape, no transpose, no upsampling -- so there is no ordering convention here
that can disagree with whatever the terrain generator wrote.

    <hfield name="empty_hfield" file="..." size="5 5 30 2"/>        <- 100 x 100
    <texture name="maze_tex" type="2d" builtin="flat" rgb1="1 0 1"
             width="100" height="100"/>                             <- must match

Wall edges are then exactly as coarse as the collision geometry, which is the
point: the colour boundary lands on the same grid line as the wall itself.
"""

import numpy as np
import mujoco

# flip_rows is the default: a texture's first row is its TOP, while hfield row 0
# sits at minimum y. That is a pure vertical flip and is a MuJoCo convention, so
# it holds for any maze resolution -- not something to re-derive per model.
# The eight ways a square grid can map onto a texture. The 1:1 write removes any
# reordering *this module* does, but it cannot remove MuJoCo's own convention
# for which corner of the hfield the first texel lands on. A maze boundary frame
# is invariant under all eight, so a correct-looking border proves nothing.
ORIENTS = {
    "identity":       lambda m: m,
    "flip_rows":      lambda m: m[::-1],
    "flip_cols":      lambda m: m[:, ::-1],
    "rot180":         lambda m: m[::-1, ::-1],
    "transpose":      lambda m: m.T,
    "rot90":          lambda m: np.rot90(m),
    "rot270":         lambda m: np.rot90(m, 3),
    "anti_transpose": lambda m: np.rot90(m, 2).T,
}

# seaborn "colorblind" palette
WALL_RGB = (222, 143, 5)       # #de8f05 orange
FLOOR_RGB = (85, 85, 85)       # #555555 grey

# Alternatives from the same palette:
#   sky blue (86, 180, 233) | green (2, 158, 115) | yellow (236, 225, 51)


class MazePainter:
    """Paints model.tex_data from model.hfield_data, one texel per height."""

    def __init__(self, model, hfield_name="empty_hfield", tex_name="maze_tex",
                 rel_threshold=0.5, wall_rgb=WALL_RGB, floor_rgb=FLOOR_RGB,
                 orient="flip_rows"):
        self.tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXTURE, tex_name)
        self.hid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, hfield_name)
        if self.tid < 0:
            raise ValueError(
                f"no texture named {tex_name!r}. Add "
                f"<texture name='{tex_name}' type='2d' builtin='flat' "
                f"rgb1='1 0 1' width='NCOL' height='NROW'/> to the model."
            )
        if self.hid < 0:
            raise ValueError(f"no hfield named {hfield_name!r} in the model")

        self.nrow = int(model.hfield_nrow[self.hid])
        self.ncol = int(model.hfield_ncol[self.hid])
        self.hadr = int(model.hfield_adr[self.hid])
        self.H = int(model.tex_height[self.tid])
        self.W = int(model.tex_width[self.tid])
        self.nch = int(model.tex_nchannel[self.tid]) if hasattr(model, "tex_nchannel") else 3
        self.tadr = int(model.tex_adr[self.tid])

        self.n = self.nrow * self.ncol
        if self.H * self.W != self.n:
            raise ValueError(
                f"texture is {self.W}x{self.H} ({self.H * self.W} texels) but the "
                f"heightfield is {self.ncol}x{self.nrow} ({self.n} cells). For the "
                f"1:1 mapping they must match -- set width='{self.ncol}' "
                f"height='{self.nrow}' on <texture name='{tex_name}'/>."
            )

        if orient not in ORIENTS:
            raise ValueError(f"orient must be one of {sorted(ORIENTS)}")
        self.orient = orient
        self.rel_threshold = rel_threshold
        self.wall_rgb, self.floor_rgb = wall_rgb, floor_rgb
        self._last = None
        self._warned = False

    # -- mask ---------------------------------------------------------------
    def wall_flat(self, model):
        """Flat boolean wall mask, same order and length as hfield_data."""
        h = np.asarray(model.hfield_data[self.hadr:self.hadr + self.n])
        hmax = float(h.max()) if h.size else 0.0
        if hmax <= 0.0:
            return np.zeros(self.n, bool)
        wall = h > (self.rel_threshold * hmax)
        if self.orient != "identity":
            wall = ORIENTS[self.orient](wall.reshape(self.nrow, self.ncol)).ravel()
        return wall

    # -- painting -----------------------------------------------------------
    def _write(self, model, wall_flat):
        buf = np.empty((self.n, self.nch), np.uint8)
        buf[:, :3] = np.where(np.asarray(wall_flat, bool)[:, None],
                              np.array(self.wall_rgb), np.array(self.floor_rgb))
        if self.nch == 4:
            buf[:, 3] = 255
        model.tex_data[self.tadr:self.tadr + self.n * self.nch] = buf.ravel()

    def _upload(self, model, renderer=None, viewer=None):
        # Without this the new bytes exist only CPU-side and the render still
        # shows the previous maze.
        if viewer is not None:
            viewer.update_texture(self.tid)
        if renderer is not None:
            gl = getattr(renderer, "_gl_context", None)
            if gl is not None:
                gl.make_current()
            mujoco.mjr_uploadTexture(model, renderer._mjr_context, self.tid)

    def paint(self, model, renderer=None, viewer=None, force=False):
        """Repaint if the heightfield changed. Returns True if it repainted."""
        wall = self.wall_flat(model)
        if not self._warned and wall.sum() == 0:
            h = np.asarray(model.hfield_data[self.hadr:self.hadr + self.n])
            print(f"** MazePainter: wall mask is EMPTY -> everything paints as floor. "
                  f"hfield min/max = {h.min():.6g}/{h.max():.6g}. If max <= 0 the "
                  f"heightfield reaching this model has no maze in it.")
            self._warned = True

        key = wall.tobytes()
        if not force and key == self._last:
            return False
        self._last = key
        self._write(model, wall)
        self._upload(model, renderer, viewer)
        return True

    def paint_values(self, model, values, wall_rgb=None, floor_rgb=None,
                     renderer=None, viewer=None):
        """Per-cell colour interpolation: 0 -> floor colour, 1 -> wall colour.

        values must hold nrow*ncol numbers in hfield_data order (same flat
        order as paint()/wall masks); they are routed through the painter's
        orientation setting, so no separate flipping is needed. Used to shade
        predicted walls by certainty instead of thresholding.
        """
        v = np.clip(np.asarray(values, float).reshape(self.nrow, self.ncol),
                    0.0, 1.0)
        v = ORIENTS[self.orient](v).ravel()
        w = np.asarray(wall_rgb if wall_rgb is not None else self.wall_rgb,
                       float)
        f = np.asarray(floor_rgb if floor_rgb is not None else self.floor_rgb,
                       float)
        buf = np.zeros((self.n, self.nch), np.uint8)
        buf[:, :3] = np.round(f[None, :] + v[:, None] * (w - f)[None, :])
        if self.nch == 4:
            buf[:, 3] = 255
        model.tex_data[self.tadr:self.tadr + self.n * self.nch] = buf.ravel()
        self._upload(model, renderer, viewer)
        self._last = None
        return True


