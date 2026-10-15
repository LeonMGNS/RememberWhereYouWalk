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
# for which corner of the hfield the first texel lands on. Run calibrate() once
# to find out which of these you need; a maze boundary frame is invariant under
# all eight, so a correct-looking border proves nothing.
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

# Distinct palette colours for the calibration render
CAL_COLORS = {
    "A": (222, 143, 5),     # orange
    "B": (86, 180, 233),    # sky blue
    "C": (2, 158, 115),     # green
    "D": (213, 94, 0),      # vermillion
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

    def paint_mask(self, model, mask, renderer=None, viewer=None):
        """Paint from an explicit wall mask, when the height threshold is
        unreliable (e.g. strong terrain roughness). Any shape is accepted as
        long as it holds nrow*ncol values in hfield_data order."""
        self._write(model, np.asarray(mask, bool).ravel())
        self._upload(model, renderer, viewer)
        self._last = None
        return True

    # -- diagnostics --------------------------------------------------------
    def diagnose(self, model, geom_name="floor"):
        """Print every link between hfield_data and the pixels on screen."""
        h = np.asarray(model.hfield_data[self.hadr:self.hadr + self.n])
        print("=== MazePainter diagnose ===")
        print(f"hfield {self.nrow}x{self.ncol} adr={self.hadr}  "
              f"min={h.min():.6g} max={h.max():.6g} nonzero={int((h != 0).sum())}")
        if h.max() <= 0:
            print("  !! max <= 0: no maze in model.hfield_data. The paint call runs "
                  "before the terrain is copied in, or on a different model object.")
        else:
            print(f"  mask: {self.wall_flat(model).mean():.1%} of cells are wall "
                  f"(threshold {self.rel_threshold * h.max():.6g})")
        print(f"texture id={self.tid} {self.W}x{self.H} nch={self.nch} adr={self.tadr} "
              f"-> 1:1 with {self.n} hfield cells")

        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if gid < 0:
            print(f"  !! no geom named {geom_name!r}")
            return
        mid = int(model.geom_matid[gid])
        if mid < 0:
            print(f"geom {geom_name!r}: NO material -- the texture can never show")
            return
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, mid)
        texid = [int(t) for t in np.atleast_1d(np.asarray(model.mat_texid[mid]))]
        print(f"geom {geom_name!r}: material {name!r} texid slots={texid} "
              f"-> uses this texture: {self.tid in texid}")
        if self.tid not in texid:
            print("  !! the floor material does not reference this texture; "
                  "painting it can have no visible effect")
        print(f"  texrepeat={model.mat_texrepeat[mid]} texuniform={model.mat_texuniform[mid]}")

        tex = model.tex_data[self.tadr:self.tadr + self.n * self.nch].reshape(-1, self.nch)
        cols, cnt = np.unique(tex, axis=0, return_counts=True)
        print("texture contents:")
        for c, k in sorted(zip(cols.tolist(), cnt.tolist()), key=lambda t: -t[1])[:4]:
            print(f"   {tuple(int(v) for v in c)}  {k / self.n:.1%}")

    # -- orientation calibration -------------------------------------------
    def calibrate(self, model, renderer=None, viewer=None):
        """Paint the four index quadrants in four colours, plus a white square
        at index (0,0). Render once from directly above and read off where each
        colour landed; the table printed below turns that into an orient value.
        """
        q = np.empty((self.nrow, self.ncol, 3), np.uint8)
        r2, c2 = self.nrow // 2, self.ncol // 2
        q[:r2, :c2] = CAL_COLORS["A"]
        q[:r2, c2:] = CAL_COLORS["B"]
        q[r2:, :c2] = CAL_COLORS["C"]
        q[r2:, c2:] = CAL_COLORS["D"]
        mark = max(self.nrow // 12, 2)
        q[:mark, :mark] = (255, 255, 255)          # index (0,0)

        buf = np.zeros((self.n, self.nch), np.uint8)
        buf[:, :3] = q.reshape(-1, 3)
        if self.nch == 4:
            buf[:, 3] = 255
        model.tex_data[self.tadr:self.tadr + self.n * self.nch] = buf.ravel()
        self._upload(model, renderer, viewer)
        self._last = None

        print("calibration painted:")
        print("   WHITE square = hfield index (0,0)")
        print("   BLUE  block  = same rows as white, higher COLUMN index")
        print("                  -> shows which screen direction +column runs")
        print("   GREEN block  = same cols as white, higher ROW index")
        print("                  -> shows which screen direction +row runs")
        print("Prefer sweep_orients(): it renders all eight and you just pick "
              "the one where the colour sits on the walls.")
        return q


def sweep_orients(painter, model, render_fn, out_dir="orient_sweep",
                  renderer=None, viewer=None):
    """Render the maze once per orientation and save each to a PNG.

    No reasoning about axis conventions: exactly one of the eight images will
    have the orange sitting on the walls instead of beside them. Use that name.

        from maze_visual_runtime import sweep_orients
        sweep_orients(self.maze_painter, self.env.model,
                      lambda: render(state, self.visualizer_config,
                                     self.env, self.renderer),
                      renderer=self.renderer)

    render_fn must return an HxWx3 uint8 array.
    """
    import os
    os.makedirs(out_dir, exist_ok=True)
    saved, original = [], painter.orient
    try:
        for name in ORIENTS:
            painter.orient = name
            painter.paint(model, renderer=renderer, viewer=viewer, force=True)
            img = np.asarray(render_fn())
            path = os.path.join(out_dir, f"{name}.png")
            try:
                from PIL import Image
                Image.fromarray(img.astype(np.uint8)).save(path)
            except ImportError:
                import cv2
                cv2.imwrite(path, img[..., ::-1])
            saved.append(path)
            print(f"  wrote {path}")
    finally:
        painter.orient = original
    print(f"\nOpen the {len(saved)} images and pick the one where the orange "
          f"lands ON the walls, then pass that name as orient=.")
    return saved