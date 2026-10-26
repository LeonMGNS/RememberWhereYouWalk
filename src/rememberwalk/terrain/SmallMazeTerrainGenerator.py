"""Terrain of the 5x5 deployment maze: the training generator with a random wall thickness per edge,
the walls around the start cell always open and the robot starting exactly at the origin."""

import jax
import jax.numpy as jnp

from rememberwalk.terrain.MazeStaticTerrainGenerator import MazeStaticGenerator as _TrainingMazeGenerator


class MazeStaticGenerator(_TrainingMazeGenerator):
    """Deployment-maze variant of the training generator (see the module docstring)."""

    def read_variant_parameters(self, terrain_parameters: dict):
        """Reads the wall thickness range instead of a fixed thickness; no start position noise."""
        self.min_wall_thickness = terrain_parameters["min_wall_thickness"]
        self.max_wall_thickness = terrain_parameters["max_wall_thickness"]

    def open_start_cell(self, maze_edge_curriculum):
        """Removes the walls around the start cell (world origin, the maze centre cell).

        Regardless of the curriculum level, so init-state randomisation can't spawn the robot facing a wall.
        """
        start_cell_r, start_cell_c = self.R // 2, self.C // 2
        maze_edge_curriculum = maze_edge_curriculum.at[start_cell_r + 1, start_cell_c + 1, 0].set(
            False
        )  # east
        maze_edge_curriculum = maze_edge_curriculum.at[start_cell_r + 1, start_cell_c + 1, 1].set(
            False
        )  # south
        maze_edge_curriculum = maze_edge_curriculum.at[start_cell_r, start_cell_c + 1, 1].set(False)  # north
        maze_edge_curriculum = maze_edge_curriculum.at[start_cell_r + 1, start_cell_c, 0].set(False)  # west
        return maze_edge_curriculum

    def build_walls(self, maze_description, key):
        """Walls with a random thickness per edge; consumes one key split."""
        new_hfield = self.generate_maze_walls(maze_description, key)
        key, _ = jax.random.split(key)
        return new_hfield, key

    def start_position(self, key):
        """Exactly the origin (no noise)."""
        return jnp.array([0.01, 0.0])

    def generate_maze_walls(self, maze_description, key):
        """Wall height field with a random thickness in [min, max] per wall edge (boundaries: max)."""
        new_maze = jnp.zeros((self.maze_shape[0], self.maze_shape[1]), dtype=jnp.float32)

        horizontal_edges = maze_description[1:, 1:, 1].reshape((self.R, self.C))
        horizontal_edges_repeat = jnp.repeat(
            horizontal_edges[:-1, :],
            repeats=int(self.space_per_maze_cell / self.height_field_resolution),
            axis=1,
        )
        vertical_edges = maze_description[1:, 1:, 0].reshape((self.R, self.C))
        vertical_edges_repeat = jnp.repeat(
            vertical_edges[:, :-1],
            repeats=int(self.space_per_maze_cell / self.height_field_resolution),
            axis=0,
        )

        new_maze = new_maze.at[self.slice_vals[0] :: self.slice_vals[0], :].set(
            self.maze_wall_height * horizontal_edges_repeat
        )
        new_maze = new_maze.at[:, self.slice_vals[1] :: self.slice_vals[1]].set(
            self.maze_wall_height * vertical_edges_repeat
        )

        # Boundaries
        new_maze = new_maze.at[0, :].set(self.maze_wall_height)
        new_maze = new_maze.at[:, 0].set(self.maze_wall_height)
        new_maze = new_maze.at[-1, :].set(self.maze_wall_height)
        new_maze = new_maze.at[:, -1].set(self.maze_wall_height)

        K = self.max_wall_thickness
        rep = int(self.space_per_maze_cell / self.height_field_resolution)
        k_h, k_v = jax.random.split(key)

        # per-edge random thickness, kept as integers
        h_thick = jax.random.randint(
            k_h, horizontal_edges.shape, self.min_wall_thickness, K + 1
        )  # int32, {1,2,3}
        v_thick = jax.random.randint(k_v, vertical_edges.shape, self.min_wall_thickness, K + 1)

        h_thick_repeat = jnp.repeat(h_thick[:-1, :], repeats=rep, axis=1)
        v_thick_repeat = jnp.repeat(v_thick[:, :-1], repeats=rep, axis=0)

        thick = jnp.zeros_like(new_maze, dtype=jnp.int32)
        thick = thick.at[self.slice_vals[0] :: self.slice_vals[0], :].set(h_thick_repeat)
        thick = thick.at[:, self.slice_vals[1] :: self.slice_vals[1]].set(v_thick_repeat)

        # keep thickness ONLY where an actual wall exists (robust float>0 test, not float ==)
        wall_mask = new_maze > 0
        thick = jnp.where(wall_mask, thick, 0)

        # boundaries: fixed thickness (they are walls, so wall_mask is already True there)
        thick = thick.at[0, :].set(K)
        thick = thick.at[:, 0].set(K)
        thick = thick.at[-1, :].set(K)
        thick = thick.at[:, -1].set(K)

        # layered dilation: static window sizes, integer-exact selection
        result = new_maze  # thickness-1 walls stay as-is
        for t in range(2, K + 1):  # unrolled, t is static
            masked = jnp.where(thick == t, new_maze, 0.0)  # integer == is exact
            dil = jax.lax.reduce_window(masked, 0.0, jax.lax.max, (t, t), (1, 1), "SAME")
            result = jnp.maximum(result, dil)

        return result
