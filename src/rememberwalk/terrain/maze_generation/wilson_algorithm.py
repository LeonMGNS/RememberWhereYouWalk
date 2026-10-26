"""Offline maze generation with Wilson's algorithm (uniform spanning trees), braided to remove dead ends.

Run this file directly to (re)create the maze pickle files in data/mazes (parameters in the __main__
block). The env only loads those files (terrain/MazeStaticTerrainGenerator.py). A maze is stored as a
(rows + 1, cols + 1, 2) edge description: [..., 0] east wall and [..., 1] south wall of each cell, with
row and column 0 as the outer boundary.
"""

import pickle
import random
from copy import deepcopy

import jax.numpy as jnp
import numpy as np

from rememberwalk.utils.paths import MAZE_DIR


class Maze:
    """Generator of rectangular mazes of ``maze_rows`` x ``maze_cols`` cells (from ``terrain_parameters``)."""

    def __init__(self, terrain_parameters, save_to_file=False):
        self.terrain_parameters = terrain_parameters
        self.height_field_resolution = self.terrain_parameters["height_field_resolution"]
        self.maze_rows = self.terrain_parameters["maze_rows"]
        self.maze_cols = self.terrain_parameters["maze_cols"]
        self.nr_mazes = self.terrain_parameters["nr_mazes"]
        self.space_per_maze_cell = self.terrain_parameters["space_maze_cell"]
        self.one_meter_in_height_field = int(1 / self.height_field_resolution)
        # Constants for directions
        self.E, self.W, self.N, self.S = 1, 2, 4, 8
        self.dir_list = [self.E, self.S, self.N, self.W]
        self.DX = {self.E: 0, self.W: 0, self.N: -1, self.S: 1}
        self.DY = {self.E: 1, self.W: -1, self.N: 0, self.S: 0}
        self.OPPOSITE = {self.E: self.W, self.W: self.E, self.N: self.S, self.S: self.N}
        self.IN = 0x10
        self.wall_mask = self.get_outer_mask()
        self.save_to_file = save_to_file

    def generate_only_maze_description(self):
        """Generates nr_mazes unique mazes. Saves them to maze_only_desc_<nr_mazes>.pkl if save_to_file,
        else returns them as (nr_mazes, rows + 1, cols + 1, 2).
        """
        maze_descriptions = jnp.zeros((self.nr_mazes, self.maze_rows + 1, self.maze_cols + 1, 2))
        training_descriptions = []
        seen_mazes = set()
        while len(training_descriptions) < self.nr_mazes:
            maze = self.generate_maze()
            maze_tensor = self.convert_maze_to_wall_tensor(maze)
            maze_bytes = np.array(maze_tensor).tobytes()
            if maze_bytes not in seen_mazes:
                seen_mazes.add(maze_bytes)
                training_descriptions.append(maze_tensor)
                print(f"Generated unique training maze: {len(training_descriptions)} / {self.nr_mazes}")
            else:
                print("Duplicate found. Regenerating...")

        maze_descriptions = jnp.stack(training_descriptions)
        if self.save_to_file:
            save_dict = {
                "maze_descriptions": np.array(maze_descriptions),
            }

            save_path = f"{MAZE_DIR}/"
            with open(f"{save_path}/maze_only_desc_{self.nr_mazes}.pkl", "wb") as f:
                pickle.dump(save_dict, f)
        else:
            return maze_descriptions

    def generate_eval_only_maze_description(self):
        """Generates nr_mazes held-out eval mazes (none in maze_only_desc_300000.pkl) and saves them to
        maze_only_desc_<nr_mazes>_eval.pkl.
        """
        with open(f"{MAZE_DIR}/maze_only_desc_300000.pkl", "rb") as f:
            maze_dict = pickle.load(f)
        training_descriptions = maze_dict["maze_descriptions"]
        eval_mazes = []
        training_descriptions = jnp.array(training_descriptions)

        while len(eval_mazes) < self.nr_mazes:
            maze = self.generate_maze()
            maze_tensor = self.convert_maze_to_wall_tensor(maze)
            is_in_train = jnp.any(jnp.all(maze_tensor == training_descriptions, axis=(-3, -2, -1)))

            is_in_eval = False
            if len(eval_mazes) > 0:
                eval_stack = jnp.stack(eval_mazes)
                is_in_eval = jnp.any(jnp.all(maze_tensor == eval_stack, axis=(-3, -2, -1)))

            # Only add if it is novel to BOTH the training set and the current eval set
            if not is_in_train and not is_in_eval:
                eval_mazes.append(maze_tensor)
                print(f"Generated unique eval maze: {len(eval_mazes)} / {self.nr_mazes}")
            else:
                print("Duplicate found. Regenerating...")

        # Convert the accumulated list into the final JAX array shape
        maze_descriptions = jnp.stack(eval_mazes)

        save_dict = {
            "maze_descriptions": np.array(maze_descriptions),
        }

        save_path = f"{MAZE_DIR}/"
        with open(f"{save_path}/maze_only_desc_{self.nr_mazes}_eval.pkl", "wb") as f:
            pickle.dump(save_dict, f)

    def generate_maze(self):
        """One maze as a grid of direction bit masks (E, W, N, S = open passages), braided."""
        grid = [[0 for _ in range(self.maze_cols)] for _ in range(self.maze_rows)]
        grid[random.randint(0, self.maze_rows - 1)][random.randint(0, self.maze_cols - 1)] = self.IN

        remaining = self.maze_rows * self.maze_cols - 1

        while remaining > 0:
            for x, y, direction in self.walk(grid):
                nx, ny = x + self.DX[direction], y + self.DY[direction]
                grid[x][y] |= direction

                grid[nx][ny] |= self.OPPOSITE[direction]
                grid[x][y] |= self.IN
                grid[nx][ny] |= self.IN
                remaining -= 1

        braid_grid = self.fix_and_braid_maze(deepcopy(grid))
        return braid_grid

    def walk(self, grid):
        """Wilson's loop-erased random walk from a random unvisited cell to the current tree."""
        while True:
            cx, cy = random.randint(0, self.maze_rows - 1), random.randint(0, self.maze_cols - 1)
            if grid[cx][cy] != 0:
                continue

            visits = {(cx, cy): 0}
            start_x, start_y = cx, cy
            walking = True
            path = []

            while walking:
                if (cx, cy) in path:
                    loop_start_index = path.index((cx, cy))
                    path = path[: loop_start_index + 1]
                else:
                    path.append((cx, cy))

                walking = False

                for dir in random.sample(self.dir_list, 4):
                    nx, ny = cx + self.DX[dir], cy + self.DY[dir]
                    if 0 <= nx < self.maze_rows and 0 <= ny < self.maze_cols:
                        visits[(cx, cy)] = dir
                        if grid[nx][ny] != 0:
                            break
                        else:
                            cx, cy = nx, ny
                            walking = True
                            break
            path = []
            x, y = start_x, start_y
            while (x, y) in visits:
                dir = visits[(x, y)]
                path.append((x, y, dir))
                x, y = x + self.DX[dir], y + self.DY[dir]

            return path

    def print_maze(self, grid):
        """Prints the grid maze as ASCII."""
        maze = grid
        rows, cols = self.maze_rows, self.maze_cols
        S, E = self.S, self.E

        # Top border
        print(" " + "_" * (cols * 2 - 1))
        for x in range(rows):
            line = "|"
            for y in range(cols):
                cell = maze[x][y]
                # Draw '_' if there is a south wall or if it's the bottom row and no open south wall
                if (cell & S) == 0 or x == rows - 1:
                    south = "_"
                else:
                    south = " "
                # Draw '|' if there is a east wall or if it's the rightmost column
                if (cell & E) == 0 or y == cols - 1:
                    east = "|"
                else:
                    east = " "
                line += south + east
            print(line)

    def convert_maze_to_wall_tensor(self, maze):
        """Grid maze -> (rows + 1, cols + 1, 2) edge description.

        [..., 0] is the east wall and [..., 1] the south wall of each cell.
        """
        R, C = self.maze_rows, self.maze_cols
        maze_edge_representation = jnp.zeros((R + 1, C + 1, 2), jnp.bool_)
        maze_edge_representation = maze_edge_representation.at[0, :, 1].set(True)
        maze_edge_representation = maze_edge_representation.at[:, 0, 0].set(True)
        for r in range(1, self.maze_rows + 1):
            for c in range(1, self.maze_cols + 1):
                cell = maze[r - 1][c - 1]

                south_wall = 1 if (cell & self.S) == 0 else 0
                east_wall = 1 if (cell & self.E) == 0 else 0
                maze_edge_representation = maze_edge_representation.at[r, c, 0].set(east_wall)
                maze_edge_representation = maze_edge_representation.at[r, c, 1].set(south_wall)

        return maze_edge_representation

    def print_maze_edge_representation(self, maze):
        """Prints an edge description as ASCII."""
        rows, cols = self.maze_rows + 1, self.maze_cols + 1

        # Top border
        print(" " + "_" * ((cols - 1) * 2 - 1))
        for x in range(1, rows):
            line = "|"
            for y in range(1, cols):
                south = maze[x, y, 1]
                east = maze[x, y, 0]
                # Draw '_' if there is a south wall or if it's the bottom row and no open south wall
                if south:
                    south = "_"
                else:
                    south = " "
                # Draw '|' if there is a east wall or if it's the rightmost column
                if east:
                    east = "|"
                else:
                    east = " "
                line += south + east
            print(line)

    def get_outer_mask(self):
        """
        Returns a flattened boolean mask of shape ((R+1)*C + R*(C+1),)
        where only the outer walls are True.
        """
        # Horizontal edges: (R+1, C)
        R, C = self.maze_rows, self.maze_cols
        horizontal_mask = jnp.zeros((R + 1, C), dtype=bool)

        # Top boundary (row 0)
        horizontal_mask = horizontal_mask.at[0, :].set(True)

        # Bottom boundary (row R)
        horizontal_mask = horizontal_mask.at[R, :].set(True)

        # Vertical edges: (R, C+1)
        vertical_mask = jnp.zeros((R, C + 1), dtype=bool)

        # Left boundary (col 0)
        vertical_mask = vertical_mask.at[:, 0].set(True)

        # Right boundary (col C)
        vertical_mask = vertical_mask.at[:, C].set(True)

        # Flatten and concatenate
        horizontal_flat = horizontal_mask.reshape(-1)
        vertical_flat = vertical_mask.reshape(-1)

        return jnp.concatenate([horizontal_flat, vertical_flat], axis=0)

    def get_hand_crafted_maze(self):
        """The hand-designed lab maze as an edge description; also saved to maze_only_desc_lab_maze.pkl."""
        R, C = 5, 5

        # Walls read directly off the drawing.
        # V[r][c]: vertical wall on grid-line c for row r. c=0 -> west border ... c=5 -> east border.
        V = [
            [1, 0, 0, 0, 0, 1],
            [1, 0, 1, 1, 0, 1],
            [1, 1, 0, 0, 1, 1],
            [1, 0, 0, 0, 1, 1],
            [1, 0, 0, 0, 0, 1],
        ]
        # H[rl][c]: horizontal wall on grid-line rl for col c. rl=0 -> north border ... rl=4 -> south border.
        H = [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 0, 1],
            [0, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ]

        rep = jnp.zeros((R + 1, C + 1, 2), dtype=bool)
        rep = rep.at[0, :, 1].set(True)  # top (north) border
        rep = rep.at[:, 0, 0].set(True)  # left (west) border
        for r in range(1, R + 1):
            for c in range(1, C + 1):
                rep = rep.at[r, c, 0].set(bool(V[r - 1][c]))  # east wall of cell (r-1, c-1)
                rep = rep.at[r, c, 1].set(bool(H[r][c - 1]))  # south wall of cell (r-1, c-1)

        self.print_maze_edge_representation(rep)
        maze_description = jnp.expand_dims(rep, 0)
        save_dict = {
            "maze_descriptions": np.array(maze_description),
        }

        save_path = f"{MAZE_DIR}/"
        with open(f"{save_path}/maze_only_desc_lab_maze.pkl", "wb") as f:
            pickle.dump(save_dict, f)
        return rep

    def fix_and_braid_maze(self, grid, p=0.7):
        """
        Pass 1: Erases any existing 2x2 open rooms.
        Pass 2: Braids the maze by opening dead ends without creating new 2x2 rooms.
        """

        # ==========================================
        # PASS 1: Erase existing 2x2 rooms
        # ==========================================
        for x in range(self.maze_rows - 1):
            for y in range(self.maze_cols - 1):
                # A 2x2 room has 4 open internal walls. Check them:
                n_wall_open = (grid[x][y] & self.E) != 0  # Between (x,y) and (x,y+1)
                s_wall_open = (grid[x + 1][y] & self.E) != 0  # Between (x+1,y) and (x+1,y+1)
                w_wall_open = (grid[x][y] & self.S) != 0  # Between (x,y) and (x+1,y)
                e_wall_open = (grid[x][y + 1] & self.S) != 0  # Between (x,y+1) and (x+1,y+1)

                if n_wall_open and s_wall_open and w_wall_open and e_wall_open:
                    # It is a 2x2 room. Pick a random wall and close it.
                    wall_to_close = random.choice(["N", "S", "W", "E"])

                    if wall_to_close == "N":
                        grid[x][y] &= ~self.E
                        grid[x][y + 1] &= ~self.W
                    elif wall_to_close == "S":
                        grid[x + 1][y] &= ~self.E
                        grid[x + 1][y + 1] &= ~self.W
                    elif wall_to_close == "W":
                        grid[x][y] &= ~self.S
                        grid[x + 1][y] &= ~self.N
                    elif wall_to_close == "E":
                        grid[x][y + 1] &= ~self.S
                        grid[x + 1][y + 1] &= ~self.N

        # ==========================================
        # PASS 2: Braid the maze
        # ==========================================
        dead_ends = []
        for x in range(self.maze_rows):
            for y in range(self.maze_cols):
                paths = grid[x][y] & ~self.IN
                if paths in [self.N, self.S, self.E, self.W]:
                    dead_ends.append((x, y))

        random.shuffle(dead_ends)

        for cx, cy in dead_ends:
            # Re-verify it's still a dead end (Pass 2 may have changed it)
            if (grid[cx][cy] & ~self.IN) not in [self.N, self.S, self.E, self.W]:
                continue

            if random.random() > p:
                continue

            # Gather closed walls that don't lead out of bounds
            closed_dirs = []
            for d in self.dir_list:
                if (grid[cx][cy] & d) == 0:
                    nx, ny = cx + self.DX[d], cy + self.DY[d]
                    if 0 <= nx < self.maze_rows and 0 <= ny < self.maze_cols:
                        closed_dirs.append(d)

            # Filter out walls that would create a 2x2 open room
            safe_dirs = [d for d in closed_dirs if not self._would_form_2x2(grid, cx, cy, d)]

            # If all options create a 2x2, leave this dead end intact
            if not safe_dirs:
                continue

            best_dir = random.choice(safe_dirs)
            nx, ny = cx + self.DX[best_dir], cy + self.DY[best_dir]

            grid[cx][cy] |= best_dir
            grid[nx][ny] |= self.OPPOSITE[best_dir]

        return grid

    def _would_form_2x2(self, grid, cx, cy, d):
        """
        Checks if opening direction 'd' from (cx, cy) creates a 2x2 block of open cells.
        """
        nx, ny = cx + self.DX[d], cy + self.DY[d]

        if d in [self.N, self.S]:
            if cy + 1 < self.maze_cols:
                if (grid[cx][cy] & self.E) and (grid[nx][ny] & self.E) and (grid[cx][cy + 1] & d):
                    return True
            if cy - 1 >= 0:
                if (grid[cx][cy] & self.W) and (grid[nx][ny] & self.W) and (grid[cx][cy - 1] & d):
                    return True

        elif d in [self.E, self.W]:
            if cx + 1 < self.maze_rows:
                if (grid[cx][cy] & self.S) and (grid[nx][ny] & self.S) and (grid[cx + 1][cy] & d):
                    return True
            if cx - 1 >= 0:
                if (grid[cx][cy] & self.N) and (grid[nx][ny] & self.N) and (grid[cx - 1][cy] & d):
                    return True

        return False


if __name__ == "__main__":
    # Offline maze generation: run this file directly to (re)create the maze pickle files.
    # Maze parameters
    maze_rows = 5
    maze_cols = 5
    cell_visit_size_in_meters = 1.0
    nr_mazes = 100
    use_file_maze = True
    file_maze_nr = "only_desc"
    space_maze_cell = 2.0
    global_height_field_x_length_in_meters = nr_mazes * maze_rows * space_maze_cell
    global_height_field_y_length_in_meters = maze_cols * space_maze_cell
    distribution_size = int(2 * (maze_cols + 1) * (maze_rows + 1))
    exploration_min = 1.0
    exploration_max = 1.0
    local_height_field_x_length_in_meters = maze_rows * space_maze_cell
    local_height_field_y_length_in_meters = maze_cols * space_maze_cell
    height_field_resolution = 0.1

    terrain_parameters = {
        "height_field_resolution": height_field_resolution,
        "local_height_field_x_length_in_meters": local_height_field_x_length_in_meters,
        "local_height_field_y_length_in_meters": local_height_field_y_length_in_meters,
        "height_field_x_length_in_meters": global_height_field_x_length_in_meters,
        "height_field_y_length_in_meters": global_height_field_y_length_in_meters,
        "cells_in_x": 50,
        "cells_in_y": 50,
        "wall_height": 0.5,
        "length_in_x": 2.5,  # 4m from center to edge
        "length_in_y": 2.5,  # 2m from center to edge ==> Overall 8m x 4m terrain map
        "max_possible_height": 30,  # 30 m maximum possible height
        "maze_rows": maze_rows,
        "maze_cols": maze_cols,
        "nr_mazes": nr_mazes,
        "space_maze_cell": space_maze_cell,
        "cell_visit_size_in_meters": cell_visit_size_in_meters,
        "cell_visit_matrix_x_size": int((maze_rows * space_maze_cell) // cell_visit_size_in_meters),
        "cell_visit_matrix_y_size": int((maze_cols * space_maze_cell) // cell_visit_size_in_meters),
        "critic_max_pool_window_size": 2,
        "maze_file": file_maze_nr,
        "terrain_curr_coeff_window_size": 0.15,
        "wall_thickness": 2,
    }

    # maze_generator = Maze(terrain_parameters,True)
    # maze_generator.get_hand_crafted_maze()
    # maze_generator.generate_only_maze_description()
    # maze_generator.generate_eval_only_maze_description()
