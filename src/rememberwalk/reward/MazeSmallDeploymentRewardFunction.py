"""Reward of the 5x5 deployment maze (``imports.reward_handler``).

The training reward with the wall distances and cell width of the smaller maze.
"""

from rememberwalk.reward.MazeSmallRewardFunction import MazeReward as _TrainingMazeReward


class MazeReward(_TrainingMazeReward):
    """Training reward with the wall distances and cell width of the 5x5 deployment maze."""

    CLOSE_TO_WALL_DISTANCE = 0.2
    CLOSE_TO_WALL_OFFSET = 0.1
    HALF_CELL_WIDTH = 0.5
