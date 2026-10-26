"""State of the history-based approaches (``imports.state``): MazeState plus the observation history."""

import jax
from flax import struct

from rememberwalk.states.MazeState import MazeState


@struct.dataclass
class MazeStateObservationHistory(MazeState):
    """MazeState plus the observation history of the history-based approaches (Transformer, FixedWindow)."""

    observation_history: jax.Array
