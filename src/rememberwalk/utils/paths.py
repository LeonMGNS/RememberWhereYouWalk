"""Filesystem locations used by rememberwalk.

The robot model ships inside the package (ASSETS_DIR). Everything else -- configs,
maze files, checkpoints, logs and videos -- lives under ROOT_DIR:

* In a source checkout (``<root>/src/rememberwalk`` next to ``<root>/pyproject.toml``)
  ROOT_DIR is the repository root, regardless of the working directory.
* Otherwise ROOT_DIR is the current working directory.

Each location can be overridden with an environment variable:
REMEMBERWALK_ROOT, REMEMBERWALK_DATA_DIR, REMEMBERWALK_CKPT_DIR.
"""

import os
from importlib.resources import files
from pathlib import Path


def _default_root() -> Path:
    package_dir = Path(__file__).resolve().parent.parent
    repo_root = package_dir.parent.parent
    if package_dir.parent.name == "src" and (repo_root / "pyproject.toml").is_file():
        return repo_root
    return Path.cwd()


ASSETS_DIR = Path(str(files("rememberwalk") / "assets"))
ROOT_DIR = Path(os.environ.get("REMEMBERWALK_ROOT", _default_root())).resolve()
DATA_DIR = Path(os.environ.get("REMEMBERWALK_DATA_DIR", ROOT_DIR / "data")).resolve()
MAZE_DIR = DATA_DIR / "mazes"
CKPT_DIR = Path(os.environ.get("REMEMBERWALK_CKPT_DIR", ROOT_DIR / "checkpoints")).resolve()
CONFIG_DIR = ROOT_DIR / "configs"
