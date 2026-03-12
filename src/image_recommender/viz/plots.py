from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")  # non-interactive backend

import matplotlib.pyplot as plt


def validate_coordinates(coords: np.ndarray, expected_dim: int):
    """
    Validates that coordinates are a 2D array with shape (N, expected_dim).
    """
    coords = np.asarray(coords)

    if coords.ndim != 2 or coords.shape[1] != expected_dim:
        raise ValueError(f"Coordinates must have shape (N,{expected_dim})")


# -------------------- PLOTS --------------------

POINT_COLOR = (0.86, 0, 0.33)

# 2D-Plot:


def plot_2d(
    coords: np.ndarray,
    title: str | None = None,
    xlabel: str = "X",
    ylabel: str = "Y",
    point_size: float = 20.0,
    alpha: float = 0.8,
    run_dir: str | None = None,
    filename: str = "plot_2d.png",
) -> plt.Figure:
    """
    Generates a 2D scatter plot from coordinates (N,2) and optionally saves it.
    Input:
    - coords (N,2)
    - title (optional)
    - xlabel, ylabel (optional)
    - point_size (optional)
    - alpha (optional, 0-1)
    - run_dir (optional)
    - filename (optional)
    Output:
    - Matplotlib Figure (.png)
    """
    validate_coordinates(coords, 2)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(coords[:, 0], coords[:, 1], s=point_size, alpha=alpha, color=POINT_COLOR)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title or "2D Plot")
    ax.grid(True)

    if run_dir is not None:  # If the run_dir is provided...
        output_path = Path(run_dir) / filename  # ...determine the output path,
        output_path.parent.mkdir(parents=True, exist_ok=True)  # ...create parent directory and
        fig.savefig(output_path, bbox_inches="tight", dpi=150)  # ...save.
        plt.close(fig)

    return fig


# 3D-Plot:


def plot_3d(
    coords: np.ndarray,
    title: str | None = None,
    xlabel: str = "X",
    ylabel: str = "Y",
    zlabel: str = "Z",
    point_size: float = 20.0,
    alpha: float = 0.8,
    run_dir: str | None = None,
    filename: str = "plot_3d.png",
) -> plt.Figure:
    """
    Generates a 3D scatter plot from coordinates (N,3) and optionally saves it.
    Input:
    - coords (N,3)
    - title (optional)
    - xlabel, ylabel, zlabel (optional)
    - point_size (optional)
    - alpha (optional, 0-1)
    - run_dir (optional)
    - filename (optional)
    Output:
    - Matplotlib Figure (.png)
    """
    validate_coordinates(coords, 3)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        coords[:, 0], coords[:, 1], coords[:, 2], s=point_size, alpha=alpha, color=POINT_COLOR
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title or "3D Plot")

    if run_dir is not None:  # If the run_dir is provided...
        output_path = Path(run_dir) / filename  # ...determine the output path,
        output_path.parent.mkdir(parents=True, exist_ok=True)  # ...create parent directory and
        fig.savefig(output_path, bbox_inches="tight", dpi=150)  # ...save.
        plt.close(fig)

    return fig
