import matplotlib.pyplot as plt
import numpy as np


def validate_coordinates(coords: np.ndarray, expected_dim: int):
    """
    Validates that coordinates are a 2D array with shape (N, expected_dim).
    """
    if not isinstance(coords, np.ndarray):
        raise ValueError("Coordinates must be a numpy array!")

    if coords.ndim != 2 or coords.shape[1] != expected_dim:
        raise ValueError(f"Coordinates must have shape (N,{expected_dim})")


def plot_2d(coords: np.ndarray, title: str | None = None) -> plt.Figure:
    """Generates a 2D scatter plot from coordinates (N,2)."""
    validate_coordinates(coords, 2)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(coords[:, 0], coords[:, 1])
    ax.set_title(title or "2D Plot")
    return fig


def plot_3d(coords: np.ndarray, title: str | None = None) -> plt.Figure:
    """Generates a 3D scatter plot from coordinates (N,3)."""
    validate_coordinates(coords, 3)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2])
    ax.set_title(title or "3D Plot")
    return fig
