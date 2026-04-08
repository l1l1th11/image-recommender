import base64
import io
import json
import logging
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html
from PIL import Image
from sklearn.neighbors import NearestNeighbors

from image_recommender.db.connector import get_path_by_id
from image_recommender.viz.plots import POINT_COLOR, get_cluster_color_mapping

_thumbnail_cache: dict[tuple[str, int], str] = {}


def load_coordinates(coords_path: Path, ids_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads embedding coordinates and corresponding image IDs.

    IMPORTANT ALIGNMENT INVARIANT:
    coords[i], ids[i], and embeddings[i] must refer to the same image.
    """
    if not coords_path.exists():
        raise FileNotFoundError(f"Coordinate file not found: {coords_path}")

    if not ids_path.exists():
        raise FileNotFoundError(f"ID file not found: {ids_path}")

    coords = np.load(coords_path)

    if ids_path.suffix == ".npy":
        ids = np.load(ids_path, allow_pickle=True)

    elif ids_path.suffix == ".json":
        with open(ids_path, encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            ids = np.array(data)

        elif isinstance(data, dict) and "ids" in data:
            ids = np.array(data["ids"])

        else:
            raise ValueError(f"IDs must be explicitly provided in {ids_path}.")

    else:
        raise ValueError(f"Unsupported ID file format: {ids_path}")

    if coords.shape[0] != len(ids):
        raise ValueError("Coordinate array and ID array length mismatch!")

    if coords.shape[1] != 2:
        raise ValueError("Coordinates must have shape (N,2)")

    if not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("IDs must be integers!")

    return coords, ids


def load_embeddings_from_shards(embeddings_root: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads embeddings from shard directories.
    """
    all_embeddings = []
    all_ids = []

    for shard in sorted(embeddings_root.glob("shard_*")):
        emb_path = shard / "features.npy"
        ids_path = shard / "ids.npy"

        if not emb_path.exists() or not ids_path.exists():
            continue

        emb = np.load(emb_path)
        ids = np.load(ids_path)

        all_embeddings.append(emb)
        all_ids.append(ids)

    if not all_embeddings:
        raise ValueError(f"No embedding shards found in {embeddings_root}")

    embeddings = np.vstack(all_embeddings)
    ids = np.concatenate(all_ids)

    return embeddings, ids


def build_neighbor_model(embeddings: np.ndarray, k: int) -> NearestNeighbors:
    """
    Builds a k-nearest neighbor index using the embeddings.
    """
    n_neighbors = min(k + 1, len(embeddings))

    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(embeddings)

    return nn


def show_neighbor_grid(neighbor_ids: list[int], db_path: Path) -> list:
    """
    Displays a grid of thumbnail images for given neighbor IDs.
    """
    images = []
    for img_id in neighbor_ids:
        try:
            path = resolve_image_path(int(img_id), db_path)
        except Exception as e:
            logging.warning(f"No path for image_id={img_id}: {e}")
            continue

        if path is None:
            continue
        thumb = create_thumbnail_cached(str(path), 128)
        if not thumb:
            continue
        images.append(
            html.Div(
                [
                    html.Img(
                        src=thumb,
                        style={
                            "width": "128px",
                            "height": "128px",
                            "objectFit": "cover",
                            "margin": "5px",
                        },
                    ),
                    html.Div(
                        f"ID: {img_id}",
                        style={
                            "textAlign": "center",
                            "fontSize": "12px",
                        },
                    ),
                ],
                style={
                    "display": "inline-block",
                    "textAlign": "center",
                    "margin": "5px",
                },
            )
        )

    return images


def build_scatter(
    coords: np.ndarray, ids: np.ndarray, clusters: np.ndarray | None = None
) -> go.Figure:
    """
    Creates a Plotly ScatterGL figure for 2D embeddings.
    """
    if clusters is not None:
        label_to_color = get_cluster_color_mapping(clusters)

        marker_colors = [
            f"rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, {a})"
            for (r, g, b, a) in [label_to_color[label] for label in clusters]
        ]
    else:
        r, g, b = [int(c * 255) for c in POINT_COLOR]
        marker_colors = f"rgba({r},{g},{b},1)"

    fig = go.Figure(
        go.Scattergl(
            x=coords[:, 0],
            y=coords[:, 1],
            mode="markers",
            marker=dict(
                size=5,
                opacity=0.6,
                color=marker_colors,
            ),
            text=[f"<b>ID:</b> {i}" for i in ids],
            hovertemplate="%{text}<extra></extra>",
        )
    )

    fig.update_layout(title="Embedding Explorer", dragmode="pan", hovermode="closest")

    return fig


def run_embedding_explorer(
    coords_path: Path,
    ids_path: Path,
    db_path: Path,
    embeddings_path: Path,
    k: int = 5,
    show: bool = True,
    return_figure: bool = False,
    cluster_path: Path | None = None,
) -> go.Figure | None:
    """
    Launches interactive embedding explorer.
    Input:
    - coords_path (path to coordinate file)
    - ids_path (path to ID file)
    - embeddings_path (path to embeddings shard directory)
    - k (number of neighbors)
    - show (whether to show the app)
    Output: figure
    """
    coords, ids = load_coordinates(coords_path, ids_path)

    clusters = None
    if cluster_path is not None and cluster_path.exists():
        clusters = np.load(cluster_path)

        if len(clusters) != len(coords):
            raise ValueError("Cluster labels and coordinates length mismatch!")

    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings path not found: {embeddings_path}")

    embeddings, emb_ids = load_embeddings_from_shards(embeddings_path)

    id_to_idx = {int(i): idx for idx, i in enumerate(emb_ids)}

    try:
        idxs = np.array([id_to_idx[int(i)] for i in ids])
        embeddings = embeddings[idxs]
    except KeyError as e:
        raise ValueError(f"ID {e} from coords not found in embeddings!") from e

    if len(embeddings) != len(coords):
        raise ValueError("Embeddings and coordinates length mismatch!")

    nn = build_neighbor_model(embeddings, k)

    fig = build_scatter(coords, ids, clusters)

    app = Dash(__name__)

    app.layout = html.Div(
        [
            html.H2("Embedding Explorer"),
            html.Div(
                [
                    dcc.Graph(
                        id="scatter",
                        figure=fig,
                        style={"height": "70vh", "width": "70%"},
                    ),
                    html.Div(
                        [
                            html.H3("Preview"),
                            html.Img(
                                id="hover-preview",
                                style={"width": "100%", "maxHeight": "300px"},
                            ),
                        ],
                        style={
                            "width": "30%",
                            "padding": "10px",
                        },
                    ),
                ],
                style={
                    "display": "flex",
                    "alignItems": "flex-start",
                    "gap": "10px",
                },
            ),
            html.Div(
                [
                    html.H3("Nearest Neighbors"),
                    html.Div(
                        id="neighbors",
                        style={
                            "display": "flex",
                            "flexWrap": "wrap",
                            "gap": "10px",
                        },
                    ),
                ],
                style={"marginTop": "0px"},
            ),
        ]
    )

    @app.callback(
        Output("hover-preview", "src"),
        Input("scatter", "hoverData"),
    )
    def update_hover(hoverData):
        if not hoverData or "points" not in hoverData:
            return ""

        idx = hoverData["points"][0].get("pointIndex")
        if idx is None:
            return ""

        image_id = int(ids[idx])

        try:
            path = resolve_image_path(image_id, db_path)
        except Exception:
            logging.warning(f"Failed hover preview for {image_id}")
            return ""

        if path is None:
            return ""

        return create_thumbnail_cached(str(path), 256)

    @app.callback(
        Output("neighbors", "children"),
        Input("scatter", "clickData"),
    )
    def update_neighbors(clickData):
        if not clickData or "points" not in clickData:
            return html.Div("Click a point to see neighbors")

        idx = clickData["points"][0].get("pointIndex")
        if idx is None:
            return html.Div("Invalid selection")

        _, neighbors = nn.kneighbors([embeddings[idx]])

        neighbor_ids = ids[neighbors[0]]

        neighbor_ids = [int(i) for i in neighbor_ids if i != ids[idx]]

        neighbor_ids = neighbor_ids[:k]

        return show_neighbor_grid(neighbor_ids, db_path)

    if show:
        app.run(debug=False)

    if return_figure:
        return fig


def resolve_image_path(image_id: int, db_path: Path) -> Path | None:
    """
    Resolves image path from database.
    """
    try:
        path = get_path_by_id(image_id, db_path)
        if path:
            p = Path(path)
            if p.exists():
                return p
    except Exception as e:
        logging.warning(f"DB lookup failed for {image_id}: {e}")

    return None


def create_thumbnail_cached(image_path: str, size: int) -> str:
    """
    Creates thumbnail with caching (in-memory).
    """
    key = (image_path, size)

    if key in _thumbnail_cache:
        return _thumbnail_cache[key]

    result = _create_thumbnail(Path(image_path), size)
    _thumbnail_cache[key] = result

    return result


def _create_thumbnail(image_path: Path | None, size: int = 96) -> str:
    """
    Creates base64 encoded thumbnail for hover preview.
    """
    if image_path is None:
        return ""

    try:
        with Image.open(image_path) as img:
            img.thumbnail((size, size))

            buf = io.BytesIO()
            img.save(buf, format="PNG")

        encoded = base64.b64encode(buf.getvalue()).decode()

        return f"data:image/png;base64,{encoded}"

    except Exception:
        logging.warning(f"Failed to load image: {image_path}")
        return ""
