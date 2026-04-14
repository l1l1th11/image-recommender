from pathlib import Path

import matplotlib.pyplot as plt


def pstats_to_dataframe(stats, project_filter="image_recommender"):
    import pandas as pd

    """
    Converts pstats to a DataFrame for analysis and visualization.

    Inputs:
    - stats: pstats.Stats object
    - project_filter: Optional string to filter for functions in the project
    
    Output: DataFrame with columns: file, function, label, ncalls, tottime, cumtime
    """

    records = []

    for func, (_, nc, tt, ct, _) in stats.stats.items():
        filename = func[0]
        func_name = func[2]

        if project_filter is not None and project_filter not in filename:
            continue

        records.append(
            {
                "file": filename,
                "function": func_name,
                "label": f"{Path(filename).name}:{func_name}",
                "ncalls": nc,
                "tottime": tt,
                "cumtime": ct,
            }
        )

    df = pd.DataFrame(records)

    if len(df) == 0:
        return df

    return df.sort_values("cumtime", ascending=False)


def _interpolate_color(value, min_v, max_v):
    """
    Interpolates color between green (fast) and red (slow).

    Inputs:
    - value: the value to determine the color for
    - min_v: the minimum value in the range (mapped to green)
    - max_v: the maximum value in the range (mapped to red)

    Output: a tuple representing the RGB color normalized to [0, 1]
    """

    green = (28, 160, 68)
    red = (220, 0, 85)

    if max_v == min_v:
        t = 0
    else:
        t = (value - min_v) / (max_v - min_v)

    r = int(green[0] + t * (red[0] - green[0]))
    g = int(green[1] + t * (red[1] - green[1]))
    b = int(green[2] + t * (red[2] - green[2]))

    return (r / 255, g / 255, b / 255)


def plot_top_bottlenecks(df, output_path: Path, top_n=20):
    """
    Plots the top bottlenecks from the profiling data.

    Inputs:
    - df: DataFrame containing profiling data with columns 'label' and 'cumtime'
    - output_path: Path to save the plot image
    - top_n: number of top bottlenecks to include in the plot

    Output: None (saves plot to disk)
    """

    if df.empty:
        print("No profiling data found.")
        return

    df = df.head(top_n).copy()
    df = df.sort_values("cumtime", ascending=True)

    min_v = df["cumtime"].min()
    max_v = df["cumtime"].max()

    colors = [_interpolate_color(v, min_v, max_v) for v in df["cumtime"]]

    plt.figure(figsize=(12, 6))

    bars = plt.barh(df["label"], df["cumtime"], color=colors)

    for bar, value in zip(bars, df["cumtime"], strict=False):
        plt.text(
            bar.get_width() + (max_v * 0.01),  # small offset to the right of the bar
            bar.get_y() + bar.get_height() / 2,
            f"{value * 1000:.1f} ms",
            va="center",
            ha="left",
            fontsize=8,
        )

    plt.xlabel("Cumulative Time (s)")
    plt.title("Profiling Bottlenecks")

    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=150)
    plt.close()
