import matplotlib.pyplot as plt
from pathlib import Path
from typing import Generator, Tuple

import pandas as pd


def load_history(data_path: Path) -> Tuple[Path, Tuple[Generator, None, None]]:
    for file in data_path.rglob("*_dynamic_his.csv"):
        yield file, pd.read_csv(file)


def plot_moving_average_reward(dataframe: pd.DataFrame, filename: str, output_dir: Path, ylabel: str, xlabel: str,
                               title: str):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot([x for x in range(len(dataframe['episode']))],
            dataframe['episode_reward'].rolling(window=100).mean(),
            label='Moving average')
    ax.legend()
    plt.title(label=title)
    ax.set(ylabel=ylabel, xlabel=xlabel)
    plt.grid()
    fig.savefig(output_dir / filename.replace('_dynamic_his.csv', 'mv_avg_plot.jpg'))
    plt.close(fig)


def plot_metric(dataframe: pd.DataFrame, filename: str, output_dir: Path, ylabel: str, xlabel: str,
                title: str):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot([x for x in range(len(dataframe['episode']))],
            dataframe['metric'],
            label='Metric')
    ax.legend()
    plt.title(label=title)
    ax.set(ylabel=ylabel, xlabel=xlabel)
    plt.grid()
    fig.savefig(output_dir / filename.replace('_dynamic_his.csv', 'metric_plot.jpg'))
    plt.close(fig)
