"""Small notebook utilities. Analysis and plotting logic stays in lesson cells."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_SNAPSHOT = Path(__file__).parent / 'data/recorded_results.json'
COLORS = ['#246a9c', '#d47721', '#28836b', '#925697', '#c04b55']


def load_demo(demo, snapshot_path=None):
    """An explicit custom path must exist; never silently fall back to old results."""
    path = DEFAULT_SNAPSHOT if snapshot_path is None else Path(snapshot_path)
    data = json.loads(path.read_text())
    if data.get('schema_version') != 1 or data.get('evidence_kind') != 'recorded_local_experiments':
        raise ValueError('Expected version-1 recorded experiment snapshot')
    if not data.get('sources') or demo not in data.get('demos', {}):
        raise ValueError(f'Missing source provenance or demo {demo}')
    print(f'{demo} | recorded experiment data | {path.name}')
    print(data['scope'])
    record = data['demos'][demo]
    for r in record.get('runs', [record]):
        print(f"Run: {r.get('run')} | recorded: {r.get('recorded_at')} | seed: {r.get('seed')}")
    return record


def style():
    plt.rcParams.update({'figure.figsize': (10, 4.8), 'figure.dpi': 110,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.titleweight': 'bold', 'axes.titlesize': 13,
                         'axes.labelsize': 11, 'font.size': 10,
                         'axes.prop_cycle': plt.cycler(color=COLORS)})


def grouped_bars(labels, series, *, title, ylabel='Fraction', fraction=True):
    """Shared layout only; callers compute all quantities explicitly."""
    fig, ax = plt.subplots(layout='constrained')
    x = np.arange(len(labels))
    width = 0.8 / len(series)
    for i, (name, values) in enumerate(series.items()):
        ax.bar(x + (i - (len(series) - 1) / 2) * width, values, width, label=name)
    ax.set(xticks=x, xticklabels=labels, title=title, ylabel=ylabel)
    if fraction:
        ax.set_ylim(0, 1.08)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
    ax.grid(axis='y', alpha=.18)
    ax.set_axisbelow(True)
    return fig, ax
