import typing
from torchsig.signals.signal_lists import TorchSigSignalLists
from torchsig.datasets.dataset_metadata import NarrowbandMetadata
from contrastive_sandbox.data_loader import generate_dataset
from collections import defaultdict
import torch.nn as nn
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
from dtaidistance import dtw, innerdistance


def get_signals():
    """
    Generate two signals from each class
    """
    num_signals = len(TorchSigSignalLists.all_signals)
    metadata = NarrowbandMetadata(
        num_samples=None,
        num_iq_samples_dataset=128,
        fft_size=32,
        impairment_level=0,
        num_signals_min=1,
        signal_duration_percent_min=100
    )
    dataset = generate_dataset(metadata)
    class_counter = defaultdict(lambda: [])
    while any([len(v) != 2 for v in class_counter.values()]) or len(class_counter) < num_signals:
        signal = next(dataset)
        class_idx = signal[1][0]['class_index']
        if len(class_counter[class_idx]) < 2:
            class_counter[class_idx].append(signal)
    signals_1 = []
    signals_2 = []
    for key in class_counter:
        signals_1.append(class_counter[key][0])
        signals_2.append(class_counter[key][1])

    assert len(signals_1) == len(TorchSigSignalLists.all_signals)
    assert len(signals_2) == len(TorchSigSignalLists.all_signals)

    print(signals_1[0][0].shape, signals_1[0][0].dtype)

    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        "Different signals", "Same signals"))

    fig.update_layout(
        title="Dynamic time warpings, abs(complex) to compute inner distance",
        coloraxis=dict(
            # cmax=1,
            # cmin=-1,
            colorscale='Viridis',
            colorbar=dict(
                title="DTW distance",
                #     tickvals=[-1, -0.5, 0, 0.5, 1],
            )
        )
    )

    # similarities = cosine_similarities(signals_1, signals_2,
    #    complex_to_float_fn=complex_to_1d_interleaved)
    similarities = dynamic_time_warpings(signals_1, signals_2)

    fig.add_trace(go.Heatmap(
        z=similarities,
        x=TorchSigSignalLists.all_signals,
        y=TorchSigSignalLists.all_signals,
        coloraxis="coloraxis"
    ),
        row=1, col=1
    )
    fig.update_xaxes(title_text="Signal 1", row=1, col=1)
    fig.update_yaxes(title_text="Signal 2", row=1, col=1)

    # similarities = cosine_similarities(signals_1, signals_1,
    #                                    complex_to_float_fn=complex_to_1d_interleaved)
    similarities = dynamic_time_warpings(signals_1, signals_1)

    fig.add_trace(go.Heatmap(
        z=similarities,
        x=TorchSigSignalLists.all_signals,
        y=TorchSigSignalLists.all_signals,
        coloraxis="coloraxis"
    ),
        row=1, col=2
    )
    fig.update_xaxes(title_text="Signal 1", row=1, col=2)
    fig.update_yaxes(title_text="Signal 1", row=1, col=2)

    fig.write_html('dtw.html')


def cosine_similarities(signals_1: np.ndarray, signals_2: np.ndarray, complex_to_float_fn: typing.Callable):
    cosine_similarities = np.empty((len(signals_1), len(signals_2)))

    cosine_similarity_fn = nn.CosineSimilarity(dim=0)

    for signal_1 in signals_1:
        class_idx_1 = signal_1[1][0]['class_index']
        signal_1 = complex_to_float_fn(signal_1[0])
        signal_1 = torch.tensor(signal_1)
        for signal_2 in signals_2:
            class_idx_2 = signal_2[1][0]['class_index']
            signal_2 = complex_to_float_fn(signal_2[0])
            signal_2 = torch.tensor(signal_2)

            cosine_similarities[class_idx_1, class_idx_2] = cosine_similarity_fn(
                signal_1, signal_2)

    return cosine_similarities


class ComplexInnerDistance(innerdistance.CustomInnerDist):

    @staticmethod
    def inner_dist(x, y):
        """
        The distance between two points in the series.
        """
        return np.abs(x - y)

    @staticmethod
    def result(x):
        return x

    @staticmethod
    def inner_val(x):
        return x


def dynamic_time_warpings(signals_1: np.ndarray, signals_2: np.ndarray):
    dtws = np.empty((len(signals_1), len(signals_2)))
    for signal_1 in signals_1:
        class_idx_1 = signal_1[1][0]['class_index']
        for signal_2 in signals_2:
            class_idx_2 = signal_2[1][0]['class_index']

            dtws[class_idx_1, class_idx_2] = dtw.distance(
                signal_1[0], signal_2[0], use_c=False, inner_dist=ComplexInnerDistance)

    return dtws


def complex_to_1d_interleaved(complex_array: np.ndarray) -> np.ndarray:
    real_part = np.real(complex_array)
    imag_part = np.imag(complex_array)
    output = np.empty((len(complex_array)*2))
    output[0::2] = real_part
    output[1::2] = imag_part
    return output


if __name__ == "__main__":
    get_signals()
