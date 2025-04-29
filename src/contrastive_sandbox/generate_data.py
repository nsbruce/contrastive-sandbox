from torchsig.datasets.dataset_metadata import NarrowbandMetadata
from torchsig.utils.generate import generate
import click
import math
import os

@click.command
@click.option("--num-signals", type=int)
@click.option("--num-iq-samples-per-signal", type=int)
@click.option("--impaired", is_flag=True, default=False)
@click.option("--root", type=str, help="directory to save the dataset in")
@click.option("--batch-size", type=int, default=32)
def main(num_signals: int, num_iq_samples_per_signal: int, impaired: bool, root: str, batch_size: int):
    dataset_metadata = NarrowbandMetadata(
        num_samples = num_signals,
        num_iq_samples_dataset=num_iq_samples_per_signal,
        fft_size= int(math.sqrt(num_iq_samples_per_signal)), # wth is this
        impairment_level= 2 if impaired else 0,
        num_signals_min = 1,
        signal_duration_percent_min=100
    )

    cpu_count = os.cpu_count()
    if cpu_count is None:
        cpu_count = 3

    generate(
        root=root,
        dataset_metadata=dataset_metadata,
        batch_size=batch_size,
        num_workers=cpu_count# // 3,
    )

if __name__ == "__main__":
    main()
