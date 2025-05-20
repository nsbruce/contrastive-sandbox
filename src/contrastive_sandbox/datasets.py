from pathlib import Path
from typing import Optional
from torchsig.datasets.dataset_metadata import NarrowbandMetadata
import click
import math
from torchsig.datasets.datamodules import NarrowbandDataModule
from torchsig.datasets.dataset_utils import to_dataset_metadata
import os
from torchsig.datasets.narrowband import StaticNarrowband, NewNarrowband
from torchsig.utils.writer import DatasetCreator


def get_train_val_data_module(root: str, impaired: bool = False, num_signals: int | None = None, num_iq_samples_per_signal: int | None = None,  transforms: Optional[list] = None, target_transforms: Optional[list] = None) -> tuple[NarrowbandMetadata, NarrowbandDataModule]:

    root = Path(root) / \
        f"torchsig_narrowband_{'impaired' if impaired else 'clean'}"
    train_root = root / "train"
    val_root = root / "val"

    if not train_root.exists() or not val_root.exists():
        assert num_iq_samples_per_signal is not None and num_signals is not None, "num_iq_samples_per_signal and num_signals must be provided if the dataset does not exist"
    else:
        train_meta = to_dataset_metadata(
            (train_root / 'create_dataset_info.yaml').resolve())
        val_meta = to_dataset_metadata(
            (val_root / 'create_dataset_info.yaml').resolve())
        num_signals = train_meta.num_samples + val_meta.num_samples
        num_iq_samples_per_signal = train_meta.num_iq_samples_dataset
        assert num_iq_samples_per_signal == val_meta.num_iq_samples_dataset, "num_iq_samples_per_signal must be the same for train and val datasets"

    dataset_metadata = NarrowbandMetadata(
        num_samples=num_signals,
        num_iq_samples_dataset=num_iq_samples_per_signal,
        # fft size chosen so that one can generate a square spectrogram
        fft_size=int(math.sqrt(num_iq_samples_per_signal)),
        impairment_level=2 if impaired else 0,
        num_signals_min=1,
        signal_duration_percent_min=100,
        seed=123456789
    )

    data_module = NarrowbandDataModule(
        root=root,
        dataset_metadata=dataset_metadata,
        num_samples_train=int(num_signals * 0.8),
        num_samples_val=int(num_signals * 0.2),
        batch_size=32,
        transforms=transforms,
        target_transforms=target_transforms,
        num_workers=os.cpu_count() if os.cpu_count() is not None else 3,
        overwrite=False,
    )

    data_module.prepare_data()
    data_module.setup()

    return dataset_metadata, data_module


def get_test_dataset(root: str, impaired: bool = False, num_signals: int | None = None, num_iq_samples_per_signal: int | None = None, transforms: list | None = None, target_transforms: list | None = None) -> tuple[NarrowbandMetadata, StaticNarrowband]:

    root = Path(root) / \
        f"torchsig_narrowband_{'impaired' if impaired else 'clean'}"
    test_root = root / "test"

    try:
        test_meta = to_dataset_metadata(
            root / 'create_dataset_info.yaml')
    except ValueError:

        assert num_iq_samples_per_signal is not None and num_signals is not None, "num_iq_samples_per_signal and num_signals must be provided if the dataset does not exist"

        test_meta = NarrowbandMetadata(
            num_samples=num_signals,
            num_iq_samples_dataset=num_iq_samples_per_signal,
            # fft size chosen so that one can generate a square spectrogram
            fft_size=int(math.sqrt(num_iq_samples_per_signal)),
            impairment_level=2 if impaired else 0,
            num_signals_min=1,
            signal_duration_percent_min=100,
            seed=123456789
        )
        dc = DatasetCreator(
            dataset=NewNarrowband(
                dataset_metadata=test_meta,
            ),
            root=test_root,
            overwrite=False,
            batch_size=32,
            num_workers=os.cpu_count() if os.cpu_count() is not None else 3,
        )
        dc.create()
        # if test_root / 'data' directory exists, rename it to 'test'
        if (test_root / 'data').exists():
            os.rename(test_root / 'data', test_root / 'test')

    test_dataset = StaticNarrowband(
        root=test_root,
        impairment_level=2 if impaired else 0,
        transforms=transforms,
        target_transforms=target_transforms,
    )

    return test_meta, test_dataset


@click.group()
def cli():
    pass


@cli.command
@click.option("--root", type=str, help="directory to save the dataset in")
@click.option("--impaired", is_flag=True, default=False)
@click.option("--num-signals", type=int, required=False)
@click.option("--num-iq-samples-per-signal", type=int, required=False)
def generate_train_val_datasets(root: str, impaired: bool = False, num_signals: int | None = None, num_iq_samples_per_signal: int | None = None) -> None:
    get_train_val_data_module(
        root=root,
        impaired=impaired,
        num_signals=num_signals,
        num_iq_samples_per_signal=num_iq_samples_per_signal
    )


@cli.command
@click.option("--root", type=str, help="directory to save the dataset in")
@click.option("--impaired", is_flag=True, default=False)
@click.option("--num-signals", type=int, required=False)
@click.option("--num-iq-samples-per-signal", type=int, required=False)
def generate_test_dataset(root: str, impaired: bool = False, num_signals: int | None = None, num_iq_samples_per_signal: int | None = None) -> None:
    get_test_dataset(
        root=root,
        impaired=impaired,
        num_signals=num_signals,
        num_iq_samples_per_signal=num_iq_samples_per_signal
    )


@cli.command
@click.option("--root", type=str, help="directory to save the dataset in")
@click.option("--impaired", is_flag=True, default=False)
@click.option("--num-signals", type=int, required=False)
@click.option("--num-iq-samples-per-signal", type=int, required=False)
def generate_all_datasets(root: str, impaired: bool = False, num_signals: int | None = None, num_iq_samples_per_signal: int | None = None) -> None:
    get_train_val_data_module(
        root=root,
        impaired=impaired,
        num_signals=num_signals,
        num_iq_samples_per_signal=num_iq_samples_per_signal
    )
    get_test_dataset(
        root=root,
        impaired=impaired,
        num_signals=int(num_signals*0.1),
        num_iq_samples_per_signal=num_iq_samples_per_signal
    )


if __name__ == "__main__":
    cli()
