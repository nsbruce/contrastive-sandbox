from torchsig.transforms.dataset_transforms import ComplexTo2D
from torchsig.transforms.target_transforms import ClassIndex
from torchsig.datasets.narrowband import StaticNarrowband, NewNarrowband
from torchsig.datasets.dataset_metadata import NarrowbandMetadata
from torchsig.datasets.datamodules import NarrowbandDataModule


def load_datamodule(
    root: str,
    batch_size: int,
) -> NarrowbandDataModule:
    transforms = [ComplexTo2D()]
    target_transforms = [ClassIndex()]

    dataset = StaticNarrowband(root, impaired=False)

    datamodule = NarrowbandDataModule(
        root=root,
        dataset_metadata=dataset.dataset_metadata,
        num_samples_train=int(dataset.num_samples * 0.8),
        num_samples_val=int(dataset.num_samples * 0.2),
    )

    return datamodule


def generate_dataset(metadata: NarrowbandMetadata):

    return NewNarrowband(dataset_metadata=metadata)
