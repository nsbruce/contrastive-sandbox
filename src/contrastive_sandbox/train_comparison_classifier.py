from torchsig.transforms.target_transforms import ClassIndex
from torchsig.transforms.dataset_transforms import ComplexTo2D
from torchsig.signals.signal_lists import TorchSigSignalLists
from torchsig.datasets.dataset_metadata import NarrowbandMetadata
from torchinfo import summary
from torchsig.datasets.narrowband import NewNarrowband, StaticNarrowband
from torchsig.utils.writer import DatasetCreator
from contrastive_sandbox.encoders import XCiT
from torchsig.models import XCiTClassifier
import torch
import pytorch_lightning as pl
from torchsig.datasets.narrowband import StaticNarrowband
from pytorch_lightning.callbacks import ModelCheckpoint
from torchsig.datasets.datamodules import NarrowbandDataModule
import yaml
from contrastive_sandbox.datasets import get_test_dataset, get_train_val_data_module
# num_epochs = 10

# with open('.data/torchsig_narrowband_clean/create_dataset_info.yaml', 'r') as f:
#     metadata = yaml.safe_load(f)

# print(metadata)
# print("Loading dataset into data module")
# # dataset = StaticNarrowband('.data', impairment_level=0)
# data_module = NarrowbandDataModule(
#     root='.data',
#     dataset_metadata=metadata.copy(),
#     num_samples_train=int(metadata['overrides']['num_samples']*0.8),
#     num_samples_val=int(metadata['overrides']['num_samples']*0.2),
#     batch_size=32,
#     num_workers=4,
# )


# checkpoint_callback = ModelCheckpoint(dirpath='.models/xcit-comparison',
#                                       every_n_epochs=1, mode="min", monitor="val_loss", save_top_k=3, save_last=True)

# trainer = pl.Trainer(
#     max_epochs=num_epochs,
#     accelerator='gpu'
# )

# print("Building model")
# print(metadata)
# model = XCiTClassifier(input_channels=2, num_classes=len(
#     metadata['read_only']['signals']['class_list']))

# print("Training model")
# trainer.fit(model, data_module)
# Variables

train_val_metadata, train_val_data_module = get_train_val_data_module(
    root='.data',
    transforms=[ComplexTo2D()],
    target_transforms=[ClassIndex()],
)

test_metadata, test_dataset = get_test_dataset(
    root='.data',
    transforms=[ComplexTo2D()],
    target_transforms=[ClassIndex()],
)

model = XCiTClassifier(
    input_channels=2,
    num_classes=len(train_val_metadata.class_list),
)

num_epochs = 10

trainer = pl.Trainer(
    max_epochs=5,
    accelerator='auto',
    devices=1
)

trainer.fit(model, train_val_data_module)

torch.cuda.empty_cache()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# We can do this over the whole test dataset to check to accurarcy of our model
num_correct = 0

for sample in test_dataset:
    data, actual_class = sample
    model.to(device)
    model.eval()
    with torch.no_grad():
        data = torch.from_numpy(data).to(device).unsqueeze(dim=0)
        pred = model(data)
        predicted_class = torch.argmax(pred).cpu().numpy()
        if predicted_class == actual_class:
            num_correct += 1

# try increasing num_epochs or train dataset size to increase accuracy
print(f"Correct Predictions = {num_correct}")
print(f"Percent Correct = {num_correct / len(test_dataset)}%")
