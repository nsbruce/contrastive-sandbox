# from contrastive_sandbox.encoders import XCiT
# from torchsig.models import XCiTClassifier
# import torch
# import pytorch_lightning as pl
# from torchsig.datasets.narrowband import StaticNarrowband
# from pytorch_lightning.callbacks import ModelCheckpoint
# from torchsig.datasets.datamodules import NarrowbandDataModule
# import yaml

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
from torchsig.utils.writer import DatasetCreator
from torchsig.datasets.narrowband import NewNarrowband, StaticNarrowband
import pytorch_lightning as pl
import torch
from torchinfo import summary
from torchsig.models import XCiTClassifier
from torchsig.datasets.datamodules import NarrowbandDataModule
from torchsig.datasets.dataset_metadata import NarrowbandMetadata
from torchsig.signals.signal_lists import TorchSigSignalLists
from torchsig.transforms.dataset_transforms import ComplexTo2D
from torchsig.transforms.target_transforms import ClassIndex

root = "./datasets/narrowband_classifier_example"
fft_size = 256
num_iq_samples_dataset = fft_size ** 2
class_list = TorchSigSignalLists.all_signals
num_classes = len(class_list)
num_samples_train = len(class_list) * 10  # roughly 10 samples per class
num_samples_val = len(class_list) * 2
impairment_level = 0
seed = 123456789

# ComplexTo2D turns a IQ array of complex values into a 2D array, with one channel for the real component, while the other is for the imaginary component
transforms = [ComplexTo2D()]
# ClassIndex turns our target labels into the index of the class according to class_list
target_transforms = [ClassIndex()]


dataset_metadata = NarrowbandMetadata(
    num_iq_samples_dataset=num_iq_samples_dataset,
    fft_size=fft_size,
    impairment_level=impairment_level,
    class_list=class_list,
    seed=seed
)

narrowband_datamodule = NarrowbandDataModule(
    root=root,
    dataset_metadata=dataset_metadata,
    num_samples_train=num_samples_train,
    num_samples_val=num_samples_val,
    transforms=transforms,
    target_transforms=target_transforms,
    create_batch_size=4,
    create_num_workers=4,
    batch_size=4,
    num_workers=4,
    # overwrite = True
)
narrowband_datamodule.prepare_data()
narrowband_datamodule.setup()

data, targets = narrowband_datamodule.train[0]
print(f"Data shape: {data.shape}")
print(f"Targets: {targets}")


model = XCiTClassifier(
    input_channels=2,
    num_classes=num_classes,
)
summary(model)


num_epochs = 5

trainer = pl.Trainer(
    max_epochs=num_epochs,
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    devices=1
)
# print(trainer)

trainer.fit(model, narrowband_datamodule)

torch.cuda.empty_cache()

test_dataset_size = 10

dataset_metadata_test = NarrowbandMetadata(
    num_iq_samples_dataset=num_iq_samples_dataset,
    fft_size=fft_size,
    impairment_level=impairment_level,
    class_list=class_list,
    num_samples=test_dataset_size,
    transforms=transforms,
    target_transforms=target_transforms,
    seed=123456788  # different than train
)
# print(dataset_metadata_test)

dc = DatasetCreator(
    dataset=NewNarrowband(
        dataset_metadata=dataset_metadata_test,
    ),
    root=f"{root}/test",
    overwrite=True,
    batch_size=1,
    num_workers=1,
)
dc.create()

test_narrowband = StaticNarrowband(
    root=f"{root}/test",
    impairment_level=impairment_level,
)


data, class_index = test_narrowband[0]
print(f"Data shape: {data.shape}")
print(f"Targets: {targets}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data, class_index = test_narrowband[0]
# move to model to the same device as the data
model.to(device)
# turn the model into evaluation mode
model.eval()
with torch.no_grad():  # do not update model weights
    # convert to tensor and add a batch dimension
    data = torch.from_numpy(data).to(device).unsqueeze(dim=0)
    # have model predict data
    # returns a probability the data is each signal class
    pred = model(data)
    # print(pred) # if you want to see the list of probabilities

    # choose the class with highest confidence
    predicted_class = torch.argmax(pred).cpu().numpy()
    print(f"Predicted = {predicted_class} ({class_list[predicted_class]})")
    print(f"Actual = {class_index} ({class_list[class_index]})")


# We can do this over the whole test dataset to check to accurarcy of our model
num_correct = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for sample in test_narrowband:
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
print(f"Percent Correct = {num_correct / len(test_narrowband)}%")
