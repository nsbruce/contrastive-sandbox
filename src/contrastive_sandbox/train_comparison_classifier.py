from contrastive_sandbox.encoders import XCiT
from torchsig.models import XCiTClassifier
import torch
import pytorch_lightning as pl
from torchsig.datasets.narrowband import StaticNarrowband
from pytorch_lightning.callbacks import ModelCheckpoint

num_epochs = 10

print("Loading dataset")
dataset = StaticNarrowband('.data', impairment_level=0)

checkpoint_callback = ModelCheckpoint(dirpath='.models/xcit-comparison',
                                      every_n_epochs=1, mode="min", monitor="val_loss", save_top_k=3, save_last=True)

trainer = pl.Trainer(
    max_epochs=num_epochs,
    accelerator='gpu'
)

print("Building model")
model = XCiTClassifier(input_channels=2, num_classes=len(
    dataset.dataset_metadata.class_list))

print("Training model")
trainer.fit(model, dataset)
