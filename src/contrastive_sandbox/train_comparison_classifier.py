from contrastive_sandbox.encoders import XCiT
import torch
import pytorch_lightning as pl
from torchsig.datasets.narrowband import StaticNarrowband
from pytorch_lightning.callbacks import ModelCheckpoint

num_epochs = 10

checkpoint_callback = ModelCheckpoint(dirpath='.models/xcit-comparison', every_n_epochs=1, mode="min", monitor="val_loss", save_top_k=3,save_last=True)

trainer = pl.Trainer(
    max_epochs = num_epochs,
    accelerator = 'gpu'
)

model = XCiT(num_channels = 2, num_classes=)

dataset = StaticNarrowband('.data', impaired=False)

num_classes = len(dataset.dataset_metadata.class_list)

trainer.fit(model, dataset)
