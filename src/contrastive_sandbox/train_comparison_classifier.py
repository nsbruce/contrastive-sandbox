from torchsig.transforms.target_transforms import ClassIndex
from torchsig.transforms.dataset_transforms import ComplexTo2D
from torchsig.models import XCiTClassifier
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from contrastive_sandbox.datasets import get_test_dataset, get_train_val_data_module


print("Loading training and validation datasets")
train_val_metadata, train_val_data_module = get_train_val_data_module(
    root='.data',
    transforms=[ComplexTo2D()],
    target_transforms=[ClassIndex()],
)


model = XCiTClassifier(
    input_channels=2,
    num_classes=len(train_val_metadata.class_list),
)

num_epochs = 100

checkpoint_callback = ModelCheckpoint(dirpath='.models/xcit-comparison',
                                      every_n_epochs=1, mode="min", monitor="val_loss", save_top_k=3, save_last=True)

logger = pl.loggers.TensorBoardLogger(
    save_dir='.logs',
    name='xcit-comparison',
)
trainer = pl.Trainer(
    max_epochs=num_epochs,
    accelerator='auto',
    devices='auto',
    callbacks=[checkpoint_callback],
    logger=logger,
)

trainer.fit(model, train_val_data_module)

print("Emptying cache")
torch.cuda.empty_cache()

print("Loading test dataset")
test_metadata, test_dataset = get_test_dataset(
    root='.data',
    transforms=[ComplexTo2D()],
    target_transforms=[ClassIndex()],
)

print("Loading best model")
model = XCiTClassifier.load_from_checkpoint(
    checkpoint_callback.best_model_path,
    input_channels=2,
    num_classes=len(train_val_metadata.class_list),
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# We can do this over the whole test dataset to check to accurarcy of our model
num_correct = 0

print("Testing model")
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
print(f"Percent Correct = {100 * num_correct / len(test_dataset)}%")
