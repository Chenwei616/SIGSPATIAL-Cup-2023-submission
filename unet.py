import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Hyperparameters

LEARNING_RATE = 1e-4
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MAX_EPOCHS = 1500 # enough epochs number for the target loss
TARGET_LOSS = 0.001
NUM_WORKERS = 4
PIN_MEMORY = True

class TifDataset(Dataset):
    def __init__(self, tif_array, mask_array, transform=None):
        self.transform = transform
        self.tif_array = tif_array
        self.mask_array = mask_array

    def __len__(self):
        return self.tif_array.shape[0]

    def __getitem__(self, index):
        image = self.tif_array[index]
        mask = self.mask_array[index]
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        return image, mask


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNET(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        features=[32, 64, 128, 256, 512],
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2,
                    feature,
                    kernel_size=2,
                    stride=2,
                )
            )
            self.ups.append(DoubleConv(feature * 2, feature))
            self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
            self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)
        return self.final_conv(x)


def test():
    x = torch.rand((3, 1, 161, 161))
    model = UNET(in_channels=1, out_channels=1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert preds.shape == x.shape


def save_checkpoint(state, filename="model/model.pth.tar"):
    print("=> Saving Checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading Checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def get_loaders(
    train_tif,
    train_mask,
    val_tif,
    val_mask,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    if len(train_tif) == 0:
        train_tif = np.zeros((1, 256, 256, 3))
        train_mask = np.zeros((1, 256, 256))
    if len(val_tif) == 0:
        val_tif = np.zeros((1, 256, 256, 3))
        val_mask = np.zeros((1, 256, 256))

    train_ds = TifDataset(
        transform=train_transform,
        tif_array=train_tif,
        mask_array=train_mask,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = TifDataset(
        transform=val_transform,
        tif_array=val_tif,
        mask_array=val_mask,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader


def pred_sliced_dataset(loader, model, device=DEVICE):
    preds = []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            pred = torch.sigmoid(model(x))
            pred = (pred > 0.5).float()
            pred = pred.cpu().numpy()
            preds.append(pred)
            # print(pred.shape)
        preds = np.concatenate(preds, axis=0)
        preds = np.squeeze(preds, axis=1)
        preds = preds.astype(np.uint8)
        # print(preds.shape)
        model.train()
    return preds

def train_fn(loader, model, optimizer, loss_fn, scaler, epoch, print_tqdm=False):
    if print_tqdm == True:
        loop = tqdm(loader, desc=f"Epoch {epoch} training")
    else:
        loop = loader

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)
        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)
        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # update tqdm loop
        if print_tqdm == True:
            loop.set_postfix(loss=loss.item())
    return loss.item()


def run_unet(
    train_tif, train_mask, val_tif, val_mask, print_tqdm=False, BATCH_SIZE=64
):
    train_transform = A.Compose(
        [
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            # A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        train_tif,
        train_mask,
        val_tif,
        val_mask,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if len(train_tif) != 0:
        scaler = torch.cuda.amp.GradScaler()
        for epoch in range(MAX_EPOCHS):
            loss = train_fn(
                train_loader, model, optimizer, loss_fn, scaler, epoch, print_tqdm
            )
            print(f"Epoch {epoch + 1} loss: {loss:.4f}")
            if (epoch + 1) % 10 == 0:
                checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),}
                save_checkpoint(checkpoint)

            if loss < TARGET_LOSS:
                print(
                    f"Training stopped: Desired loss {TARGET_LOSS} reached: epoch {epoch + 1}, loss {loss}."
                )
                torch.cuda.empty_cache()
                break
            elif epoch == (MAX_EPOCHS - 1):
                print(
                    f"Training stopped: Max epochs {MAX_EPOCHS} reached: epoch {epoch + 1}, loss {loss}."
                )
            torch.cuda.empty_cache()

        checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),}
        save_checkpoint(checkpoint)

    if len(val_tif) != 0:
        load_checkpoint(torch.load("model/model.pth.tar"), model)
        sliced_dataset = pred_sliced_dataset(val_loader, model, device=DEVICE)
        return sliced_dataset

    return None
