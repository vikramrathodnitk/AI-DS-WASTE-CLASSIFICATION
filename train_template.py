
"""
Waste Classification Using CNN
--------------------------------
This script trains a CNN model to classify waste images into six categories:

1. cardboard
2. glass
3. metal
4. paper
5. plastic
6. trash

Dataset folder structure:
data/public/
    train/
        cardboard/
        glass/
        metal/
        paper/
        plastic/
        trash/
    val/
        cardboard/
        glass/
        metal/
        paper/
        plastic/
        trash/

Input image size: 96 x 96 RGB
Model output: 6 class scores

Run:
python train_template.py --data_dir data/public --epochs 10 --batch_size 32 --lr 0.001 --output model.onnx
"""

import argparse
import copy
import os
import random

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ==========================================
# 1. CONFIGURATION
# ==========================================

# Keep this order the same as the model output classes.
CLASSES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

IMAGE_SIZE = 96

# ImageNet normalization values used during training.
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def set_seed(seed=42):
    """Set random seeds to make training more reproducible."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==========================================
# 2. CNN BUILDING BLOCKS
# ==========================================

class ConvBNAct(nn.Module):
    """
    A basic CNN block containing:
    Convolution -> Batch Normalization -> SiLU activation.
    """

    def __init__(self, in_ch, out_ch, kernel=3, stride=1):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=kernel,
                stride=stride,
                padding=kernel // 2,
                bias=False
            ),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class SqueezeExcitation(nn.Module):
    """
    Helps the network focus on more useful feature channels.
    """

    def __init__(self, channels, reduction=4):
        super().__init__()

        hidden = max(channels // reduction, 8)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attention = self.fc(self.pool(x))
        return x * attention


class DepthwiseSeparableBlock(nn.Module):
    """
    Uses depthwise and pointwise convolutions to extract
    image features efficiently.
    """

    def __init__(self, in_ch, out_ch, stride=1, use_se=True):
        super().__init__()

        # Depthwise convolution processes each input channel separately.
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                in_ch,
                in_ch,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_ch,
                bias=False
            ),
            nn.BatchNorm2d(in_ch),
            nn.SiLU(inplace=True)
        )

        # Pointwise convolution combines information between channels.
        self.pointwise = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(out_ch)
        )

        # Apply channel attention when enabled.
        self.se = (
            SqueezeExcitation(out_ch)
            if use_se else nn.Identity()
        )

        # Shortcut helps information pass through the block.
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(
                    in_ch,
                    out_ch,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_ch)
            )
            if stride != 1 or in_ch != out_ch
            else nn.Identity()
        )

        self.activation = nn.SiLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.se(out)

        return self.activation(out + identity)


# ==========================================
# 3. CNN CLASSIFICATION MODEL
# ==========================================

class ImprovedClassifier(nn.Module):
    """
    CNN model for classifying images into six waste categories.
    The original model architecture is preserved.
    """

    def __init__(self, num_classes=6):
        super().__init__()

        # Initial convolution layer.
        self.stem = ConvBNAct(
            3, 24, kernel=3, stride=2
        )

        # Five feature extraction stages.
        self.features = nn.Sequential(

            # Stage 1
            DepthwiseSeparableBlock(
                24, 24, stride=1
            ),

            # Stage 2
            DepthwiseSeparableBlock(
                24, 32, stride=2
            ),
            DepthwiseSeparableBlock(
                32, 32, stride=1
            ),

            # Stage 3
            DepthwiseSeparableBlock(
                32, 64, stride=2
            ),
            DepthwiseSeparableBlock(
                64, 64, stride=1
            ),

            # Stage 4
            DepthwiseSeparableBlock(
                64, 96, stride=2
            ),
            DepthwiseSeparableBlock(
                96, 96, stride=1
            ),

            # Stage 5
            DepthwiseSeparableBlock(
                96, 128, stride=2
            ),
            DepthwiseSeparableBlock(
                128, 128, stride=1
            )
        )

        # Converts each feature channel into one value.
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Final layers produce six class scores.
        self.classifier = nn.Sequential(
            nn.Dropout(0.30),
            nn.Linear(128, 64),
            nn.SiLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):

        x = self.stem(x)
        x = self.features(x)
        x = self.pool(x)

        # Convert feature maps into a one-dimensional feature vector.
        x = torch.flatten(x, 1)

        return self.classifier(x)


# ==========================================
# 4. DATA LOADING AND AUGMENTATION
# ==========================================

def get_dataloaders(data_dir, batch_size=32):
    """
    Load training and validation images from their folders.
    Apply data augmentation only to training images.
    """

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(
            f"Training directory not found: {train_dir}"
        )

    if not os.path.isdir(val_dir):
        raise FileNotFoundError(
            f"Validation directory not found: {val_dir}"
        )

    # Random transformations help the model learn from varied images.
    train_transform = transforms.Compose([

        transforms.RandomResizedCrop(
            IMAGE_SIZE,
            scale=(0.75, 1.0),
            ratio=(0.85, 1.15)
        ),

        transforms.RandomHorizontalFlip(p=0.5),

        transforms.RandomRotation(12),

        transforms.ColorJitter(
            brightness=0.20,
            contrast=0.20,
            saturation=0.15,
            hue=0.03
        ),

        transforms.ToTensor(),

        transforms.Normalize(MEAN, STD),

        transforms.RandomErasing(
            p=0.15,
            scale=(0.02, 0.12)
        )
    ])

    # Validation images are resized and normalized without random changes.
    val_transform = transforms.Compose([

        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

        transforms.ToTensor(),

        transforms.Normalize(MEAN, STD)
    ])

    train_dataset = datasets.ImageFolder(
        train_dir,
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        val_dir,
        transform=val_transform
    )

    # Make sure folder names match the model's output class order.
    expected_mapping = {
        name: i for i, name in enumerate(CLASSES)
    }

    if train_dataset.class_to_idx != expected_mapping:
        raise ValueError(
            f"Incorrect training classes: "
            f"{train_dataset.class_to_idx}"
        )

    if val_dataset.class_to_idx != expected_mapping:
        raise ValueError(
            f"Incorrect validation classes: "
            f"{val_dataset.class_to_idx}"
        )

    if len(train_dataset) == 0:
        raise ValueError("The training dataset contains no images.")

    if len(val_dataset) == 0:
        raise ValueError("The validation dataset contains no images.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    print("\nTraining image distribution:")

    for name, index in expected_mapping.items():

        count = sum(
            1 for _, label in train_dataset.samples
            if label == index
        )

        print(f"{name}: {count} images")

    return train_loader, val_loader


# ==========================================
# 5. TRAINING FUNCTION
# ==========================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    scaler
):
    """Train the model for one complete epoch."""

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        # Mixed precision is enabled when training on a GPU.
        with torch.cuda.amp.autocast(
            enabled=(device.type == "cuda")
        ):

            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        # Prevent excessively large gradients.
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / max(total, 1)
    accuracy = 100.0 * correct / max(total, 1)

    return epoch_loss, accuracy


# ==========================================
# 6. VALIDATION FUNCTION
# ==========================================

@torch.no_grad()
def validate(model, loader, criterion, device):
    """Evaluate the model using validation images."""

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    val_loss = running_loss / max(total, 1)
    val_accuracy = 100.0 * correct / max(total, 1)

    return val_loss, val_accuracy


# ==========================================
# 7. ONNX MODEL EXPORT
# ==========================================

def export_to_onnx(model, output_path):
    """
    Export the trained PyTorch model into ONNX format.
    ONNX allows the model to be used by other applications.
    """

    model.eval()
    model.cpu()

    # Create a sample input with the expected image shape.
    dummy_input = torch.randn(
        1, 3, IMAGE_SIZE, IMAGE_SIZE
    )

    # Create the output folder if it does not already exist.
    output_directory = os.path.dirname(
        os.path.abspath(output_path)
    )
    os.makedirs(output_directory, exist_ok=True)

    print("\nExporting ONNX model...")

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        },
        dynamo=False
    )

    size_mb = os.path.getsize(output_path) / (
        1024 * 1024
    )

    print(f"ONNX file saved: {output_path}")
    print(f"ONNX file size: {size_mb:.3f} MB")


# ==========================================
# 8. MAIN TRAINING FUNCTION
# ==========================================

def main():

    parser = argparse.ArgumentParser(
        description="Train a CNN for waste image classification."
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/public",
        help="Folder containing train and val directories."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="Number of training epochs."
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Number of images processed in one batch."
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate for the optimizer."
    )

    parser.add_argument(
        "--output",
        type=str,
        default="model.onnx",
        help="Path where the ONNX model will be saved."
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=8,
        help="Number of epochs to wait before early stopping."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility."
    )

    args = parser.parse_args()

    if args.epochs <= 0:
        parser.error("--epochs must be greater than zero.")

    if args.batch_size <= 0:
        parser.error("--batch_size must be greater than zero.")

    if args.lr <= 0:
        parser.error("--lr must be greater than zero.")

    if args.patience <= 0:
        parser.error("--patience must be greater than zero.")

    set_seed(args.seed)

    # Automatically select GPU when available; otherwise use CPU.
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 55)
    print("       WASTE CLASSIFICATION MODEL TRAINING")
    print("=" * 55)

    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # Load the training and validation datasets.
    train_loader, val_loader = get_dataloaders(
        args.data_dir,
        args.batch_size
    )

    print("\nTraining images:", len(train_loader.dataset))
    print("Validation images:", len(val_loader.dataset))

    # Create the CNN model.
    model = ImprovedClassifier(
        num_classes=len(CLASSES)
    ).to(device)

    # Loss function measures the difference between predictions and labels.
    criterion = nn.CrossEntropyLoss(
        label_smoothing=0.05
    )

    # AdamW updates the model's trainable parameters.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )

    # Adjust the learning rate during training.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01
    )

    # Mixed precision scaler for GPU training.
    scaler = torch.cuda.amp.GradScaler(
        enabled=(device.type == "cuda")
    )

    # Track the best model using validation accuracy and loss.
    best_val_acc = 0.0
    best_val_loss = float("inf")

    best_weights = copy.deepcopy(model.state_dict())

    patience_counter = 0

    # Save the best weights in the project folder.
    checkpoint_path = "best_model.pth"

    print("\nStarting training...\n")

    for epoch in range(1, args.epochs + 1):

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler
        )

        val_loss, val_acc = validate(
            model,
            val_loader,
            criterion,
            device
        )

        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] "
            f"| Train Loss: {train_loss:.4f} "
            f"| Train Acc: {train_acc:.2f}% "
            f"| Val Loss: {val_loss:.4f} "
            f"| Val Acc: {val_acc:.2f}% "
            f"| LR: {current_lr:.6f}"
        )

        # A better model has higher validation accuracy.
        # If accuracy is equal, prefer the lower validation loss.
        improved = (
            val_acc > best_val_acc
            or (
                val_acc == best_val_acc
                and val_loss < best_val_loss
            )
        )

        if improved:

            best_val_acc = val_acc
            best_val_loss = val_loss

            best_weights = copy.deepcopy(
                model.state_dict()
            )

            patience_counter = 0

            torch.save(
                best_weights,
                checkpoint_path
            )

            print("  -> Best model saved!")

        else:

            patience_counter += 1

        # Stop training when validation performance does not improve
        # for the specified number of consecutive epochs.
        if patience_counter >= args.patience:

            print("\nEarly stopping activated.")
            break

    # Restore the model weights with the best validation performance.
    model.load_state_dict(best_weights)

    print("\n" + "=" * 55)
    print("TRAINING COMPLETED")
    print("=" * 55)

    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Best weights saved at: {checkpoint_path}")

    # Export the best model to ONNX format.
    export_to_onnx(
        model,
        args.output
    )

    print("\nNext step:")
    print(
        f"python verify_submission.py --model {args.output} "
        f"--val_dir {os.path.join(args.data_dir, 'val')}"
    )


if __name__ == "__main__":
    main()
