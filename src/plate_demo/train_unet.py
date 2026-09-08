from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    """ NOTE：①  """
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        return self.conv(torch.cat([x, skip], dim=1))
    """ END """


class UNetSmall(nn.Module):
    """ NOTE：②  """
    def __init__(self, in_channels: int = 9, out_channels: int = 1) -> None:
        super().__init__()
        self.enc1 = ConvBlock(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(128, 256)
        self.up3 = UpBlock(256, 128, 128)
        self.up2 = UpBlock(128, 64, 64)
        self.up1 = UpBlock(64, 32, 32)
        self.out = nn.Conv2d(32, out_channels, kernel_size=1)
    """ END """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))
        d3 = self.up3(b, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        # A linear output avoids imposing an artificial [0, 1] ceiling.  A
        # validation or test case can legitimately exceed the training-set
        # normalization maximum.
        return self.out(d1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a U-Net surrogate for stress distribution maps")
    parser.add_argument("--dataset", default="data/map_dataset.npz", help="rasterized map dataset path")
    parser.add_argument("--epochs", type=int, default=50, help="training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--seed", type=int, default=7, help="random seed")
    parser.add_argument("--output-dir", default="outputs/map_prediction", help="output directory")
    return parser.parse_args()


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """ NOTE：③  """
    weight = 1.0 + 4.0 * target
    diff = (pred - target) ** 2 * mask * weight
    denom = torch.clamp(mask.sum(), min=1.0)
    """ END """
    return diff.sum() / denom


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    total = 0.0
    """ NOTE：④  """
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        mask = (x[:, 0:1] > 0).float()
        optimizer.zero_grad()
        pred = model(x) * mask
        loss = masked_mse(pred, y, mask)
        loss.backward()
        optimizer.step()
        total += float(loss.item()) * x.size(0)
    """ END """
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    total = 0.0
    preds = []
    trues = []
    masks = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        mask = (x[:, 0:1] > 0).float()
        pred = model(x) * mask
        loss = masked_mse(pred, y, mask)
        total += float(loss.item()) * x.size(0)
        preds.append(pred.cpu().numpy())
        trues.append(y.cpu().numpy())
        masks.append(mask.cpu().numpy())
    pred_all = np.concatenate(preds, axis=0)
    true_all = np.concatenate(trues, axis=0)
    mask_all = np.concatenate(masks, axis=0)
    mae = float(np.abs((pred_all - true_all) * mask_all).sum() / np.clip(mask_all.sum(), 1.0, None))
    return total / len(loader.dataset), mae, pred_all, true_all, mask_all


def render_examples(output_dir: Path, inputs: np.ndarray, preds: np.ndarray, trues: np.ndarray, count: int = 3) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    stress_cmap = plt.colormaps["viridis"].copy()
    error_cmap = plt.colormaps["viridis"].copy()
    stress_cmap.set_bad(color="white")
    error_cmap.set_bad(color="white")
    for idx in range(min(count, len(inputs))):
        material = inputs[idx, 0] > 0.5
        pred_map = np.where(material, preds[idx, 0], np.nan)
        true_map = np.where(material, trues[idx, 0], np.nan)
        err_map = np.where(material, np.abs(preds[idx, 0] - trues[idx, 0]), np.nan)
        stress_vmax = float(np.nanmax([true_map, pred_map]))
        error_vmax = float(np.nanmax(err_map))
        fig, axes = plt.subplots(1, 4, figsize=(14, 4))
        im0 = axes[0].imshow(inputs[idx, 0], cmap="gray", vmin=0.0, vmax=1.0)
        axes[0].set_title("material")
        im1 = axes[1].imshow(true_map, cmap=stress_cmap, vmin=0.0, vmax=stress_vmax)
        axes[1].set_title("true stress")
        im2 = axes[2].imshow(pred_map, cmap=stress_cmap, vmin=0.0, vmax=stress_vmax)
        axes[2].set_title("predicted stress")
        im3 = axes[3].imshow(err_map, cmap=error_cmap, vmin=0.0, vmax=error_vmax)
        axes[3].set_title("abs error")
        for ax in axes:
            ax.axis("off")
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(plot_dir / f"sample_{idx:02d}.png", dpi=160)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = np.load(Path(args.dataset))
    inputs = torch.tensor(data["inputs"], dtype=torch.float32)
    targets = torch.tensor(data["targets"], dtype=torch.float32)
    dataset = TensorDataset(inputs, targets)

    if "splits" not in data:
        raise ValueError("dataset has no train/validation/test split; rerun make_map_dataset")
    splits = data["splits"].astype(str)
    train_set = Subset(dataset, np.flatnonzero(splits == "train").tolist())
    validation_set = Subset(dataset, np.flatnonzero(splits == "validation").tolist())
    test_set = Subset(dataset, np.flatnonzero(splits == "test").tolist())

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetSmall(in_channels=inputs.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=6)

    history = {"train_loss": [], "validation_loss": [], "validation_mae": []}
    best_loss = None
    best_validation_mae = None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = output_dir / "unet_stress_map.pt"

    """ NOTE：⑤  """
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        validation_loss, validation_mae, _, _, _ = evaluate(model, validation_loader, device)
        scheduler.step(validation_loss)
        history["train_loss"].append(train_loss)
        history["validation_loss"].append(validation_loss)
        history["validation_mae"].append(validation_mae)
        print(f"epoch {epoch:03d} train_loss={train_loss:.6f} validation_loss={validation_loss:.6f} validation_mae={validation_mae:.6f}")
        if best_loss is None or validation_loss < best_loss:
            best_loss = validation_loss
            best_validation_mae = validation_mae
            torch.save(model.state_dict(), best_model_path)
    """ END """

    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    test_loss, test_mae, test_preds, test_trues, _ = evaluate(model, test_loader, device)
    test_inputs = np.stack([test_set[i][0].numpy() for i in range(len(test_set))])
    metrics = {
        "best_validation_loss": best_loss,
        "best_validation_mae": best_validation_mae,
        "test_loss": test_loss,
        "test_mae": test_mae,
        "epochs": args.epochs,
        "device": str(device),
        "stress_scale": float(data["stress_scale"][0]) if "stress_scale" in data else 1.0,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")

    render_examples(output_dir, test_inputs, test_preds, test_trues)
    render_history(output_dir / "plots" / "loss_curve.png", history)
    print(f"wrote {output_dir}")


def render_history(plot_path: Path, history: dict[str, list[float]]) -> None:
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history["train_loss"], label="train")
    ax.plot(history["validation_loss"], label="validation")
    ax.set_xlabel("epoch")
    ax.set_ylabel("masked MSE")
    ax.set_title("U-Net training history")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
