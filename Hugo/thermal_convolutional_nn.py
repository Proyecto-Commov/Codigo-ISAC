from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# 1. Clases térmicas
# =============================================================================

CLASS_NAMES = ("frío", "templado", "caliente")

CLASS_TO_IDX = {
    "frío": 0,
    "templado": 1,
    "caliente": 2,
}

IDX_TO_CLASS = {
    0: "frío",
    1: "templado",
    2: "caliente",
}


# =============================================================================
# 2. Conversión de H compleja a imagen de dos canales
# =============================================================================

def complex_H_to_2ch_image(
    H: np.ndarray,
    *,
    input_order: Literal["mk", "km"] = "mk",
) -> np.ndarray:
    """
    Convierte una matriz compleja H en una imagen de dos canales:

        canal 0: parte real
        canal 1: parte imaginaria

    Entrada
    -------
    H : np.ndarray
        Matriz compleja de canal.

        Si input_order="mk":
            H.shape = (M, N)
            H[m, k]

        Si input_order="km":
            H.shape = (N, M)
            H[k, m]

    Salida
    ------
    img : np.ndarray
        Imagen real con shape (2, M, N).

        img[0, :, :] = Re(H)
        img[1, :, :] = Im(H)
    """

    H = np.asarray(H)

    if H.ndim != 2:
        raise ValueError(f"H debe ser 2D, recibido shape={H.shape}")

    if not np.iscomplexobj(H):
        raise TypeError("H debe ser una matriz compleja.")

    if input_order == "mk":
        H_mk = H
    elif input_order == "km":
        H_mk = H.T
    else:
        raise ValueError("input_order debe ser 'mk' o 'km'.")

    real = np.nan_to_num(
        H_mk.real,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    imag = np.nan_to_num(
        H_mk.imag,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    img = np.stack([real, imag], axis=0).astype(np.float32)

    return img


# =============================================================================
# 3. Normalización de imágenes complejas
# =============================================================================

@dataclass
class ComplexImageStandardizer:
    """
    Normaliza los dos canales de entrada:

        canal 0: Re(H)
        canal 1: Im(H)

    Se ajusta usando únicamente el conjunto de entrenamiento.
    """

    mean: np.ndarray | None = None
    std: np.ndarray | None = None
    eps: float = 1e-8

    def fit(
        self,
        H_list: Sequence[np.ndarray],
        *,
        input_order: Literal["mk", "km"] = "mk",
    ) -> "ComplexImageStandardizer":
        """
        Calcula media y desviación típica global por canal.
        """

        sums = np.zeros(2, dtype=np.float64)
        sums_sq = np.zeros(2, dtype=np.float64)
        counts = np.zeros(2, dtype=np.float64)

        for H in H_list:
            img = complex_H_to_2ch_image(H, input_order=input_order)

            for c in range(2):
                x = img[c].ravel().astype(np.float64)

                sums[c] += np.sum(x)
                sums_sq[c] += np.sum(x**2)
                counts[c] += x.size

        mean = sums / np.maximum(counts, 1.0)
        var = sums_sq / np.maximum(counts, 1.0) - mean**2
        var = np.maximum(var, 0.0)

        std = np.sqrt(var)
        std = np.where(std < self.eps, 1.0, std)

        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

        return self

    def transform_image(self, img: np.ndarray) -> np.ndarray:
        """
        Normaliza una imagen con shape (2, M, N).
        """

        if self.mean is None or self.std is None:
            raise RuntimeError("El normalizador no está ajustado. Llama antes a fit(...).")

        img = np.asarray(img, dtype=np.float32)

        if img.ndim != 3 or img.shape[0] != 2:
            raise ValueError(f"img debe tener shape (2, M, N), recibido {img.shape}")

        mean = self.mean.reshape(2, 1, 1)
        std = self.std.reshape(2, 1, 1)

        return (img - mean) / std


@dataclass
class TargetStandardizer:
    """
    Normalizador para la temperatura real.

    La red predice temperatura normalizada durante el entrenamiento.
    Después se desnormaliza para obtener ºC.
    """

    mean: float | None = None
    std: float | None = None
    eps: float = 1e-8

    def fit(self, y: np.ndarray) -> "TargetStandardizer":
        y = np.asarray(y, dtype=np.float32).reshape(-1)

        self.mean = float(y.mean())
        self.std = float(y.std())

        if self.std < self.eps:
            self.std = 1.0

        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("El normalizador de temperatura no está ajustado.")

        y = np.asarray(y, dtype=np.float32).reshape(-1)
        return (y - self.mean) / self.std

    def inverse_transform(self, y_norm: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("El normalizador de temperatura no está ajustado.")

        y_norm = np.asarray(y_norm, dtype=np.float32).reshape(-1)
        return y_norm * self.std + self.mean


# =============================================================================
# 4. Dataset
# =============================================================================

class ThermalHDataset(Dataset):
    """
    Dataset para entrenar directamente con matrices H complejas.

    Cada muestra:
        H_i       -> matriz compleja, shape (M, N)
        y_class_i -> clase térmica:
                        0 = frío
                        1 = templado
                        2 = caliente
        y_temp_i  -> temperatura real en ºC

    La CNN recibe:
        x_i -> tensor real, shape (2, M, N)
    """

    def __init__(
        self,
        H_list: Sequence[np.ndarray],
        y_class: Sequence[int],
        y_temp: Sequence[float],
        *,
        input_order: Literal["mk", "km"] = "mk",
        image_standardizer: ComplexImageStandardizer | None = None,
        resize_to: tuple[int, int] | None = None,
    ):
        if len(H_list) != len(y_class) or len(H_list) != len(y_temp):
            raise ValueError(
                "H_list, y_class e y_temp deben tener el mismo número de muestras."
            )

        self.H_list = list(H_list)
        self.y_class = np.asarray(y_class, dtype=np.int64).reshape(-1)
        self.y_temp = np.asarray(y_temp, dtype=np.float32).reshape(-1)

        self.input_order = input_order
        self.image_standardizer = image_standardizer
        self.resize_to = resize_to

    def __len__(self) -> int:
        return len(self.H_list)

    def __getitem__(self, idx: int):
        H = self.H_list[idx]

        img = complex_H_to_2ch_image(
            H,
            input_order=self.input_order,
        )

        if self.image_standardizer is not None:
            img = self.image_standardizer.transform_image(img)

        x = torch.tensor(img, dtype=torch.float32)

        # Si las matrices H no tienen todas el mismo tamaño, se puede forzar
        # un tamaño común mediante interpolación bilineal.
        #
        # Por ejemplo:
        #   resize_to=(100, 1024)
        #
        # Si todas tus H ya son (100, 1024), deja resize_to=None.
        if self.resize_to is not None:
            x = x.unsqueeze(0)  # (1, 2, M, N)
            x = F.interpolate(
                x,
                size=self.resize_to,
                mode="bilinear",
                align_corners=False,
            )
            x = x.squeeze(0)  # (2, M_resize, N_resize)

        return {
            "x": x,
            "y_class": torch.tensor(self.y_class[idx], dtype=torch.long),
            "y_temp": torch.tensor(self.y_temp[idx], dtype=torch.float32),
        }


# =============================================================================
# 5. Bloques CNN
# =============================================================================

class ConvBlock(nn.Module):
    """
    Bloque convolucional básico:

        Conv2d
        BatchNorm2d
        SiLU
        Conv2d
        BatchNorm2d
        SiLU
        MaxPool2d

    La operación MaxPool2d reduce las dimensiones espaciales aproximadamente
    a la mitad.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        dropout2d: float = 0.0,
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),

            nn.Dropout2d(dropout2d),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# =============================================================================
# 6. Red CNN multitarea
# =============================================================================

class ThermalHCNNMultiTaskNet(nn.Module):
    """
    CNN multitarea para matriz de canal compleja H.

    Entrada:
        x.shape = (B, 2, M, N)

        Canal 0: Re(H)
        Canal 1: Im(H)

    Salidas:
        class_logits.shape = (B, 3)
            logits para frío / templado / caliente

        temp_pred.shape = (B,)
            temperatura normalizada
    """

    def __init__(
        self,
        dropout2d: float = 0.05,
        dropout_fc: float = 0.20,
    ):
        super().__init__()

        self.cnn = nn.Sequential(
            ConvBlock(2, 16, dropout2d=dropout2d),
            ConvBlock(16, 32, dropout2d=dropout2d),
            ConvBlock(32, 64, dropout2d=dropout2d),
            ConvBlock(64, 128, dropout2d=dropout2d),
        )

        # Hace que la red acepte imágenes H de distinto tamaño,
        # siempre que sean suficientemente grandes.
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.shared_fc = nn.Sequential(
            nn.Flatten(),          # (B, 128, 1, 1) -> (B, 128)
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Dropout(dropout_fc),
        )

        self.class_head = nn.Linear(64, 3)
        self.temp_head = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.cnn(x)
        z = self.global_pool(z)
        z = self.shared_fc(z)

        class_logits = self.class_head(z)
        temp_pred = self.temp_head(z).squeeze(-1)

        return {
            "class_logits": class_logits,
            "temp_pred": temp_pred,
        }


# =============================================================================
# 7. Entrenamiento y evaluación
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: str | torch.device = "cpu",
    lambda_temp: float = 1.0,
) -> dict[str, float]:
    """
    Entrena una época.

    Pérdida total:

        loss = CrossEntropyLoss + lambda_temp * MSELoss

    CrossEntropyLoss:
        clasificación frío / templado / caliente

    MSELoss:
        regresión de temperatura normalizada
    """

    model.train()

    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    total_loss = 0.0
    total_class_loss = 0.0
    total_temp_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in loader:
        x = batch["x"].to(device)
        y_class = batch["y_class"].to(device)
        y_temp = batch["y_temp"].to(device)

        optimizer.zero_grad()

        out = model(x)

        loss_class = ce_loss(out["class_logits"], y_class)
        loss_temp = mse_loss(out["temp_pred"], y_temp)

        loss = loss_class + lambda_temp * loss_temp

        loss.backward()
        optimizer.step()

        batch_size = x.shape[0]

        total_loss += float(loss.item()) * batch_size
        total_class_loss += float(loss_class.item()) * batch_size
        total_temp_loss += float(loss_temp.item()) * batch_size

        pred_class = torch.argmax(out["class_logits"], dim=1)
        total_correct += int((pred_class == y_class).sum().item())
        total_samples += batch_size

    return {
        "loss": total_loss / total_samples,
        "class_loss": total_class_loss / total_samples,
        "temp_loss": total_temp_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: str | torch.device = "cpu",
    lambda_temp: float = 1.0,
) -> dict[str, float]:
    """
    Evalúa el modelo en validación o test.
    """

    model.eval()

    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    total_loss = 0.0
    total_class_loss = 0.0
    total_temp_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in loader:
        x = batch["x"].to(device)
        y_class = batch["y_class"].to(device)
        y_temp = batch["y_temp"].to(device)

        out = model(x)

        loss_class = ce_loss(out["class_logits"], y_class)
        loss_temp = mse_loss(out["temp_pred"], y_temp)

        loss = loss_class + lambda_temp * loss_temp

        batch_size = x.shape[0]

        total_loss += float(loss.item()) * batch_size
        total_class_loss += float(loss_class.item()) * batch_size
        total_temp_loss += float(loss_temp.item()) * batch_size

        pred_class = torch.argmax(out["class_logits"], dim=1)
        total_correct += int((pred_class == y_class).sum().item())
        total_samples += batch_size

    return {
        "loss": total_loss / total_samples,
        "class_loss": total_class_loss / total_samples,
        "temp_loss": total_temp_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


def fit_cnn_model(
    H_train: Sequence[np.ndarray],
    y_class_train: Sequence[int],
    y_temp_train: Sequence[float],
    H_val: Sequence[np.ndarray],
    y_class_val: Sequence[int],
    y_temp_val: Sequence[float],
    *,
    input_order: Literal["mk", "km"] = "mk",
    resize_to: tuple[int, int] | None = None,
    batch_size: int = 8,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    lambda_temp: float = 1.0,
    device: str | torch.device | None = None,
):
    """
    Entrena la CNN multitarea.

    Devuelve:
        model
        image_scaler
        temp_scaler
        history
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------------------------------------------------------
    # Normalizadores
    # -------------------------------------------------------------------------

    image_scaler = ComplexImageStandardizer().fit(
        H_train,
        input_order=input_order,
    )

    temp_scaler = TargetStandardizer().fit(
        np.asarray(y_temp_train, dtype=np.float32)
    )

    y_temp_train_norm = temp_scaler.transform(y_temp_train)
    y_temp_val_norm = temp_scaler.transform(y_temp_val)

    # -------------------------------------------------------------------------
    # Datasets
    # -------------------------------------------------------------------------

    train_ds = ThermalHDataset(
        H_train,
        y_class_train,
        y_temp_train_norm,
        input_order=input_order,
        image_standardizer=image_scaler,
        resize_to=resize_to,
    )

    val_ds = ThermalHDataset(
        H_val,
        y_class_val,
        y_temp_val_norm,
        input_order=input_order,
        image_standardizer=image_scaler,
        resize_to=resize_to,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    # -------------------------------------------------------------------------
    # Modelo
    # -------------------------------------------------------------------------

    model = ThermalHCNNMultiTaskNet(
        dropout2d=0.05,
        dropout_fc=0.20,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    history = {
        "train_loss": [],
        "train_class_loss": [],
        "train_temp_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_class_loss": [],
        "val_temp_loss": [],
        "val_accuracy": [],
    }

    best_val_loss = np.inf
    best_state = None

    # -------------------------------------------------------------------------
    # Bucle de entrenamiento
    # -------------------------------------------------------------------------

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            lambda_temp=lambda_temp,
        )

        val_metrics = evaluate(
            model,
            val_loader,
            device=device,
            lambda_temp=lambda_temp,
        )

        history["train_loss"].append(train_metrics["loss"])
        history["train_class_loss"].append(train_metrics["class_loss"])
        history["train_temp_loss"].append(train_metrics["temp_loss"])
        history["train_accuracy"].append(train_metrics["accuracy"])

        history["val_loss"].append(val_metrics["loss"])
        history["val_class_loss"].append(val_metrics["class_loss"])
        history["val_temp_loss"].append(val_metrics["temp_loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:04d} | "
                f"train_loss={train_metrics['loss']:.4f} | "
                f"train_acc={train_metrics['accuracy']:.3f} | "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"val_acc={val_metrics['accuracy']:.3f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, image_scaler, temp_scaler, history


# =============================================================================
# 8. Predicción
# =============================================================================

@torch.no_grad()
def predict_from_H(
    model: nn.Module,
    H: np.ndarray,
    image_scaler: ComplexImageStandardizer,
    temp_scaler: TargetStandardizer,
    *,
    input_order: Literal["mk", "km"] = "mk",
    resize_to: tuple[int, int] | None = None,
    device: str | torch.device = "cpu",
) -> dict[str, object]:
    """
    Predice clase térmica y temperatura para una única matriz H compleja.
    """

    model.eval()

    img = complex_H_to_2ch_image(
        H,
        input_order=input_order,
    )

    img = image_scaler.transform_image(img)

    x = torch.tensor(img, dtype=torch.float32)

    if resize_to is not None:
        x = x.unsqueeze(0)
        x = F.interpolate(
            x,
            size=resize_to,
            mode="bilinear",
            align_corners=False,
        )
    else:
        x = x.unsqueeze(0)

    x = x.to(device)

    out = model(x)

    probs = torch.softmax(out["class_logits"], dim=1).cpu().numpy()[0]
    class_idx = int(np.argmax(probs))
    class_name = IDX_TO_CLASS[class_idx]

    temp_norm = out["temp_pred"].cpu().numpy()
    temp_celsius = float(temp_scaler.inverse_transform(temp_norm)[0])

    return {
        "class_idx": class_idx,
        "class_name": class_name,
        "prob_frio": float(probs[0]),
        "prob_templado": float(probs[1]),
        "prob_caliente": float(probs[2]),
        "temperature_celsius": temp_celsius,
    }


# =============================================================================
# 9. Guardado y carga
# =============================================================================

def save_cnn_checkpoint(
    path: str,
    model: nn.Module,
    image_scaler: ComplexImageStandardizer,
    temp_scaler: TargetStandardizer,
    history: dict | None = None,
    *,
    resize_to: tuple[int, int] | None = None,
    input_order: Literal["mk", "km"] = "mk",
) -> None:
    """
    Guarda modelo y normalizadores.
    """

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "class_names": CLASS_NAMES,
        "image_scaler_mean": image_scaler.mean,
        "image_scaler_std": image_scaler.std,
        "temp_scaler_mean": temp_scaler.mean,
        "temp_scaler_std": temp_scaler.std,
        "resize_to": resize_to,
        "input_order": input_order,
        "history": history,
    }

    torch.save(checkpoint, path)


def load_cnn_checkpoint(
    path: str,
    *,
    device: str | torch.device = "cpu",
):
    """
    Carga modelo y normalizadores.
    """

    checkpoint = torch.load(path, map_location=device)

    model = ThermalHCNNMultiTaskNet()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    image_scaler = ComplexImageStandardizer(
        mean=checkpoint["image_scaler_mean"],
        std=checkpoint["image_scaler_std"],
    )

    temp_scaler = TargetStandardizer(
        mean=checkpoint["temp_scaler_mean"],
        std=checkpoint["temp_scaler_std"],
    )

    resize_to = checkpoint.get("resize_to", None)
    input_order = checkpoint.get("input_order", "mk")
    history = checkpoint.get("history", None)

    return model, image_scaler, temp_scaler, resize_to, input_order, history