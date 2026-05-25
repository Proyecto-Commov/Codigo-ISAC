from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# 1. Definición de clases y orden fijo de características
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


# OJO:
# Este orden asume que has extraído las características con:
# percentiles=(25, 50, 75)
#
# 5 características de módulo:
#   mag_mean, mag_var, mag_rms, mag_energy_total, mag_max
# 3 percentiles:
#   mag_percentile_25, mag_percentile_50, mag_percentile_75
# 1 rango dinámico:
#   mag_dynamic_range_linear
# 2 PDP:
#   pdp_los_peak_power, pdp_total_energy
# 3 SVD:
#   svd_sigma_1, svd_sigma_2, svd_spectral_flatness
# 1 Doppler:
#   doppler_variance_energy
#
# Total: 15 características

FEATURE_NAMES = (
    "mag_mean",
    "mag_var",
    "mag_rms",
    "mag_energy_total",
    "mag_max",
    "mag_percentile_25",
    "mag_percentile_50",
    "mag_percentile_75",
    "mag_dynamic_range_linear",
    "pdp_los_peak_power",
    "pdp_total_energy",
    "svd_sigma_1",
    "svd_sigma_2",
    "svd_spectral_flatness",
    "doppler_variance_energy",
)


def features_dict_to_vector(
    features: dict[str, float],
    *,
    feature_names: Sequence[str] = FEATURE_NAMES,
) -> np.ndarray:
    """
    Convierte el diccionario devuelto por extract_channel_matrix_features(...)
    en un vector NumPy con orden fijo.

    Esto es importante: la red neuronal no entiende nombres de columnas,
    solo posiciones.
    """

    missing = [name for name in feature_names if name not in features]

    if missing:
        raise KeyError(
            "Faltan características en el diccionario:\n"
            + "\n".join(f"  - {name}" for name in missing)
        )

    x = np.array([features[name] for name in feature_names], dtype=np.float32)

    x = np.nan_to_num(
        x,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return x


# =============================================================================
# 2. Normalizadores
# =============================================================================

@dataclass
class Standardizer:
    """
    Normalizador z-score:

        x_norm = (x - mean) / std

    Se debe ajustar SOLO con train.
    """

    mean: np.ndarray | None = None
    std: np.ndarray | None = None
    eps: float = 1e-8

    def fit(self, X: np.ndarray) -> "Standardizer":
        X = np.asarray(X, dtype=np.float32)

        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)

        self.std = np.where(self.std < self.eps, 1.0, self.std)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("El Standardizer no está ajustado. Llama antes a fit(...).")

        X = np.asarray(X, dtype=np.float32)
        return (X - self.mean) / self.std

    def inverse_transform(self, X_norm: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("El Standardizer no está ajustado. Llama antes a fit(...).")

        X_norm = np.asarray(X_norm, dtype=np.float32)
        return X_norm * self.std + self.mean


@dataclass
class TargetStandardizer:
    """
    Normalizador para la temperatura.

    Conviene entrenar la regresión con temperatura normalizada, porque si la
    temperatura está en grados Celsius y las clases se entrenan con cross-entropy,
    las escalas de las pérdidas pueden quedar muy descompensadas.
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
            raise RuntimeError("El TargetStandardizer no está ajustado.")

        y = np.asarray(y, dtype=np.float32).reshape(-1)
        return (y - self.mean) / self.std

    def inverse_transform(self, y_norm: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("El TargetStandardizer no está ajustado.")

        y_norm = np.asarray(y_norm, dtype=np.float32).reshape(-1)
        return y_norm * self.std + self.mean


# =============================================================================
# 3. Dataset
# =============================================================================

class ThermalFeatureDataset(Dataset):
    """
    Dataset para muestras ya convertidas a características.

    X:
        Matriz de entrada con shape (num_muestras, 15).

    y_class:
        Etiquetas de clase:
            0 -> frío
            1 -> templado
            2 -> caliente

    y_temp:
        Temperatura real en ºC.
    """

    def __init__(
        self,
        X: np.ndarray,
        y_class: np.ndarray,
        y_temp: np.ndarray,
    ):
        X = np.asarray(X, dtype=np.float32)
        y_class = np.asarray(y_class, dtype=np.int64).reshape(-1)
        y_temp = np.asarray(y_temp, dtype=np.float32).reshape(-1)

        if X.ndim != 2:
            raise ValueError(f"X debe ser 2D, recibido shape={X.shape}")

        if X.shape[1] != 15:
            raise ValueError(f"X debe tener 15 características, recibido {X.shape[1]}")

        if len(X) != len(y_class) or len(X) != len(y_temp):
            raise ValueError(
                "X, y_class e y_temp deben tener el mismo número de muestras."
            )

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_class = torch.tensor(y_class, dtype=torch.long)
        self.y_temp = torch.tensor(y_temp, dtype=torch.float32)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return {
            "x": self.X[idx],
            "y_class": self.y_class[idx],
            "y_temp": self.y_temp[idx],
        }


# =============================================================================
# 4. Red neuronal multitarea
# =============================================================================

class ThermalMultiTaskNet(nn.Module):
    """
    Red neuronal multitarea:

        Entrada:
            vector de 15 características

        Salidas:
            - logits de clasificación: frío / templado / caliente
            - temperatura normalizada: valor real

    Importante:
        La salida de clasificación NO lleva softmax dentro del modelo, porque
        nn.CrossEntropyLoss espera logits directamente.
    """

    def __init__(
        self,
        input_dim: int = 15,
        hidden_dim_1: int = 64,
        hidden_dim_2: int = 32,
        dropout: float = 0.10,
    ):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.LayerNorm(hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.LayerNorm(hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.class_head = nn.Linear(hidden_dim_2, 3)
        self.temp_head = nn.Linear(hidden_dim_2, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.shared(x)

        class_logits = self.class_head(z)
        temp_pred = self.temp_head(z).squeeze(-1)

        return {
            "class_logits": class_logits,
            "temp_pred": temp_pred,
        }


# =============================================================================
# 5. Entrenamiento
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

        loss = loss_clasificacion + lambda_temp * loss_temperatura

    Donde:
        - loss_clasificacion = CrossEntropyLoss
        - loss_temperatura = MSELoss
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


def fit_model(
    X_train: np.ndarray,
    y_class_train: np.ndarray,
    y_temp_train: np.ndarray,
    X_val: np.ndarray,
    y_class_val: np.ndarray,
    y_temp_val: np.ndarray,
    *,
    batch_size: int = 16,
    epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    lambda_temp: float = 1.0,
    device: str | torch.device | None = None,
):
    """
    Entrena el modelo completo.

    Devuelve:
        model
        x_scaler
        temp_scaler
        history
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------------------------------------------------------
    # Normalización ajustada solo con train
    # -------------------------------------------------------------------------

    x_scaler = Standardizer().fit(X_train)
    temp_scaler = TargetStandardizer().fit(y_temp_train)

    X_train_norm = x_scaler.transform(X_train)
    X_val_norm = x_scaler.transform(X_val)

    y_temp_train_norm = temp_scaler.transform(y_temp_train)
    y_temp_val_norm = temp_scaler.transform(y_temp_val)

    # -------------------------------------------------------------------------
    # Datasets y loaders
    # -------------------------------------------------------------------------

    train_ds = ThermalFeatureDataset(
        X_train_norm,
        y_class_train,
        y_temp_train_norm,
    )

    val_ds = ThermalFeatureDataset(
        X_val_norm,
        y_class_val,
        y_temp_val_norm,
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

    model = ThermalMultiTaskNet(
        input_dim=15,
        hidden_dim_1=64,
        hidden_dim_2=32,
        dropout=0.10,
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

    return model, x_scaler, temp_scaler, history


# =============================================================================
# 6. Predicción
# =============================================================================

@torch.no_grad()
def predict_from_feature_matrix(
    model: nn.Module,
    X: np.ndarray,
    x_scaler: Standardizer,
    temp_scaler: TargetStandardizer,
    *,
    device: str | torch.device = "cpu",
) -> dict[str, np.ndarray]:
    """
    Predice clase y temperatura para una matriz X de shape (num_muestras, 15).
    """

    model.eval()

    X_norm = x_scaler.transform(X)

    x_tensor = torch.tensor(X_norm, dtype=torch.float32).to(device)

    out = model(x_tensor)

    probs = torch.softmax(out["class_logits"], dim=1).cpu().numpy()
    class_idx = np.argmax(probs, axis=1)

    temp_norm = out["temp_pred"].cpu().numpy()
    temp_celsius = temp_scaler.inverse_transform(temp_norm)

    class_names = np.array([IDX_TO_CLASS[int(idx)] for idx in class_idx])

    return {
        "class_idx": class_idx,
        "class_name": class_names,
        "class_probabilities": probs,
        "temperature_celsius": temp_celsius,
    }


@torch.no_grad()
def predict_from_features_dict(
    model: nn.Module,
    features: dict[str, float],
    x_scaler: Standardizer,
    temp_scaler: TargetStandardizer,
    *,
    device: str | torch.device = "cpu",
) -> dict[str, object]:
    """
    Predice para una única captura representada como diccionario de características.
    """

    x = features_dict_to_vector(features).reshape(1, -1)

    pred = predict_from_feature_matrix(
        model,
        x,
        x_scaler,
        temp_scaler,
        device=device,
    )

    probs = pred["class_probabilities"][0]

    return {
        "class_idx": int(pred["class_idx"][0]),
        "class_name": str(pred["class_name"][0]),
        "prob_frio": float(probs[0]),
        "prob_templado": float(probs[1]),
        "prob_caliente": float(probs[2]),
        "temperature_celsius": float(pred["temperature_celsius"][0]),
    }


# =============================================================================
# 7. Guardado y carga
# =============================================================================

def save_checkpoint(
    path: str,
    model: nn.Module,
    x_scaler: Standardizer,
    temp_scaler: TargetStandardizer,
    history: dict | None = None,
) -> None:
    """
    Guarda modelo y normalizadores.
    """

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "feature_names": FEATURE_NAMES,
        "class_names": CLASS_NAMES,
        "x_scaler_mean": x_scaler.mean,
        "x_scaler_std": x_scaler.std,
        "temp_scaler_mean": temp_scaler.mean,
        "temp_scaler_std": temp_scaler.std,
        "history": history,
    }

    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    *,
    device: str | torch.device = "cpu",
):
    """
    Carga modelo y normalizadores.
    """

    checkpoint = torch.load(path, map_location=device)

    model = ThermalMultiTaskNet(input_dim=15)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    x_scaler = Standardizer(
        mean=checkpoint["x_scaler_mean"],
        std=checkpoint["x_scaler_std"],
    )

    temp_scaler = TargetStandardizer(
        mean=checkpoint["temp_scaler_mean"],
        std=checkpoint["temp_scaler_std"],
    )

    history = checkpoint.get("history", None)

    return model, x_scaler, temp_scaler, history