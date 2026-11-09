"""Parte 1: regresión logística multinomial implementada por el equipo (numpy).

Predice la categoría ICA de PM2.5 del día siguiente (buena / aceptable / dañina)
a partir del PM2.5 reciente, la meteorología y la estacionalidad.

Líneas base: persistencia (mañana = hoy) y clase mayoritaria.
Validación temporal: se entrena con los primeros meses y se prueba con los últimos.

Uso:
    python -m src.model     # entrena, evalúa y guarda data/processed/modelo.npz
"""
import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED, STATES, state_of
from src.data import load_daily

MODEL_PATH = DATA_PROCESSED / "modelo.npz"
TEST_FROM = "2024-04-01"  # últimos ~4 meses como prueba

FEATURES = [
    "pm25_hoy",
    "pm25_ayer",
    "pm25_media3",
    "pm25_tendencia",
    "lluvia_mm",
    "viento_max_kmh",
    "humedad_pct",
    "fin_semana",
    "mes_sin",
    "mes_cos",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Una fila por estación-día con features de hoy y la clase de mañana."""
    out = []
    for _, g in df.groupby("codigo"):
        g = g.set_index("fecha").asfreq("D")  # huecos quedan como NaN
        g["codigo"] = g["codigo"].ffill()
        f = pd.DataFrame(index=g.index)
        f["codigo"] = g["codigo"]
        f["pm25_hoy"] = g["pm25"]
        f["pm25_ayer"] = g["pm25"].shift(1)
        f["pm25_media3"] = g["pm25"].rolling(3).mean()
        f["pm25_tendencia"] = f["pm25_hoy"] - f["pm25_ayer"]
        for col in ("lluvia_mm", "viento_max_kmh", "humedad_pct"):
            f[col] = g[col]
        f["fin_semana"] = (g.index.dayofweek >= 5).astype(float)
        mes = g.index.month
        f["mes_sin"] = np.sin(2 * np.pi * mes / 12)
        f["mes_cos"] = np.cos(2 * np.pi * mes / 12)
        f["pm25_manana"] = g["pm25"].shift(-1)
        out.append(f.reset_index())
    feats = pd.concat(out, ignore_index=True)
    return feats


class SoftmaxRegression:
    """Regresión logística multinomial con descenso de gradiente y L2."""

    def __init__(self, n_classes=3, lr=0.3, l2=1e-3, epochs=3000, class_weight_power=0.25):
        self.k = n_classes
        self.lr = lr
        self.l2 = l2
        self.epochs = epochs
        # 0 = sin balanceo, 1 = balanceo total; 0.25 dio el mejor F1-macro
        # (con balanceo total el modelo sobre-predice "buena")
        self.class_weight_power = class_weight_power

    @staticmethod
    def _softmax(z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def fit(self, X, y):
        self.mu = X.mean(axis=0)
        self.sigma = X.std(axis=0) + 1e-9
        Xs = (X - self.mu) / self.sigma
        n, d = Xs.shape
        Y = np.eye(self.k)[y]
        # Pesos por clase inversos a la frecuencia (la clase "dañina" es rara)
        counts = np.bincount(y, minlength=self.k)
        w = (n / (self.k * np.maximum(counts, 1))) ** self.class_weight_power
        sw = w[y][:, None]
        self.W = np.zeros((d, self.k))
        self.b = np.zeros(self.k)
        self.loss_history = []
        for _ in range(self.epochs):
            P = self._softmax(Xs @ self.W + self.b)
            G = sw * (P - Y) / n
            self.W -= self.lr * (Xs.T @ G + self.l2 * self.W)
            self.b -= self.lr * G.sum(axis=0)
            loss = -(sw * Y * np.log(P + 1e-12)).sum() / n
            self.loss_history.append(loss)
        return self

    def predict_proba(self, X):
        return self._softmax(((X - self.mu) / self.sigma) @ self.W + self.b)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def save(self, path=MODEL_PATH):
        np.savez(path, W=self.W, b=self.b, mu=self.mu, sigma=self.sigma)

    @classmethod
    def load(cls, path=MODEL_PATH):
        z = np.load(path)
        m = cls()
        m.W, m.b, m.mu, m.sigma = z["W"], z["b"], z["mu"], z["sigma"]
        return m


def confusion(y_true, y_pred, k=3):
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def metrics(y_true, y_pred, k=3):
    cm = confusion(y_true, y_pred, k)
    acc = np.trace(cm) / cm.sum()
    f1s = []
    for c in range(k):
        tp = cm[c, c]
        prec = tp / cm[:, c].sum() if cm[:, c].sum() else 0.0
        rec = tp / cm[c, :].sum() if cm[c, :].sum() else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return {"accuracy": acc, "f1_macro": float(np.mean(f1s)), "f1_por_clase": f1s, "cm": cm}


def dataset():
    feats = build_features(load_daily()).dropna(subset=FEATURES + ["pm25_manana"])
    feats["y"] = feats["pm25_manana"].map(state_of).astype(int)
    feats["y_hoy"] = feats["pm25_hoy"].map(state_of).astype(int)
    train = feats[feats["fecha"] < TEST_FROM]
    test = feats[feats["fecha"] >= TEST_FROM]
    return train, test


def evaluate():
    train, test = dataset()
    model = SoftmaxRegression().fit(train[FEATURES].to_numpy(), train["y"].to_numpy())
    y = test["y"].to_numpy()
    results = {
        "Regresión logística (propia)": metrics(y, model.predict(test[FEATURES].to_numpy())),
        "Persistencia (mañana = hoy)": metrics(y, test["y_hoy"].to_numpy()),
        "Clase mayoritaria": metrics(
            y, np.full_like(y, np.bincount(train["y"]).argmax())
        ),
    }
    return model, results, train, test


def main():
    model, results, train, test = evaluate()
    print(f"Entrenamiento: {len(train)} filas | Prueba: {len(test)} filas (desde {TEST_FROM})")
    print("Distribución prueba:", dict(zip(STATES, np.bincount(test["y"], minlength=3))))
    for name, r in results.items():
        f1c = ", ".join(f"{s}={v:.2f}" for s, v in zip(STATES, r["f1_por_clase"]))
        print(f"\n{name}\n  accuracy={r['accuracy']:.3f}  F1-macro={r['f1_macro']:.3f}  ({f1c})")
        print("  matriz de confusión (filas = real):\n", r["cm"])
    model.save()
    print("\nModelo guardado en", MODEL_PATH)


if __name__ == "__main__":
    main()
