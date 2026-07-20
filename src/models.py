"""
CyberGrid-AI | Step 4: Model Architectures

Implements the four models compared in the paper (Tables 3-5):
  - CNN       : convolutional baseline
  - LSTM      : recurrent baseline
  - AE-CNN    : ablation (AE denoiser + CNN, no LSTM)
  - AE-CLSTM  : paper's proposed model (AE + CNN + LSTM)

Architecture source: Table 2, Moradzadeh et al., Energy Reports 2022.

DOCUMENTED INTERPRETATION:
Table 2 lists CNN kernels as 2D (4x4, 3x3, 3x3) but never specifies
the input tensor shape. A flat feature vector cannot survive three
rounds of 2D max-pooling. We implement 1D convolutions over a temporal
lookback window instead — the standard approach in the CNN-LSTM load
forecasting literature this paper cites (Farsi et al. 2021, Rafi et al.
2021). All concrete numbers from Table 2 are preserved exactly:
filter counts (3, 16, 20), kernel lengths (4, 3, 3), pool size 2,
LSTM units 20 bidirectional, dropout 0.3, dense(12), tanh activation.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ── Shared CNN trunk (3 conv + 3 pool, per Table 2) ───────────────────
def _cnn_trunk(x):
    """1D CNN feature extractor shared by CNN, AE-CNN, AE-CLSTM."""
    x = layers.Conv1D(3,  kernel_size=4, strides=1,
                      padding="same", activation="relu",
                      name="conv1")(x)
    x = layers.MaxPooling1D(pool_size=2, strides=2,
                            padding="same", name="pool1")(x)
    x = layers.Conv1D(16, kernel_size=3, strides=1,
                      padding="same", activation="relu",
                      name="conv2")(x)
    x = layers.MaxPooling1D(pool_size=2, strides=2,
                            padding="same", name="pool2")(x)
    x = layers.Conv1D(20, kernel_size=3, strides=1,
                      padding="same", activation="relu",
                      name="conv3")(x)
    x = layers.MaxPooling1D(pool_size=2, strides=2,
                            padding="same", name="pool3")(x)
    return x


# ── AE denoising stage ────────────────────────────────────────────────
def _ae_stage(inputs, input_dim, window_size, n_features,
              latent_dim=32):
    """
    Denoising autoencoder front-end.
    Reconstructs clean input before the CNN sees it.
    """
    flat        = layers.Reshape((input_dim,), name="ae_flatten")(inputs)
    encoded     = layers.Dense(latent_dim, activation="relu",
                               name="encoder")(flat)
    decoded     = layers.Dense(input_dim,  activation="linear",
                               name="decoder")(encoded)
    denoised    = layers.Reshape((window_size, n_features),
                                 name="ae_reshape")(decoded)
    return denoised


# ── Model builders ────────────────────────────────────────────────────
def build_cnn(window_size: int, n_features: int) -> keras.Model:
    """Conventional CNN baseline: CNN trunk + FC head."""
    inputs = keras.Input(shape=(window_size, n_features), name="input")
    x = _cnn_trunk(inputs)
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(50, activation="relu", name="fc")(x)
    out = layers.Dense(1, activation="linear", name="output")(x)
    return keras.Model(inputs, out, name="CNN")


def build_lstm(window_size: int, n_features: int) -> keras.Model:
    """
    Conventional LSTM baseline (Table 2):
    Bidirectional LSTM(20) → Dropout(0.3) → Flatten →
    Dense(12) → output.
    """
    inputs = keras.Input(shape=(window_size, n_features), name="input")
    x = layers.Bidirectional(
        layers.LSTM(20, dropout=0.3, return_sequences=False,
                    activation="tanh"),
        name="bidirectional_1"
    )(inputs)
    x = layers.Dense(12, activation="relu",  name="dense_1")(x)
    out = layers.Dense(1, activation="linear", name="output")(x)
    return keras.Model(inputs, out, name="LSTM")


def build_ae_cnn(window_size: int, n_features: int,
                 latent_dim: int = 32) -> keras.Model:
    """
    Ablation model: AE denoiser → CNN trunk → FC head.
    AE-CLSTM minus the LSTM stage.
    """
    input_dim = window_size * n_features
    inputs  = keras.Input(shape=(window_size, n_features), name="input")
    x       = _ae_stage(inputs, input_dim, window_size,
                        n_features, latent_dim)
    x       = _cnn_trunk(x)
    x       = layers.Flatten(name="flatten")(x)
    x       = layers.Dense(50, activation="relu", name="fc")(x)
    out     = layers.Dense(1, activation="linear", name="output")(x)
    return keras.Model(inputs, out, name="AE_CNN")


def build_ae_clstm(window_size: int, n_features: int,
                   latent_dim: int = 32) -> keras.Model:
    """
    Proposed model (paper's headline architecture):
    AE → CNN trunk → LSTM replaces FC layers → Dense(12) → output.

    The key insight: LSTM replaces the fully-connected layers in the
    CNN to add time-series state modelling and prevent overfitting.
    """
    input_dim = window_size * n_features
    inputs  = keras.Input(shape=(window_size, n_features), name="input")

    # Stage 1: AE denoises the raw input window
    x = _ae_stage(inputs, input_dim, window_size, n_features, latent_dim)

    # Stage 2: CNN extracts temporal features
    x = _cnn_trunk(x)

    # Stage 3: LSTM replaces FC layers (Table 2 spec)
    x = layers.Bidirectional(
        layers.LSTM(20, dropout=0.3, return_sequences=False,
                    activation="tanh"),
        name="bidirectional_1"
    )(x)
    x   = layers.Dense(12, activation="relu",   name="dense_1")(x)
    out = layers.Dense(1,  activation="linear", name="output")(x)
    return keras.Model(inputs, out, name="AE_CLSTM")


# ── Sanity check ──────────────────────────────────────────────────────
if __name__ == "__main__":
    WINDOW   = 24
    N_FEATS  = 5
    BATCH    = 16

    dummy_x = np.random.randn(BATCH, WINDOW, N_FEATS).astype("float32")
    dummy_y = np.random.randn(BATCH, 1).astype("float32")

    builders = {
        "CNN"     : build_cnn,
        "LSTM"    : build_lstm,
        "AE_CNN"  : build_ae_cnn,
        "AE_CLSTM": build_ae_clstm,
    }

    print("=" * 55)
    print("  CyberGrid-AI  |  Step 4: Model Sanity Check")
    print("=" * 55)

    for name, builder in builders.items():
        model   = builder(WINDOW, N_FEATS)
        model.compile(optimizer="adam", loss="mse")
        out     = model(dummy_x, training=False)
        history = model.fit(dummy_x, dummy_y, epochs=2,
                            verbose=0, batch_size=BATCH)
        params  = model.count_params()
        loss    = history.history["loss"][-1]
        print(f"  {name:<10} | output: {tuple(out.shape)} "
              f"| params: {params:>7,} "
              f"| 2-epoch loss: {loss:.4f}  [OK]")

    print("\n[✓] All four architectures verified")
    print("[✓] Done. Next: run src/train.py")