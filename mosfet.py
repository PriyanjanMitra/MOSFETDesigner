import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
from typing import Optional, Tuple, Dict, Any

import warnings

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')


# noinspection PyTypeChecker
class NanoMOSFETDesigner:
    def __init__(self, material: str = 'silicon'):
        """
        Initialize the MOSFET designer with specified channel material.

        Args:
            material: Channel material ('silicon', 'sige', 'gaas', 'gan', 'sic', 'inp', 'diamond')
        """
        # Fundamental constants
        self.mobility_p = None
        self.eps_si = None
        self.ni = None
        self.material_name = None
        self.bandgap = None
        self.mobility_n = None
        self.electron_affinity = None
        self.phi_ms = None
        self.q: tf.Tensor = tf.constant(1.6e-19, dtype=tf.float32)
        self.eps_0: tf.Tensor = tf.constant(8.85e-12, dtype=tf.float32)
        self.eps_ox: float = 3.9
        self.k: tf.Tensor = tf.constant(1.38e-23, dtype=tf.float32)
        self.temperature: tf.Tensor = tf.constant(300.0, dtype=tf.float32)

        # Set material properties
        self.material = material
        self.set_material_properties(material)

        # Nanoscale constraints with material-specific ranges
        if material in ['gan', 'sic', 'diamond']:
            self.min_Lg: float = 0.5
            self.max_Lg: float = 5.0
            self.max_Lch: float = 20.0
            self.min_Vd: float = 0.5
            self.max_Vd: float = 5.0
        elif material in ['gaas', 'inp']:
            self.min_Lg: float = 0.2
            self.max_Lg: float = 2.0
            self.max_Lch: float = 10.0
            self.min_Vd: float = 0.2
            self.max_Vd: float = 1.5
        else:
            self.min_Lg: float = 0.1
            self.max_Lg: float = 1.0
            self.max_Lch: float = 5.0
            self.min_Vd: float = 0.1
            self.max_Vd: float = 1.2

        self.feature_scaler: StandardScaler = StandardScaler()
        self.design_scaler: StandardScaler = StandardScaler()

        self.model: Optional[keras.Model] = None
        self.training_history: Optional[Any] = None

        self.build_model()

    def set_material_properties(self, material: str) -> None:
        """Set semiconductor material properties."""

        materials = {
            'silicon': {
                'eps_r': 11.7,
                'ni': 1.5e10,
                'mobility_n': 0.14,
                'mobility_p': 0.045,
                'bandgap': 1.12,
                'affinity': 4.05,
                'name': 'Silicon (Si)'
            },
            'sige': {
                'eps_r': 12.0,
                'ni': 2.0e10,
                'mobility_n': 0.10,
                'mobility_p': 0.08,
                'bandgap': 1.0,
                'affinity': 4.0,
                'name': 'Silicon Germanium (SiGe)'
            },
            'gaas': {
                'eps_r': 12.9,
                'ni': 2.1e6,
                'mobility_n': 0.85,
                'mobility_p': 0.04,
                'bandgap': 1.42,
                'affinity': 4.07,
                'name': 'Gallium Arsenide (GaAs)'
            },
            'gan': {
                'eps_r': 9.0,
                'ni': 1.9e-10,
                'mobility_n': 0.15,
                'mobility_p': 0.03,
                'bandgap': 3.4,
                'affinity': 4.1,
                'name': 'Gallium Nitride (GaN)'
            },
            'sic': {
                'eps_r': 9.7,
                'ni': 8.2e-9,
                'mobility_n': 0.09,
                'mobility_p': 0.01,
                'bandgap': 3.26,
                'affinity': 3.8,
                'name': 'Silicon Carbide (SiC)'
            },
            'inp': {
                'eps_r': 12.5,
                'ni': 1.3e8,
                'mobility_n': 0.46,
                'mobility_p': 0.015,
                'bandgap': 1.34,
                'affinity': 4.38,
                'name': 'Indium Phosphide (InP)'
            },
            'diamond': {
                'eps_r': 5.7,
                'ni': 1.0e-27,
                'mobility_n': 0.45,
                'mobility_p': 0.38,
                'bandgap': 5.47,
                'affinity': 0.5,
                'name': 'Diamond (C)'
            }
        }

        if material not in materials:
            raise ValueError(f"Material {material} not supported. Choose from: {list(materials.keys())}")

        mat = materials[material]

        self.eps_si = mat['eps_r']
        self.ni = tf.constant(mat['ni'], dtype=tf.float32)
        self.mobility_n = mat['mobility_n']
        self.mobility_p = mat['mobility_p']
        self.bandgap = mat['bandgap']
        self.electron_affinity = mat['affinity']
        self.material_name = mat['name']

        φ_m = 5.1
        if material == 'silicon':
            self.phi_ms = φ_m - (self.electron_affinity + self.bandgap / 2)
        elif material == 'sige':
            self.phi_ms = φ_m - (self.electron_affinity + self.bandgap / 2) - 0.1
        elif material == 'gaas':
            self.phi_ms = φ_m - (self.electron_affinity + self.bandgap / 2) - 0.2
        elif material == 'gan':
            self.phi_ms = φ_m - (self.electron_affinity + self.bandgap / 2) - 0.5
        elif material == 'sic':
            self.phi_ms = φ_m - (self.electron_affinity + self.bandgap / 2) - 0.8
        elif material == 'inp':
            self.phi_ms = φ_m - (self.electron_affinity + self.bandgap / 2) - 0.3
        elif material == 'diamond':
            self.phi_ms = φ_m - (self.electron_affinity + self.bandgap / 2) - 1.0

    def build_model(self) -> None:
        """Build a more robust neural network"""
        inputs = keras.Input(shape=(4,))
        x = layers.Dense(512, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        outputs = layers.Dense(4, activation='linear')(x)

        self.model = keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0005),
            loss='mse',
            metrics=['mae']
        )

    def apply_nanoscale_constraints(self, design_params: np.ndarray) -> np.ndarray:
        """Apply material-specific constraints"""
        Lg = np.clip(design_params[:, 0], self.min_Lg, self.max_Lg)
        doping = np.clip(design_params[:, 1], 1e15, 1e20)
        Lch = np.clip(design_params[:, 2], Lg + 0.1, self.max_Lch)
        Vd = np.clip(design_params[:, 3], self.min_Vd, self.max_Vd)
        return np.column_stack([Lg, doping, Lch, Vd])

    def calculate_vd(self, design_params: np.ndarray, input_features: np.ndarray) -> float:
        """Calculate drain voltage based on device physics"""
        design_params_tensor = tf.convert_to_tensor(design_params, dtype=tf.float32)
        input_features_tensor = tf.convert_to_tensor(input_features, dtype=tf.float32)

        Lg, Na, Lch, _ = tf.unstack(design_params_tensor, axis=1)
        Id_target, SS, Vtgm, tox = tf.unstack(input_features_tensor, axis=1)

        Lg_m = Lg * 1e-9
        tox_m = tox * 1e-9
        Na_m = Na * 1e6

        phi_f = (self.k * self.temperature / self.q) * tf.math.log(Na_m / (self.ni + 1e-20))
        Cox = self.eps_0 * tf.constant(self.eps_ox, dtype=tf.float32) / tox_m
        Qb = tf.math.sqrt(2.0 * self.q * tf.constant(self.eps_si, dtype=tf.float32) *
                          self.eps_0 * Na_m * phi_f)
        Vth = tf.constant(self.phi_ms, dtype=tf.float32) + 2.0 * phi_f + Qb / Cox

        if self.material in ['gan', 'sic', 'diamond']:
            delta_Vth = 0.02 / (Lg + 0.1)
        else:
            delta_Vth = 0.05 / (Lg + 0.1)

        Vth += delta_Vth

        W = 100e-9

        R_on = Lg_m / (self.mobility_n * Cox * W * (Vtgm - Vth + 0.1))

        Vd_calc = Vtgm + Id_target * R_on * 1e6

        Vd_calc = tf.clip_by_value(Vd_calc, self.min_Vd, self.max_Vd)

        return float(Vd_calc.numpy()[0])

    def generate_synthetic_training_data(self, n_samples: int = 1000000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate large-scale synthetic training data optimized for 1M+ samples.
        Uses vectorized operations for memory and performance efficiency.
        """
        np.random.seed(42)

        # Generate input features (performance parameters) - all at once using vectorization
        Id = 10 ** np.random.uniform(-9, -3, n_samples)
        SS = np.random.uniform(60, 120, n_samples)
        Vtgm = np.random.uniform(0.1, 1.0, n_samples)
        tox = np.random.uniform(0.3, 3.0, n_samples)

        X = np.column_stack([Id, SS, Vtgm, tox]).astype(np.float32)

        # Generate corresponding design parameters using vectorized operations
        # Gate length (inversely related to Id and SS)
        Lg = np.clip(
            0.5 / (Id * 1e6 + 0.1) + np.random.normal(0, 0.1, n_samples),
            self.min_Lg, self.max_Lg
        )

        # Doping concentration (related to Vtgm)
        doping = 10 ** np.random.uniform(16, 19, n_samples)
        doping = doping * (Vtgm / 0.5)
        doping = np.clip(doping, 1e15, 1e20)

        # Channel length (Lch > Lg)
        Lch = Lg + np.random.uniform(0.2, 2.0, n_samples)
        Lch = np.clip(Lch, Lg + 0.1, self.max_Lch)

        # Drain voltage (related to Id and Vtgm)
        Vd = np.clip(
            Vtgm + Id * 1e6 * np.random.uniform(0.1, 1.0, n_samples),
            self.min_Vd, self.max_Vd
        )

        y = np.column_stack([Lg, doping, Lch, Vd]).astype(np.float32)

        return X, y

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              epochs: int = 300, batch_size: int = 64, verbose: int = 0) -> Any:

        X_train_norm = self.feature_scaler.fit_transform(X_train)
        y_train_norm = self.design_scaler.fit_transform(y_train)

        validation_data = None
        if X_val is not None and y_val is not None:
            X_val_norm = self.feature_scaler.transform(X_val)
            y_val_norm = self.design_scaler.transform(y_val)
            validation_data = (X_val_norm, y_val_norm)

        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss' if validation_data else 'loss',
            factor=0.5,
            patience=20,
            min_lr=1e-6
        )

        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss' if validation_data else 'loss',
            patience=50,
            restore_best_weights=True
        )

        history = self.model.fit(
            X_train_norm, y_train_norm,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            validation_split=0.2 if validation_data is None else 0,
            callbacks=[reduce_lr, early_stop],
            verbose=verbose
        )

        self.training_history = history
        return history

    def predict_design(self, Id: float, SS: float, Vtgm: float, tox: float) -> Dict[str, str]:
        """Predict MOSFET design parameters"""
        perf_input = np.array([[Id, SS, Vtgm, tox]], dtype=np.float32)
        perf_input_norm = self.feature_scaler.transform(perf_input)

        design_norm = self.model.predict(perf_input_norm, verbose=0)
        design_params = self.design_scaler.inverse_transform(design_norm)

        constrained_params = self.apply_nanoscale_constraints(design_params)
        Lg, doping, Lch, _ = constrained_params[0]

        Vd = self.calculate_vd(constrained_params, perf_input)

        return {
            'Material': self.material_name,
            'Gate length (Lg)': f"{Lg:.3f} nm",
            'Doping concentration': f"{doping:.2e} cm⁻³",
            'Channel length (Lch)': f"{Lch:.3f} nm",
            'Drain voltage (Vd)': f"{Vd:.3f} V",
            'Oxide thickness': f"{tox:.2f} nm"
        }


def load_and_preprocess_data(csv_file: str, material: str = 'silicon') -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and preprocess data with material-specific constraints."""
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        raise Exception(f"Error reading CSV file: {e}")

    required_columns = ['Id', 'SS', 'Vtgm', 'tox', 'Lg', 'doping', 'Lch', 'Vd']
    missing_cols = [col for col in required_columns if col not in df.columns]

    if missing_cols:
        designer = NanoMOSFETDesigner(material=material)
        X, y = designer.generate_synthetic_training_data(1000000)
        return train_test_split(X, y, test_size=0.2, random_state=42)

    X = df[['Id', 'SS', 'Vtgm', 'tox']].values.astype(np.float32)
    y = df[['Lg', 'doping', 'Lch', 'Vd']].values.astype(np.float32)

    return train_test_split(X, y, test_size=0.2, random_state=42)


def plot_training_history(history: Any, save_path: Optional[str] = None) -> None:
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(history.history['loss'], label='Training Loss', linewidth=2, color='#3498db')
    if 'val_loss' in history.history:
        ax1.plot(history.history['val_loss'], label='Validation Loss', linewidth=2, color='#e74c3c')
    ax1.set_title('Model Training Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    if 'mae' in history.history:
        ax2.plot(history.history['mae'], label='Training MAE', linewidth=2, color='#2ecc71')
        if 'val_mae' in history.history:
            ax2.plot(history.history['val_mae'], label='Validation MAE', linewidth=2, color='#e67e22')
        ax2.set_title('Model Mean Absolute Error', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()