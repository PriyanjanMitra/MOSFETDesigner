import warnings
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')


def calculate_id(Vgs, Idss, Vp):
    return Idss * (1 - Vgs / Vp) ** 2


class JFETDesigner:
    def __init__(self, material: str = 'silicon'):
        self.eps_s = None
        self.ni = None
        self.mobility_n = None
        self.mobility_n = None
        self.q = tf.constant(1.6e-19, dtype=tf.float32)
        self.eps_0 = tf.constant(8.85e-12, dtype=tf.float32)
        self.k = tf.constant(1.38e-23, dtype=tf.float32)
        self.temperature = tf.constant(300.0, dtype=tf.float32)

        self.material = material
        self.set_material_properties(material)

        self.min_L = 0.2
        self.max_L = 5.0
        self.min_Vd = 0.1
        self.max_Vd = 5.0

        self.feature_scaler = StandardScaler()
        self.design_scaler = StandardScaler()

        self.model: Optional[keras.Model] = None
        self.training_history = None

        self.build_model()

    def set_material_properties(self, material: str):
        materials = {
            'silicon': {'eps_r': 11.7, 'mobility_n': 0.14, 'ni': 1.5e10},
            'gaas': {'eps_r': 12.9, 'mobility_n': 0.85, 'ni': 2.1e6},
            'gan': {'eps_r': 9.0, 'mobility_n': 0.15, 'ni': 1.9e-10},
        }

        if material not in materials:
            raise ValueError("Unsupported material")

        mat = materials[material]
        self.eps_s = mat['eps_r']
        self.mobility_n = mat['mobility_n']
        self.ni = tf.constant(mat['ni'], dtype=tf.float32)

    def build_model(self):
        inputs = keras.Input(shape=(4,))
        x = layers.Dense(256, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(64, activation='relu')(x)
        outputs = layers.Dense(4, activation='linear')(x)

        self.model = keras.Model(inputs, outputs)
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

    def calculate_pinch_off_voltage(self, Nd, a):
        eps = self.eps_s * self.eps_0
        Vp = - (self.q * Nd * (a ** 2)) / (2 * eps)
        return Vp

    def calculate_rds(self, L, Nd, A):
        return L / (self.q * self.mobility_n * Nd * A)

    def calculate_vd(self, design_params, input_features):
        L, Nd, a, _ = design_params[0]
        Id_target, Vgs, Idss, A = input_features[0]

        Nd_m = Nd * 1e6
        a_m = a * 1e-9
        L_m = L * 1e-9

        Vp = self.calculate_pinch_off_voltage(Nd_m, a_m)
        Id = calculate_id(Vgs, Idss, Vp)

        Rds = self.calculate_rds(L_m, Nd_m, A)

        Vd = Id * Rds
        Vd = np.clip(Vd, self.min_Vd, self.max_Vd)

        return float(Vd.numpy())

    def generate_synthetic_training_data(self, n_samples=200000):
        np.random.seed(42)

        Id = 10 ** np.random.uniform(-9, -3, n_samples)
        Vgs = np.random.uniform(-2, 0, n_samples)
        Idss = 10 ** np.random.uniform(-6, -2, n_samples)
        A = np.random.uniform(1e-14, 1e-12, n_samples)

        X = np.column_stack([Id, Vgs, Idss, A]).astype(np.float32)

        L = np.random.uniform(self.min_L, self.max_L, n_samples)
        Nd = 10 ** np.random.uniform(15, 19, n_samples)
        a = np.random.uniform(10, 100, n_samples)

        Vd = np.clip(Id * 1e6, self.min_Vd, self.max_Vd)

        y = np.column_stack([L, Nd, a, Vd]).astype(np.float32)

        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train(self, X_train, y_train, X_val=None, y_val=None,
              epochs=200, batch_size=64, verbose=1):

        X_train_n = self.feature_scaler.fit_transform(X_train)
        y_train_n = self.design_scaler.fit_transform(y_train)

        validation_data = None
        if X_val is not None:
            X_val_n = self.feature_scaler.transform(X_val)
            y_val_n = self.design_scaler.transform(y_val)
            validation_data = (X_val_n, y_val_n)

        history = self.model.fit(
            X_train_n, y_train_n,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose
        )

        self.training_history = history
        return history

    def predict_design(self, Id, Vgs, Idss, A):
        X = np.array([[Id, Vgs, Idss, A]], dtype=np.float32)
        X_n = self.feature_scaler.transform(X)

        y_pred = self.model.predict(X_n, verbose=0)
        params = self.design_scaler.inverse_transform(y_pred)

        L, Nd, a, _ = params[0]

        Vd = self.calculate_vd(params, X)

        return {
            "Material": self.material,
            "Channel Length (L)": f"{L:.3f} nm",
            "Doping (Nd)": f"{Nd:.2e} cm^-3",
            "Channel Thickness (a)": f"{a:.2f} nm",
            "Drain Voltage (Vd)": f"{Vd:.3f} V"
        }


def plot_training_history(history):
    plt.figure(figsize=(10,5))
    plt.plot(history.history['loss'])
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'])
    plt.grid()
    plt.show()


if __name__ == "__main__":
    designer = JFETDesigner(material='silicon')

    X_train, X_test, y_train, y_test = designer.generate_synthetic_training_data()

    history = designer.train(X_train, y_train, X_test, y_test)

    result = designer.predict_design(
        Id=1e-5,
        Vgs=-1.0,
        Idss=1e-3,
        A=1e-13
    )

    for k, v in result.items():
        print(f"{k}: {v}")