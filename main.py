#!/usr/bin/env python
# coding: utf-8

# In[2]:


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
import argparse
import sys

import warnings
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

class NanoMOSFETDesigner:
    def __init__(self):
        self.q: tf.Tensor = tf.constant(1.6e-19, dtype=tf.float32)
        self.eps_0: tf.Tensor = tf.constant(8.85e-12, dtype=tf.float32)
        self.eps_ox: float = 3.9
        self.eps_si: float = 11.7
        self.k: tf.Tensor = tf.constant(1.38e-23, dtype=tf.float32)
        self.temperature: tf.Tensor = tf.constant(300.0, dtype=tf.float32)
        self.ni: tf.Tensor = tf.constant(1.5e10, dtype=tf.float32)
        self.phi_ms: float = -0.56

        self.min_Lg: float = 0.1
        self.max_Lg: float = 1.0
        self.max_Lch: float = 5.0

        self.feature_scaler: StandardScaler = StandardScaler()
        self.design_scaler: StandardScaler = StandardScaler()

        self.model: Optional[keras.Model] = None
        self.training_history: Optional[Any] = None

        self.build_model()

    def build_model(self) -> None:
        inputs = keras.Input(shape=(4,))
        x = layers.Dense(256, activation='tanh')(inputs)
        x = layers.Dense(256, activation='tanh')(x)
        x = layers.Dense(256, activation='tanh')(x)
        outputs = layers.Dense(4)(x)

        self.model = keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                         loss='mse')

    def apply_nanoscale_constraints(self, design_params: np.ndarray) -> np.ndarray:
        Lg = np.clip(design_params[:, 0], self.min_Lg, self.max_Lg)
        doping = design_params[:, 1]
        Lch = np.clip(design_params[:, 2], Lg + 0.1, self.max_Lch)
        Vd = design_params[:, 3]
        return np.column_stack([Lg, doping, Lch, Vd])

    def calculate_vd(self, design_params: np.ndarray, input_features: np.ndarray) -> float:
        design_params_tensor = tf.convert_to_tensor(design_params, dtype=tf.float32)
        input_features_tensor = tf.convert_to_tensor(input_features, dtype=tf.float32)

        Lg, Na, Lch, _ = tf.unstack(design_params_tensor, axis=1)
        Id, _, Vtgm, tox = tf.unstack(input_features_tensor, axis=1)

        Lg = tf.clip_by_value(Lg, float(self.min_Lg), float(self.max_Lg))
        Lch = tf.clip_by_value(Lch, Lg + 0.1, float(self.max_Lch))

        Lg_m = Lg * 1e-9
        tox_m = tox * 1e-9
        Na_m = Na * 1e6

        phi_f = (self.k * self.temperature / self.q) * tf.math.log(Na_m / self.ni)
        Cox = self.eps_0 * tf.constant(self.eps_ox, dtype=tf.float32) / tox_m
        Qb = tf.math.sqrt(2.0 * self.q * tf.constant(self.eps_si, dtype=tf.float32) *
                         self.eps_0 * Na_m * phi_f)
        Vth = tf.constant(self.phi_ms, dtype=tf.float32) + 2.0 * phi_f + Qb / Cox

        delta_Vth = 0.1 / (Lg * tox)
        Vth += delta_Vth

        Vd = Vtgm + (Id * Lg_m) / (Cox * (0.7 - Vth) * (Lg/Lch) * 1e-6)

        return float(tf.clip_by_value(Vd, 0.1, 1.2).numpy()[0])

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              epochs: int = 200, batch_size: int = 32, verbose: int = 1) -> Any:
        """Train the model on the provided data"""

        # Scale the data
        X_train_norm = self.feature_scaler.fit_transform(X_train)
        y_train_norm = self.design_scaler.fit_transform(y_train)

        validation_data = None
        if X_val is not None and y_val is not None:
            X_val_norm = self.feature_scaler.transform(X_val)
            y_val_norm = self.design_scaler.transform(y_val)
            validation_data = (X_val_norm, y_val_norm)

        # Train the model
        history = self.model.fit(
            X_train_norm, y_train_norm,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            validation_split=0.2 if validation_data is None else 0,
            verbose=verbose
        )

        self.training_history = history
        return history

    def predict_design(self, Id: float, SS: float, Vtgm: float, tox: float) -> Dict[str, str]:
        """Predict MOSFET design parameters from performance metrics"""
        perf_input = np.array([[Id, SS, Vtgm, tox]], dtype=np.float32)
        perf_input_norm = self.feature_scaler.transform(perf_input)

        design_norm = self.model.predict(perf_input_norm, verbose=0)
        design_params = self.design_scaler.inverse_transform(design_norm)

        constrained_params = self.apply_nanoscale_constraints(design_params)
        Lg, doping, Lch, _ = constrained_params[0]

        Vd = self.calculate_vd(constrained_params, perf_input)

        return {
            'Gate length (Lg)': f"{Lg:.3f} nm",
            'Doping concentration': f"{doping:.2e} cm⁻³",
            'Channel length (Lch)': f"{Lch:.3f} nm",
            'Drain voltage (Vd)': f"{Vd:.3f} V",
            'Oxide thickness': f"{tox:.2f} nm"
        }

def load_and_preprocess_data(csv_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and preprocess data from CSV file"""
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        raise Exception(f"Error reading CSV file: {e}")

    column_mapping = {
        'Id': ['Id', 'DrainCurrent', 'ID', 'Idrain'],
        'SS': ['SS', 'SubthresholdSwing', 'ss'],
        'Vtgm': ['Vtgm', 'ThresholdVoltage', 'Vth', 'VT'],
        'tox': ['tox', 'OxideThickness', 'TOX'],
        'Lg': ['Lg', 'GateLength', 'Lg_nm'],
        'doping': ['doping', 'DopingConcentration', 'Na', 'Nd'],
        'Lch': ['Lch', 'ChannelLength', 'Lch_nm'],
        'Vd': ['Vd', 'DrainVoltage', 'VDS']
    }

    actual_columns: Dict[str, str] = {}
    for standard_name, possible_names in column_mapping.items():
        for name in possible_names:
            if name in df.columns:
                actual_columns[standard_name] = name
                break

    required_features = ['Id', 'SS', 'Vtgm', 'tox']
    missing_features = [f for f in required_features if f not in actual_columns]

    if missing_features:
        print(f"Warning: Missing columns {missing_features}. Using synthetic data generation.")
        num_samples = len(df)
        X = np.random.randn(num_samples, 4).astype(np.float32)

        Lg = np.random.uniform(0.1, 1.0, num_samples).astype(np.float32)
        doping = (10 ** np.random.uniform(18, 20, num_samples)).astype(np.float32)
        Lch = np.clip(Lg + np.random.uniform(0.1, 4.9, num_samples), None, 5.0).astype(np.float32)
        Vd = np.random.uniform(0.1, 0.8, num_samples).astype(np.float32)
        y = np.column_stack([Lg, doping, Lch, Vd])

        return train_test_split(X, y, test_size=0.2, random_state=42)

    X = df[[actual_columns['Id'], actual_columns['SS'],
            actual_columns['Vtgm'], actual_columns['tox']]].values.astype(np.float32)

    if all(col in actual_columns for col in ['Lg', 'doping', 'Lch', 'Vd']):
        y = df[[actual_columns['Lg'], actual_columns['doping'],
                actual_columns['Lch'], actual_columns['Vd']]].values.astype(np.float32)
        y[:, 0] = np.clip(y[:, 0], 0.1, 1.0)
        y[:, 2] = np.clip(y[:, 2], y[:, 0] + 0.1, 5.0)
    else:
        print("Generating nanoscale-constrained synthetic data...")
        num_samples = len(X)
        Lg = np.random.uniform(0.1, 1.0, num_samples).astype(np.float32)
        doping = (10 ** np.random.uniform(18, 20, num_samples)).astype(np.float32)
        Lch = np.clip(Lg + np.random.uniform(0.1, 4.9, num_samples), None, 5.0).astype(np.float32)
        Vd = np.random.uniform(0.1, 0.8, num_samples).astype(np.float32)
        y = np.column_stack([Lg, doping, Lch, Vd])

    return train_test_split(X, y, test_size=0.2, random_state=42)

def plot_training_history(history: Any, save_path: Optional[str] = None) -> None:
    """Plot the training history"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history.history['loss'], label='Training Loss', linewidth=2, color='#3498db')
    if 'val_loss' in history.history:
        ax.plot(history.history['val_loss'], label='Validation Loss', linewidth=2, color='#e74c3c')
    ax.set_title('Model Training History', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training history plot saved to: {save_path}")
    else:
        plt.show()

def print_design_results(results: Dict[str, str]) -> None:
    """Print design results in a formatted way"""
    print("\n" + "=" * 50)
    print("NANOSCALE MOSFET DESIGN RESULTS")
    print("=" * 50)
    print()

    for param, value in results.items():
        print(f"{param:30}: {value}")

    print()
    print("=" * 50)
    print("Quantum-aware design with Lg: 0.1-1nm constraints applied")
    print("=" * 50)

def interactive_mode(designer: NanoMOSFETDesigner) -> None:
    """Run interactive command-line mode for predictions"""
    print("\n" + "=" * 50)
    print("NanoMOSFET Designer - Interactive Mode")
    print("=" * 50)
    print("Enter 'quit' to exit\n")

    while True:
        try:
            print("\nEnter MOSFET performance parameters:")
            Id_input = input("Drain Current (A) [e.g., 1e-6]: ").strip()
            if Id_input.lower() == 'quit':
                break

            SS_input = input("Subthreshold Swing (mV/dec) [e.g., 80]: ").strip()
            if SS_input.lower() == 'quit':
                break

            Vtgm_input = input("Threshold Voltage (V) [e.g., 0.3]: ").strip()
            if Vtgm_input.lower() == 'quit':
                break

            tox_input = input("Oxide Thickness (nm) [e.g., 0.5]: ").strip()
            if tox_input.lower() == 'quit':
                break

            Id = float(Id_input)
            SS = float(SS_input)
            Vtgm = float(Vtgm_input)
            tox = float(tox_input)

            if Id <= 0 or SS <= 0 or tox <= 0:
                print("Error: All values must be positive")
                continue

            if tox < 0.1 or tox > 2:
                print("Warning: Oxide thickness outside typical nanoscale range (0.1-2nm)")

            results = designer.predict_design(Id, SS, Vtgm, tox)
            print_design_results(results)

        except ValueError as e:
            print(f"Error: Invalid input - {e}")
        except Exception as e:
            print(f"Error: Prediction failed - {e}")

def run_jupyter_mode():
    """Run in Jupyter/IPython environment"""
    print("\n" + "=" * 50)
    print("NanoMOSFET Designer - Jupyter Mode")
    print("=" * 50)
    print("\nThis is a Jupyter/IPython environment.")
    print("You can use the NanoMOSFETDesigner class directly.")
    print("\nExample usage:")
    print("  designer = NanoMOSFETDesigner()")
    print("  X_train, X_test, y_train, y_test = load_and_preprocess_data('data.csv')")
    print("  designer.train(X_train, y_train, X_test, y_test, epochs=200)")
    print("  results = designer.predict_design(1e-6, 80, 0.3, 0.5)")
    print("  print_design_results(results)")
    print("\n" + "=" * 50)

    # Return the class for interactive use
    return NanoMOSFETDesigner

def main():
    # Check if running in Jupyter/IPython
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            # Running in Jupyter/IPython
            run_jupyter_mode()
            return
    except ImportError:
        pass

    # Regular command-line mode
    parser = argparse.ArgumentParser(description='NanoMOSFET Designer - Nanoscale MOSFET Design Tool')
    parser.add_argument('--csv', type=str, help='Path to CSV file with training data')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    parser.add_argument('--predict', nargs=4, metavar=('ID', 'SS', 'VTGM', 'TOX'),
                       help='Predict design from given parameters: Id SS Vtgm tox')
    parser.add_argument('--plot', type=str, help='Save training history plot to file')
    parser.add_argument('--verbose', action='store_true', help='Print detailed training information')

    # Parse arguments, ignoring unknown args (like Jupyter kernel arguments)
    args, unknown = parser.parse_known_args()

    if unknown:
        print(f"Note: Ignoring unknown arguments: {unknown}")

    # Initialize designer
    designer = NanoMOSFETDesigner()

    # Train if CSV is provided
    if args.csv:
        if not os.path.exists(args.csv):
            print(f"Error: CSV file not found: {args.csv}")
            sys.exit(1)

        print(f"Loading data from: {args.csv}")
        try:
            X_train, X_test, y_train, y_test = load_and_preprocess_data(args.csv)
            print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

            print("Training model...")
            history = designer.train(
                X_train, y_train,
                X_val=X_test, y_val=y_test,
                epochs=args.epochs,
                batch_size=args.batch_size,
                verbose=1 if args.verbose else 0
            )

            print("Training completed!")
            print(f"Final training loss: {history.history['loss'][-1]:.6f}")
            if 'val_loss' in history.history:
                print(f"Final validation loss: {history.history['val_loss'][-1]:.6f}")

            # Plot training history
            if args.plot:
                plot_training_history(history, args.plot)
            elif args.verbose:
                plot_training_history(history)

        except Exception as e:
            print(f"Error during training: {e}")
            sys.exit(1)

    # Make single prediction
    if args.predict:
        if not designer.model:
            print("Error: Model not trained. Please provide a CSV file for training first.")
            sys.exit(1)

        try:
            Id, SS, Vtgm, tox = map(float, args.predict)
            results = designer.predict_design(Id, SS, Vtgm, tox)
            print_design_results(results)
        except ValueError as e:
            print(f"Error: Invalid prediction parameters - {e}")
            sys.exit(1)

    # Interactive mode
    elif args.interactive:
        if not designer.model:
            print("Error: Model not trained. Please provide a CSV file for training first.")
            sys.exit(1)
        interactive_mode(designer)

    # No arguments - show help
    elif not args.csv and not args.predict and not args.interactive:
        parser.print_help()
        print("\nExample usage:")
        print("  Train model: python main.py --csv data.csv --epochs 200")
        print("  Predict: python main.py --csv data.csv --predict 1e-6 80 0.3 0.5")
        print("  Interactive: python main.py --csv data.csv --interactive")

if __name__ == "__main__":
    main()

