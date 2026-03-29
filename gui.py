import os
import json
import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from mosfet import MOSFETDesigner, load_and_preprocess_data
from sklearn.model_selection import train_test_split
import threading
import traceback
from datetime import datetime
import io
import base64
import matplotlib.pyplot as plt

app = Flask(__name__)
CORS(app)

# Global state management
designers = {}  # Store designers for different materials
training_status = {}
results_cache = {}


class DesignerManager:
    def __init__(self):
        self.designers = {}
        self.training_in_progress = {}
        self.datasets = {}

    def get_or_create_designer(self, material):
        if material not in self.designers:
            self.designers[material] = MOSFETDesigner(material=material)
        return self.designers[material]

    def set_dataset(self, material, X_train, X_test, y_train, y_test):
        self.datasets[material] = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }

    def get_dataset(self, material):
        return self.datasets.get(material)


manager = DesignerManager()


# ============================================================================
# HEALTH CHECK & INFO ENDPOINTS
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'version': '2.0'}), 200


@app.route('/api/info', methods=['GET'])
def app_info():
    """Application information"""
    return jsonify({
        'name': 'MOSFETDesigner Web API',
        'version': '2.0',
        'materials': ['silicon', 'sige', 'gaas', 'gan', 'sic', 'inp', 'diamond'],
        'features': [
            'multi-material support',
            'synthetic data generation',
            'neural network training',
            'design prediction',
            'training visualization',
            'custom data loading'
        ]
    }), 200


# ============================================================================
# MATERIAL MANAGEMENT ENDPOINTS
# ============================================================================

@app.route('/api/materials', methods=['GET'])
def get_materials():
    """Get list of supported materials"""
    materials = ['silicon', 'sige', 'gaas', 'gan', 'sic', 'inp', 'diamond']
    descriptions = {
        'silicon': 'Silicon (Si) - Industry standard',
        'sige': 'Silicon Germanium (SiGe) - Enhanced mobility',
        'gaas': 'Gallium Arsenide (GaAs) - High-speed III-V',
        'gan': 'Gallium Nitride (GaN) - Wide bandgap power electronics',
        'sic': 'Silicon Carbide (SiC) - High-temperature',
        'inp': 'Indium Phosphide (InP) - High electron mobility',
        'diamond': 'Diamond (C) - Emerging wide bandgap'
    }
    return jsonify({
        'materials': [
            {'id': m, 'name': descriptions[m]} for m in materials
        ]
    }), 200


@app.route('/api/material/<material>', methods=['POST'])
def select_material(material):
    """Select and initialize a material"""
    try:
        valid_materials = ['silicon', 'sige', 'gaas', 'gan', 'sic', 'inp', 'diamond']
        if material not in valid_materials:
            return jsonify({'error': f'Invalid material. Choose from {valid_materials}'}), 400

        designer = manager.get_or_create_designer(material)
        training_status[material] = {'status': 'ready', 'progress': 0}

        return jsonify({
            'material': material,
            'status': 'initialized',
            'properties': {
                'name': designer.material_name,
                'bandgap': float(designer.bandgap),
                'electron_affinity': float(designer.electron_affinity)
            }
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# DATA GENERATION ENDPOINTS
# ============================================================================

@app.route('/api/generate-data', methods=['POST'])
def generate_synthetic_data():
    """Generate synthetic training data"""
    try:
        data = request.json
        material = data.get('material', 'silicon')
        n_samples = int(data.get('n_samples', 5000))

        if n_samples < 100 or n_samples > 1000000:
            return jsonify({'error': 'n_samples must be between 100 and 1,000,000'}), 400

        designer = manager.get_or_create_designer(material)
        training_status[material] = {'status': 'generating', 'progress': 50}

        X_train, y_train = designer.generate_synthetic_training_data(n_samples)
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )

        manager.set_dataset(material, X_train, X_test, y_train, y_test)
        training_status[material] = {'status': 'ready', 'progress': 100}

        return jsonify({
            'material': material,
            'status': 'success',
            'generated_samples': n_samples,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'timestamp': datetime.now().isoformat()
        }), 200
    except Exception as e:
        training_status[material] = {'status': 'error', 'error': str(e)}
        return jsonify({'error': str(e)}), 500


@app.route('/api/load-csv', methods=['POST'])
def load_csv_data():
    """Load training data from CSV file"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        material = request.form.get('material', 'silicon')
        file = request.files['file']

        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'File must be CSV format'}), 400

        # Save temporarily and load
        temp_path = f'/tmp/{file.filename}'
        file.save(temp_path)

        X_train, X_test, y_train, y_test = load_and_preprocess_data(temp_path, material)
        manager.set_dataset(material, X_train, X_test, y_train, y_test)

        os.remove(temp_path)

        return jsonify({
            'material': material,
            'status': 'success',
            'filename': file.filename,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'timestamp': datetime.now().isoformat()
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# MODEL TRAINING ENDPOINTS
# ============================================================================

@app.route('/api/train', methods=['POST'])
def train_model():
    """Train the neural network model"""
    try:
        data = request.json
        material = data.get('material', 'silicon')
        epochs = int(data.get('epochs', 100))
        batch_size = int(data.get('batch_size', 64))

        # Validate parameters
        if epochs < 1 or epochs > 1000:
            return jsonify({'error': 'epochs must be between 1 and 1000'}), 400
        if batch_size < 1 or batch_size > 256:
            return jsonify({'error': 'batch_size must be between 1 and 256'}), 400

        # Get dataset
        dataset = manager.get_dataset(material)
        if not dataset:
            return jsonify({'error': 'Please generate or load data first'}), 400

        designer = manager.get_or_create_designer(material)
        training_status[material] = {'status': 'training', 'progress': 25}

        # Train model
        history = designer.train(
            dataset['X_train'], dataset['y_train'],
            X_val=dataset['X_test'], y_val=dataset['y_test'],
            epochs=epochs, batch_size=batch_size, verbose=0
        )

        final_loss = float(history.history['loss'][-1])
        final_val_loss = float(history.history['val_loss'][-1])

        training_status[material] = {
            'status': 'completed',
            'progress': 100,
            'loss': final_loss,
            'val_loss': final_val_loss
        }

        results_cache[material] = {
            'history': history,
            'loss': final_loss,
            'val_loss': final_val_loss
        }

        return jsonify({
            'material': material,
            'status': 'training_complete',
            'epochs': epochs,
            'batch_size': batch_size,
            'final_loss': final_loss,
            'final_val_loss': final_val_loss,
            'timestamp': datetime.now().isoformat()
        }), 200
    except Exception as e:
        training_status[material] = {'status': 'error', 'error': str(e)}
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/api/training-status/<material>', methods=['GET'])
def get_training_status(material):
    """Get training status for a material"""
    status = training_status.get(material, {'status': 'not_initialized'})
    return jsonify(status), 200


# ============================================================================
# PREDICTION ENDPOINTS
# ============================================================================

@app.route('/api/predict', methods=['POST'])
def predict_design():
    """Make a design prediction"""
    try:
        data = request.json
        material = data.get('material', 'silicon')

        # Validate inputs
        try:
            Id = float(data.get('Id'))
            SS = float(data.get('SS'))
            Vtgm = float(data.get('Vtgm'))
            tox = float(data.get('tox'))
        except (TypeError, ValueError):
            return jsonify({'error': 'Invalid input values. All must be numbers'}), 400

        # Check positive values
        if Id <= 0 or SS <= 0 or tox <= 0:
            return jsonify({'error': 'Drain current, subthreshold swing, and oxide thickness must be positive'}), 400

        designer = manager.get_or_create_designer(material)
        if designer.model is None:
            return jsonify({'error': 'Model not trained. Please train first'}), 400

        results = designer.predict_design(Id, SS, Vtgm, tox)

        return jsonify({
            'material': material,
            'input': {
                'Id': Id,
                'SS': SS,
                'Vtgm': Vtgm,
                'tox': tox
            },
            'predictions': results,
            'timestamp': datetime.now().isoformat()
        }), 200
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/api/validate-inputs', methods=['POST'])
def validate_inputs():
    """Validate input parameters before prediction"""
    try:
        data = request.json
        material = data.get('material', 'silicon')

        try:
            Id = float(data.get('Id', 0))
            SS = float(data.get('SS', 0))
            Vtgm = float(data.get('Vtgm', 0))
            tox = float(data.get('tox', 0))
        except (TypeError, ValueError):
            return jsonify({'valid': False, 'error': 'Invalid number format'}), 400

        errors = []
        if Id <= 0:
            errors.append('Drain current (Id) must be positive')
        if SS <= 0:
            errors.append('Subthreshold swing (SS) must be positive')
        if Vtgm <= 0:
            errors.append('Threshold voltage (Vtgm) must be positive')
        if tox <= 0:
            errors.append('Oxide thickness (tox) must be positive')

        # Check ranges
        if Id < 1e-9 or Id > 1e-3:
            errors.append('Drain current should be between 1 nA and 1 mA')
        if SS < 60 or SS > 120:
            errors.append('Subthreshold swing should be between 60 and 120 mV/dec')
        if tox < 0.3 or tox > 3.0:
            errors.append('Oxide thickness should be between 0.3 and 3.0 nm')

        if errors:
            return jsonify({'valid': False, 'errors': errors}), 200

        return jsonify({'valid': True, 'message': 'All inputs valid'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# VISUALIZATION ENDPOINTS
# ============================================================================

@app.route('/api/training-plot/<material>', methods=['GET'])
def get_training_plot(material):
    """Get training history plot as base64 image"""
    try:
        if material not in results_cache:
            return jsonify({'error': 'No training history available'}), 404

        history = results_cache[material]['history']

        # Create plot
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

        # Convert to base64
        img_io = io.BytesIO()
        plt.savefig(img_io, format='png', dpi=150, bbox_inches='tight')
        img_io.seek(0)
        img_base64 = base64.b64encode(img_io.getvalue()).decode()
        plt.close()

        return jsonify({
            'material': material,
            'image': f'data:image/png;base64,{img_base64}'
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/download-plot/<material>', methods=['GET'])
def download_plot(material):
    """Download training plot as PNG file"""
    try:
        if material not in results_cache:
            return jsonify({'error': 'No training history available'}), 404

        history = results_cache[material]['history']

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

        img_io = io.BytesIO()
        plt.savefig(img_io, format='png', dpi=150, bbox_inches='tight')
        img_io.seek(0)
        plt.close()

        return send_file(img_io, mimetype='image/png',
                         as_attachment=True,
                         download_name=f'training_history_{material}.png')
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)