# MOSFETDesigner: AI-Powered Nanoscale MOSFET Design Tool

## Overview

**MOSFETDesigner** is an advanced machine learning-driven application for predicting optimal nanoscale Metal-Oxide-Semiconductor Field-Effect Transistor (MOSFET) design parameters. This tool integrates deep neural networks with semiconductor physics principles to rapidly generate design specifications based on target performance metrics. It features an intuitive graphical user interface (GUI) and supports multiple semiconductor materials with material-specific design constraints.

## Project Purpose

The MOSFETDesigner addresses the complex challenge of nanoscale transistor design by leveraging artificial intelligence to bridge the gap between desired device performance characteristics and the physical design parameters required to achieve them. This tool is valuable for:

- **Semiconductor Design Engineers**: Accelerating device design exploration and optimization with interactive visual feedback
- **Research Institutions**: Investigating material alternatives for advanced node technologies with batch processing capabilities
- **Educational Institutions**: Teaching semiconductor device physics and design automation through hands-on GUI interaction
- **Design Automation Workflows**: Integrating AI-driven parameter prediction into CAD tools and design flows

## Features

### Core Capabilities

- **Multi-Material Support**: Design MOSFETs across 7 semiconductor material platforms:
  - **Silicon (Si)**: Industry-standard, baseline comparisons
  - **Silicon Germanium (SiGe)**: Enhanced mobility for advanced nodes
  - **Gallium Arsenide (GaAs)**: High-speed III-V compound semiconductor
  - **Gallium Nitride (GaN)**: Wide bandgap for power electronics
  - **Silicon Carbide (SiC)**: Wide bandgap for high-temperature applications
  - **Indium Phosphide (InP)**: High electron mobility III-V material
  - **Diamond (C)**: Emerging wide bandgap material with exceptional properties

- **Physics-Based Design Engine**: Incorporates fundamental semiconductor equations:
  - Threshold voltage (Vth) calculation with short-channel effect modeling
  - Work function difference adjustments per material
  - Material-dependent mobility considerations
  - Gate oxide capacitance relationships
  - On-resistance approximations

- **Material-Specific Constraints**: Automatic enforcement of realistic design boundaries:
  - Gate length (Lg) ranges: 0.1–5.0 nm (material dependent)
  - Channel length (Lch): 5–20 nm (material dependent)
  - Doping concentration: 10¹⁵–10²⁰ cm⁻³
  - Supply voltage: 0.1–5.0 V (material dependent)

- **Neural Network Optimization**: Deep learning model for rapid parameter prediction:
  - 3-layer architecture with batch normalization and dropout regularization
  - 512→256→128 neuron configuration for robust feature extraction
  - Adaptive learning rate scheduling and early stopping
  - Validation-based model selection

- **Graphical User Interface (GUI)**:
  - Real-time material selection with instant feedback
  - Interactive parameter input with validation
  - Training progress visualization with status updates
  - Results display with formatted design specifications
  - CSV file loading for custom training data
  - Training history plot generation and export
  - Threading support for non-blocking operations

### Performance Parameters

The tool predicts design parameters from the following performance specifications:

| Parameter | Symbol | Range | Unit |
|-----------|--------|-------|------|
| Drain Current | Id | 1 nA – 1 mA | A |
| Subthreshold Swing | SS | 60–120 | mV/dec |
| Threshold Voltage | Vtgm | 0.1–1.0 | V |
| Gate Oxide Thickness | tox | 0.3–3.0 | nm |

### Design Output Parameters

The model generates the following optimized design specifications:

| Parameter | Symbol | Description |
|-----------|--------|-------------|
| Gate Length | Lg | Critical dimension defining short-channel behavior |
| Doping Concentration | Na | Channel doping level for threshold voltage tuning |
| Channel Length | Lch | Total channel length for parasitic resistance control |
| Drain Voltage | Vd | Operating supply voltage for performance optimization |

## Technical Architecture

### Dependencies

**Machine Learning & Scientific Computing**
- TensorFlow 2.21.0 + Keras 3.13.2: Deep learning framework and high-level API
- NumPy 2.4.3: Numerical computation and array operations
- Pandas 3.0.1: Data loading, preprocessing, and CSV handling
- Scikit-learn 1.8.0: Data normalization and train-test splitting
- SciPy 1.17.1: Scientific computing utilities

**Visualization & User Interface**
- CustomTkinter 5.2.2: Modern dark-themed GUI framework
- Matplotlib 3.10.8: Training history plots and performance visualization
- Tkinter: Cross-platform GUI toolkit (included with Python)
- Pillow 12.1.1: Image processing support

**Development Environment**
- Python 3.12.13+ (required for TensorFlow compatibility)
- IPython 8.12.3: Interactive shell support
- Jupyter ecosystem: Notebook support for exploratory analysis

**Complete dependency list** available in `requirements.txt`

### System Requirements

- **Python Version**: 3.12.13 or later
- **Memory**: Minimum 4 GB RAM (8 GB+ recommended for large datasets)
- **Processor**: Modern multi-core CPU (GPU support available via TensorFlow)
- **Storage**: ~500 MB for dependencies and trained models
- **Operating System**: Windows, macOS, or Linux (with X11 for GUI)

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/PriyanjanMitra/MOSFETDesigner.git
cd MOSFETDesigner
```

### Step 2: Create Virtual Environment (Recommended)

```bash
python3.12 -m venv mosfet_env
source mosfet_env/bin/activate  # Linux/Mac
# or
mosfet_env\Scripts\activate  # Windows
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python gui.py --help
```

Or simply launch the GUI:

```bash
python gui.py
```

## Usage Guide

### Quick Start: Launching the GUI

The simplest way to use MOSFETDesigner is through the graphical interface:

```bash
python gui.py
```

This launches the interactive GUI with the following sections:
- **Material Selection**: Dropdown menu to choose semiconductor material
- **Model Training**: Data generation and model training controls
- **Performance Parameters**: Input fields for design specifications
- **Design Results**: Output display with predicted parameters

### GUI Workflow

#### Step 1: Select Material

1. Open `gui.py`
2. Choose desired material from the dropdown (default: Silicon)
3. Status indicator shows material initialization

#### Step 2: Generate or Load Training Data

**Option A: Generate Synthetic Data**
- Click "Generate Synthetic Data" button
- System creates 5,000 synthetic training samples
- Progress bar shows generation status
- Results pane confirms sample counts

**Option B: Load Custom CSV Data**
- Click "Load CSV" button
- Select your CSV file (must contain: Id, SS, Vtgm, tox, Lg, doping, Lch, Vd columns)
- System automatically splits into training/test sets (80/20 split)

#### Step 3: Train the Model

1. Configure training parameters:
   - **Epochs**: Number of training iterations (default: 100, range: 1–1000)
   - **Batch Size**: Samples per gradient update (default: 64, range: 1–256)
   - **Save Training Plot**: Optional checkbox to export training history visualization

2. Click "Train Model" button

3. Monitor progress:
   - Status label updates in real-time
   - Progress bar indicates training activity
   - Final loss metrics displayed upon completion

#### Step 4: Make Predictions

1. Enter MOSFET performance parameters:
   - **Drain Current (Id)**: Example values: 1e-6, 1e-9, 1e-3 A
   - **Subthreshold Swing (SS)**: Example values: 75, 80, 90 mV/dec
   - **Threshold Voltage (Vtgm)**: Example values: 0.3, 0.4, 0.5 V
   - **Oxide Thickness (tox)**: Example values: 1.0, 1.5, 2.0 nm

2. Click "Predict Design" button

3. Results display shows:
   - Selected material
   - Gate length (Lg)
   - Doping concentration
   - Channel length (Lch)
   - Drain voltage (Vd)
   - Oxide thickness confirmation

4. Optional: Click "Clear" to reset all input fields

### Example Workflows

#### Workflow 1: Compare Silicon vs. GaN Designs

**For Silicon:**
1. Launch GUI: `python gui.py`
2. Select "silicon" from dropdown
3. Click "Generate Synthetic Data"
4. Set Epochs: 150, Batch Size: 64
5. Click "Train Model"
6. Enter: Id=1e-6, SS=80, Vtgm=0.4, tox=1.0
7. Click "Predict Design"
8. Note the resulting Lg and Vd values

**For GaN:**
1. Select "gan" from dropdown (triggers re-initialization)
2. Click "Generate Synthetic Data"
3. Click "Train Model" (uses same parameters)
4. Enter same performance parameters: Id=1e-6, SS=80, Vtgm=0.4, tox=1.0
5. Click "Predict Design"
6. Compare gate lengths and supply voltages

**Analysis**: Observe differences in design parameters across materials to understand material trade-offs.

#### Workflow 2: Design Optimization Loop

1. Generate synthetic data for chosen material
2. Train model with default settings
3. Make initial prediction with baseline performance specs
4. Iteratively refine performance parameters
5. Observe design parameter trends
6. Identify optimal operating points
7. Export training plots for documentation

#### Workflow 3: Custom Data Training

1. Prepare CSV file with historical device data
2. Launch GUI and select material
3. Click "Load CSV" and browse to your file
4. Enable "Save Training Plot" checkbox
5. Specify plot filename: `my_training_results.png`
6. Click "Train Model" with custom epochs (e.g., 200)
7. View final loss metrics
8. Make predictions using trained model
9. Training plot automatically saved

### GUI Features in Detail

#### Material Selection Dropdown
- **Available Options**: silicon, sige, gaas, gan, sic, inp, diamond
- **Dynamic Initialization**: Selecting a material reinitializes the designer with material-specific constraints
- **Status Feedback**: Color-coded status messages (blue for changes, green for success, red for errors)

#### Training Controls
- **Generate Synthetic Data**: Creates 5,000 physics-based training samples
  - Covers full design space
  - Material-specific constraints applied
  - Automatic train/test split (80/20)

- **Train Model**: Trains neural network on loaded/generated data
  - Configurable epochs (default: 100)
  - Configurable batch size (default: 64)
  - Real-time loss monitoring
  - Early stopping on validation plateau

- **Load CSV**: Import custom training datasets
  - File browser dialog
  - Automatic data validation
  - Supports large datasets

#### Training Options
- **Epochs Entry**: Adjust number of training iterations (higher = more iterations, slower training)
- **Batch Size Entry**: Modify gradient update frequency (higher = more memory efficient, potentially slower convergence)
- **Save Training Plot Checkbox**: Enable/disable plot export
- **Plot Path Entry**: Specify output filename for training history visualization

#### Status and Progress Indicators
- **Status Label**: Real-time feedback (Ready, Generating data, Training model, Complete, Error states)
- **Color Coding**: 
  - Green: Success
  - Orange: In progress
  - Blue: State change
  - Red: Error
- **Progress Bar**: Visual indication of long-running operations

#### Parameter Input Fields
- All fields include placeholder text with example values
- Real-time validation on prediction
- Accepts scientific notation (e.g., 1e-6)
- Error messages for invalid inputs

#### Results Display Pane
- Large scrollable text area
- Formatted output with clear parameter labels
- Training progress updates
- Prediction results with physics constraints confirmation

## Data and Training

### Synthetic Data Generation

The `generate_synthetic_training_data()` method creates training samples covering the full design space:

- **Drain Current**: Logarithmically distributed from 1 nA to 1 mA
- **Subthreshold Swing**: Uniformly distributed 60–120 mV/dec
- **Threshold Voltage**: Uniformly distributed 0.1–1.0 V
- **Gate Length**: Inversely correlated with performance metrics
- **Samples**: 5,000 per generation (split 80/20 for training/validation)

### Model Architecture

```
Input Layer (4 features)
    ↓
Dense(512, relu) + BatchNorm + Dropout(0.2)
    ↓
Dense(256, relu) + BatchNorm + Dropout(0.2)
    ↓
Dense(128, relu) + BatchNorm
    ↓
Output Layer (4 design parameters)
```

**Training Configuration:**
- Optimizer: Adam (learning_rate=0.0005)
- Loss Function: Mean Squared Error (MSE)
- Metrics: Mean Absolute Error (MAE)
- Callbacks: 
  - ReduceLROnPlateau: Adaptive learning rate reduction
  - EarlyStopping: Prevents overfitting (patience=50 epochs)

### CSV Data Format

If loading custom training data, ensure the CSV contains these columns:

```csv
Id,SS,Vtgm,tox,Lg,doping,Lch,Vd
1e-6,80,0.3,1.0,0.185,2.34e+18,1.850,0.450
1e-5,75,0.4,1.2,0.150,2.50e+18,2.000,0.520
1e-7,85,0.25,0.8,0.220,2.10e+18,1.700,0.380
```

## Physical Model Implementation

### Threshold Voltage Calculation

The threshold voltage computation incorporates:

$$V_{th} = \phi_{ms} + 2\phi_f + \frac{Q_b}{C_{ox}} + \Delta V_{th}$$

Where:
- **φ_ms**: Work function difference (material dependent)
- **φ_f**: Fermi potential related to doping concentration
- **Q_b/C_ox**: Bulk charge term normalized by oxide capacitance
- **ΔV_th**: Short-channel effect correction (Lg dependent)

### Material-Specific Properties

Each material includes:
- Relative permittivity (ε_r)
- Intrinsic carrier concentration (n_i)
- Electron and hole mobility (μ_n, μ_p)
- Bandgap energy (E_g)
- Electron affinity (χ)

### Short-Channel Effect Modeling

Wide bandgap materials (GaN, SiC, Diamond) experience reduced short-channel effects:
- **ΔV_th = 0.02 / (Lg + 0.1)** for wide bandgap materials
- **ΔV_th = 0.05 / (Lg + 0.1)** for conventional materials

## Troubleshooting

### Common Issues and Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| GUI won't start | CustomTkinter not installed | Run `pip install -r requirements.txt` |
| "Please train model first" error | Attempting prediction before training | Click "Generate Synthetic Data" then "Train Model" |
| CSV load fails | Missing required columns | Ensure CSV has: Id, SS, Vtgm, tox, Lg, doping, Lch, Vd |
| Training seems slow | Large batch size or high epochs | Reduce epochs or increase batch size |
| Out of memory error | Insufficient RAM | Reduce batch size or number of samples |
| Invalid input error | Non-numeric values in parameter fields | Use valid numbers (e.g., 1e-6, not "one micron") |

## File Structure

```
MOSFETDesigner/
├── gui.py                    # Main GUI application script
├── mosfet.py                 # Core MOSFET designer and neural network
├── SimpleMOSFinal.csv        # Example training dataset
├── requirements.txt          # Python package dependencies
├── LICENSE                   # MIT License
└── README.md                # This file
```

## Limitations and Considerations

### Model Assumptions

1. **Simplified Physics**: Advanced phenomena (quantum confinement, tunneling) are not explicitly modeled
2. **Uniform Channel**: Assumes uniform doping and oxide thickness
3. **Training Data Range**: Predictions are most accurate within the training data bounds
4. **Thermal Effects**: Assumes room temperature (300 K) operation
5. **Single-Gate Devices**: Does not model multi-gate structures (FinFET, GAAFET)
6. **Static Operation**: Focuses on DC performance, not transient effects

### Accuracy Factors

- Model accuracy depends on training data quality and representativeness
- Extrapolation beyond training ranges may produce unrealistic results
- Material property values are approximations from literature
- GUI validation ensures inputs are within physically reasonable ranges

### Best Practices

1. **Validate Results**: Cross-check predictions against published device data or TCAD simulations
2. **Iterative Refinement**: Use tool in conjunction with professional simulators for final verification
3. **Parameter Ranges**: Maintain input parameters within specified realistic ranges
4. **Data Logging**: Save prediction history for design documentation and traceability
5. **Periodic Retraining**: Retrain model with new data periodically for improved accuracy

## Performance Metrics

Typical model performance on validation datasets:

- **Mean Absolute Error (MAE)**: ~2–5% of parameter range
- **Training Time**: ~60–120 seconds (5,000 samples, 100 epochs on CPU)
- **Prediction Time**: <100 ms per design (instantaneous in GUI)
- **Memory Footprint**: ~150 MB (model + GUI + dependencies)
- **GUI Responsiveness**: Threading ensures non-blocking operations

## Future Development Roadmap

- Quantum transport model integration
- Multi-gate device architecture support (FinFET, GAAFET)
- Temperature-dependent parameter predictions
- Device reliability metrics (NBTI, HCI modeling)
- Integration with commercial SPICE simulators
- GPU acceleration for batch predictions
- Advanced visualization (3D design parameter space)
- Export to CAD tool formats
- Machine learning explainability features
- Model versioning and management

## Contributing

Contributions are welcome! Please ensure:

1. Code follows PEP 8 style guidelines
2. New features include documentation and examples
3. GUI enhancements maintain usability and responsiveness
4. All dependencies are added to requirements.txt
5. Changes are tested with multiple material options
6. GUI updates use CustomTkinter for consistent theming

## License

This project is licensed under the MIT License. See the LICENSE file for details.

**Copyright © 2026 Priyanjan Mitra**

## Support and Contact

For bug reports, feature requests, or technical questions:

- **GitHub Issues**: https://github.com/PriyanjanMitra/MOSFETDesigner/issues
- **Author**: Priyanjan Mitra
- **Documentation**: This README and inline code comments

## References

### Semiconductor Physics
- Tsividis, Y., & McAndrew, C. (2011). *Operating and Modeling of the MOS Transistor* (3rd ed.). Oxford University Press.
- Colinge, J. P., et al. (2015). *FinFETs and Other Multi-Gate FETs*. Springer.

### Machine Learning for Device Design
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
- Sebastiani, T. (2019). *Machine Learning for Embedded Systems*. Packt Publishing.

### Material Properties
- Sze, S. M., & Ng, K. K. (2006). *Physics of Semiconductor Devices* (3rd ed.). John Wiley & Sons.

### GUI Development
- CustomTkinter Documentation: https://github.com/TomSchimansky/CustomTkinter

---

**Version**: 1.1.0 (GUI Edition)  
**Last Updated**: March 2026  
**Status**: Active Development  
**Main Entry Point**: `gui.py`
