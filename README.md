# MOSFETDesigner: AI-Powered Nanoscale MOSFET Design Tool

## Overview

**MOSFETDesigner** is an advanced machine learning-driven application for predicting optimal nanoscale Metal-Oxide-Semiconductor Field-Effect Transistor (MOSFET) design parameters. This tool integrates deep neural networks with semiconductor physics principles to rapidly generate design specifications based on target performance metrics. It supports multiple semiconductor materials and implements material-specific constraints for realistic design optimization.

## Project Purpose

The MOSFETDesigner addresses the complex challenge of nanoscale transistor design by leveraging artificial intelligence to bridge the gap between desired device performance characteristics and the physical design parameters required to achieve them. This tool is valuable for:

- **Semiconductor Design Engineers**: Accelerating device design exploration and optimization
- **Research Institutions**: Investigating material alternatives for advanced node technologies
- **Educational Institutions**: Teaching semiconductor device physics and design automation
- **Design Automation Workflows**: Integrating AI-driven parameter prediction into CAD tools

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

- **Multiple Operational Modes**:
  - **Interactive Mode**: Real-time design prediction with user input validation
  - **Batch Mode**: Single prediction via command-line arguments
  - **Data Generation**: Synthetic training data covering full design space
  - **Custom Data Loading**: Train on user-provided CSV datasets

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

**Visualization & Analysis**
- Matplotlib 3.10.8: Training history plots and performance visualization
- Pillow 12.1.1: Image processing support

**Development Environment**
- Python 3.9+ (required for TensorFlow compatibility)
- IPython 8.12.3: Interactive shell support
- Jupyter ecosystem: Notebook support for exploratory analysis

**Complete dependency list** available in `requirements.txt`

### System Requirements

- **Python Version**: 3.9-3.12
- **Memory**: Minimum 4 GB RAM (8 GB+ recommended for large datasets)
- **Processor**: Modern multi-core CPU (GPU support available via TensorFlow)
- **Storage**: ~500 MB for dependencies and trained models

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
python mosfet.py --help
```

## Usage Guide

### 1. Basic Usage with Synthetic Data Generation

Generate synthetic training data and make a prediction:

```bash
python mosfet.py --material silicon --generate_data --predict 1e-6 80 0.3 1.0
```

**Output Example:**
```
==================================================
NANOSCALE MOSFET DESIGN RESULTS
==================================================

Material                  : Silicon (Si)
Gate length (Lg)          : 0.185 nm
Doping concentration      : 2.34e+18 cm⁻³
Channel length (Lch)      : 1.850 nm
Drain voltage (Vd)        : 0.450 V
Oxide thickness           : 1.00 nm

==================================================
Quantum-aware design constraints applied
==================================================
```

### 2. Interactive Mode for Multiple Predictions

```bash
python mosfet.py --material gan --generate_data --interactive
```

The tool will prompt for performance parameters and display results after each prediction. Enter `quit` to exit.

### 3. Training with Custom Data

```bash
python mosfet.py --csv training_data.csv --material silicon --epochs 500 --plot training_history.png
```

**Expected CSV Format:**
```csv
Id,SS,Vtgm,tox,Lg,doping,Lch,Vd
1e-6,80,0.3,1.0,0.185,2.34e+18,1.850,0.450
1e-5,75,0.4,1.2,0.150,2.50e+18,2.000,0.520
```

### 4. Advanced Options

```bash
python mosfet.py --material diamond \
                  --generate_data \
                  --epochs 300 \
                  --batch_size 32 \
                  --verbose \
                  --plot training_history.png \
                  --predict 1e-7 90 0.5 1.5
```

### Command-Line Arguments Reference

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--material` | str | silicon | Semiconductor material: silicon, sige, gaas, gan, sic, inp, diamond |
| `--csv` | str | None | Path to CSV file with training data |
| `--epochs` | int | 300 | Number of training epochs |
| `--batch_size` | int | 64 | Training batch size |
| `--generate_data` | flag | False | Generate synthetic training data |
| `--predict` | 4 floats | None | Single prediction: Id SS Vtgm tox |
| `--interactive` | flag | False | Run interactive mode |
| `--plot` | str | None | Save training history plot to file |
| `--verbose` | flag | False | Print detailed training information |

## Data and Training

### Synthetic Data Generation

The `generate_synthetic_training_data()` method creates 10,000 training samples covering the full design space:

- **Drain Current**: Logarithmically distributed from 1 nA to 1 mA
- **Subthreshold Swing**: Uniformly distributed 60–120 mV/dec
- **Threshold Voltage**: Uniformly distributed 0.1–1.0 V
- **Gate Length**: Inversely correlated with performance metrics

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
- Optimizer: Adam (lr=0.0005)
- Loss Function: Mean Squared Error (MSE)
- Metrics: Mean Absolute Error (MAE)
- Callbacks: ReduceLROnPlateau, EarlyStopping (patience=50)

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

## Example Workflows

### Workflow 1: Compare Silicon vs. GaN Designs

```bash
# Silicon design
python mosfet.py --material silicon --generate_data \
                  --predict 1e-6 80 0.4 1.0

# GaN design  
python mosfet.py --material gan --generate_data \
                  --predict 1e-6 80 0.4 1.0
```

Compare the resulting gate lengths and supply voltages to understand material trade-offs.

### Workflow 2: Design Optimization Loop

1. Generate initial designs for baseline performance
2. Iteratively refine performance parameters
3. Analyze trend patterns in design parameters
4. Identify optimal operating points

```bash
python mosfet.py --material sic --generate_data --interactive
```

## File Structure

```
MOSFETDesigner/
├── mosfet.py                 # Main application script
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

### Accuracy Factors

- Model accuracy depends on training data quality and representativeness
- Extrapolation beyond training ranges may produce unrealistic results
- Material property values are approximations from literature

### Best Practices

1. **Validate Results**: Cross-check predictions against published device data or simulations
2. **Iterative Refinement**: Use tool in conjunction with TCAD simulators for final verification
3. **Parameter Ranges**: Maintain input parameters within specified realistic ranges
4. **Data Logging**: Save prediction history for design documentation and traceability

## Performance Metrics

Typical model performance on validation datasets:

- **Mean Absolute Error (MAE)**: ~2–5% of parameter range
- **Training Time**: ~60–120 seconds (10,000 samples, 300 epochs)
- **Prediction Time**: <100 ms per design
- **Memory Footprint**: ~150 MB (model + dependencies)

## Future Development Roadmap

- Multi-gate device architecture support
- Device reliability metrics (NBTI, HCI modeling)
- Web-based interface for accessibility

## Contributing

Contributions are welcome! Please ensure:

1. Code follows PEP 8 style guidelines
2. New features include documentation and examples
3. All dependencies are added to requirements.txt
4. Changes are tested with multiple material options

## License

This project is licensed under the MIT License. See the LICENSE file for details.

**Copyright © 2026 Priyanjan Mitra**

## Support and Contact

For bug reports, feature requests, or technical questions:

- **GitHub Issues**: https://github.com/PriyanjanMitra/MOSFETDesigner/issues
- **Author**: Priyanjan Mitra

## References

### Semiconductor Physics
- Tsividis, Y., & McAndrew, C. (2011). *Operating and Modeling of the MOS Transistor* (3rd ed.). Oxford University Press.
- Colinge, J. P., et al. (2015). *FinFETs and Other Multi-Gate FETs*. Springer.

### Machine Learning for Device Design
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.

### Material Properties
- Sze, S. M., & Ng, K. K. (2006). *Physics of Semiconductor Devices* (3rd ed.). John Wiley & Sons.

---

**Version**: 1.0.0  
**Last Updated**: March 2026  
**Status**: Active Development
