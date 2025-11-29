# PDHD-Net: Prostate Detection with Hierarchical Decoder Network

PDHD-Net is an advanced 3D object detection framework for multi-grade prostate cancer lesion detection and classification, built upon [nnDetection](https://github.com/MIC-DKFZ/nnDetection) with significant architectural and methodological improvements.

## Overview

PDHD-Net extends nnDetection by integrating:
- **Enhanced Encoder-Decoder Architecture**: Channel-aware lightweight fusion (C) + Swin Transformer encoder + Bidirectional Feature Pyramid Network (BiFPN)
- **Frequency Domain Enhancement**: FFT-based feature extraction for improved multi-scale representation
- **Class Imbalance Handling**: Focal loss with dynamic class weighting and balanced sampling strategies
- **Multi-Grade Classification**: Simultaneous detection and Gleason Grade Group (GGG 1-5) classification

## Key Features

### Architectural Improvements

1. **Channel-Aware Lightweight Fusion (C)**
   - Efficient multi-scale feature integration with minimal computational overhead
   - Adaptive channel attention for enhanced feature representation

2. **Swin Transformer Encoder**
   - Hierarchical vision transformer with shifted windows
   - Efficient self-attention mechanism for long-range dependencies
   - Multi-scale feature extraction with linear computational complexity

3. **FFT-based Feature Enhancement**
   - Fast Fourier Transform for frequency domain analysis
   - Captures global patterns and texture information
   - Complements spatial features with frequency characteristics

4. **Bidirectional Feature Pyramid Network (BiFPN)**
   - Weighted bidirectional cross-scale connections
   - Improved feature fusion across different resolution levels

### Class Imbalance Solutions

- **Focal Loss**: Addresses hard example mining and class imbalance
- **Dynamic Class Weighting**: Adaptive weights based on class distribution
- **Stratified Sampling**: Ensures balanced representation during training

## Performance

The FROC analysis on a multi-center dataset (including the public ProstateX-2 and private datasets comprising 80 cases) demonstrates:

### Overall Performance
- **Overall True Positive Recall**: 86.4% (15mm centroid distance criterion)
- Robust lesion localization across different Gleason grades

### Grade-Specific Sensitivity (@ 2.0 FP/patient)
- **GGG=1**: 85.7%
- **GGG=5** (high-risk): 76.2%

### Extended Tolerance (@ 4.0 FP/patient)
- **GGG=2**: 95.8%
- **GGG=3**: 95.2%

These results demonstrate PDHD-Net's capability to detect both index lesions and clinically significant lesions with high sensitivity while maintaining acceptable false positive rates.

## Installation

### Prerequisites
- Linux (tested on Ubuntu 18.04/20.04)
- Python 3.7+
- CUDA 10.2+ (for GPU support)
- PyTorch 1.7+

### Environment Setup

1. **Clone the repository**
```bash
git clone https://github.com/NGYLK/PDHD-Net.git
cd PDHD-Net
```

2. **Create conda environment**
```bash
conda create -n pdhd python=3.9
conda activate pdhd
```

3. **Install PyTorch** (adjust CUDA version as needed)
```bash
pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```

4. **Install nnDetection dependencies**
```bash
pip install -e .
```

5. **Install additional requirements**
```bash
pip install -r requirements.txt
```

6. **Set environment variables**
```bash
export det_data="/path/to/your/data"
export det_models="/path/to/your/models"
```

Add these to your `~/.bashrc` or `~/.zshrc` for persistence.

## Data Preparation

### Dataset Structure

PDHD-Net follows the nnDetection data format:

```
Task022_Prostate/
├── dataset.json
└── raw_splitted/
    ├── imagesTr/
    │   ├── case001_0000.nii.gz  # T2W
    │   ├── case001_0001.nii.gz  # ADC
    │   ├── case001_0002.nii.gz  # DWI
    │   └── ...
    ├── imagesTs/
    ├── labelsTr/
    │   ├── case001.json
    │   ├── case001.nii.gz
    │   └── ...
    └── labelsTs/
```

### Dataset Configuration

Create a `dataset.json` file:

```json
{
  "task": "Task022_Prostate",
  "name": "Prostate Cancer Detection",
  "dim": 3,
  "target_class": 4,
  "test_labels": true,
  "labels": {
    "0": "GGG1",
    "1": "GGG2",
    "2": "GGG3",
    "3": "GGG4",
    "4": "GGG5"
  },
  "modalities": {
    "0": "T2W",
    "1": "ADC",
    "2": "DWI"
  }
}
```

### Preprocessing

```bash
nndet_prep Task022_Prostate
```

## Training

### Single GPU Training
```bash
nndet_train 022 RetinaUNetV001_D3V001_3d 0 --fold 0
```

### Multi-GPU Training
```bash
nndet_train 022 RetinaUNetV001_D3V001_3d 0 1 2 3 --fold 0
```

### Training Configuration

Key hyperparameters in `nndet/conf/train/v001.yaml`:
- **Batch size**: 2 per GPU
- **Learning rate**: 1e-4 with cosine annealing
- **Focal loss**: α=0.25, γ=2.0
- **Class weights**: Dynamic based on inverse class frequency
- **Training epochs**: 1000 (with early stopping)

## Inference

### Consolidation (Ensemble)
```bash
nndet_consolidate Task022_Prostate RetinaUNetV001_D3V001_3d --fold 0 1 2 3 4
```

### Prediction on Test Set
```bash
nndet_predict Task022_Prostate RetinaUNetV001_D3V001_3d --fold consolidated
```

## Evaluation

### FROC Analysis
```bash
nndet_eval Task022_Prostate RetinaUNetV001_D3V001_3d --fold consolidated
```

The evaluation computes:
- FROC curves at multiple false positive rates
- Sensitivity at 1.0, 2.0, 4.0 FP/patient
- Grade-specific performance metrics
- Centroid distance-based matching (15mm threshold)

## Model Architecture

### Network Overview

```
Input (T2W + ADC + DWI)
    ↓
Swin Transformer Encoder (with C fusion)
    ↓
FFT Feature Enhancement
    ↓
BiFPN Decoder (Multi-scale feature fusion)
    ↓
Detection Head
    ├── Classification (Focal Loss)
    └── Regression (Smooth L1 Loss)
```

### Key Components

- **Backbone**: Swin Transformer with channel-aware fusion and FFT enhancement
- **Neck**: BiFPN for bidirectional multi-scale feature aggregation
- **Head**: Anchor-based detection with class-specific regression

## Citation

If you use PDHD-Net in your research, please cite:

```bibtex
@article{pdhd-net2025,
  title={PDHD-Net: Prostate Detection with Hierarchical Decoder Network for Multi-Grade Lesion Classification},
  author={Your Name},
  journal={Your Journal},
  year={2025}
}
```

Also cite the original nnDetection:

```bibtex
@article{baumgartner2021nndetection,
  title={nnDetection: A self-configuring method for medical object detection},
  author={Baumgartner, Michael and Jaeger, Paul F and Isensee, Fabian and Maier-Hein, Klaus H},
  journal={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  year={2021}
}
```

## Acknowledgments

This project is built upon [nnDetection](https://github.com/MIC-DKFZ/nnDetection). We thank the original authors for their excellent work and open-source contribution.

## License

This project inherits the Apache 2.0 license from nnDetection. See [LICENSE](LICENSE) for details.

## Contact

For questions and issues, please open an issue on GitHub or contact [your-email@example.com].

---

**Note**: This is a research project. The model is intended for research purposes only and has not been approved for clinical use.
