# Human Action Recognition: A Comparative Study of Pose Estimation and Optical Flow Methods

This repository contains the code and resources for a dissertation research project that investigates and compares different approaches to human action recognition. The project introduces intelligent preprocessing through custom speed and complexity classifiers, then evaluates pose estimation methods (LSTM) versus optical flow methods (ResNet3D), culminating in a weighted ensemble model that combines both approaches.

<img width="777" height="437" alt="Live Stream Action Recognition" src="https://github.com/user-attachments/assets/be5609fc-b4e6-4d4c-b157-37b3103c929f" />


## Research Overview

This project provides a comprehensive evaluation of human action recognition techniques by:

- **Intelligent Data Preprocessing**: Custom speed and complexity classifiers to automatically categorize action classes based on motion characteristics
- **Pose Estimation Approach**: Utilizing LSTM networks with attention mechanisms to analyze skeletal landmark data
- **Optical Flow Approach**: Implementing ResNet3D models to capture motion patterns through optical flow features  
- **Ensemble Method**: Combining both approaches through a weighted ensemble for improved accuracy
- **Live Recognition**: Real-time action recognition application for practical deployment

## Dataset and Auxiliary Files

Due to file size limitations, auxiliary files including datasets, processed features, and trained models are hosted externally:

**[Download Auxiliary Files](https://zenodo.org/records/15127574)**

The auxiliary package includes:
- Complete dataset
- Processed landmark features  
- Processed optical flow features (test set only due to size constraints)
- Pre-trained model weights

## Getting Started

### Prerequisites

Install required dependencies:
```bash
pip install -r requirements.txt
```

### Dataset Preparation

The preprocessing pipeline includes intelligent classification of action data to optimize model training:

#### 1. Speed Classification
Automatically categorizes actions based on motion velocity characteristics:
```bash
# Update input/output directories in speed_classifier.py (lines 180-181)
python3 speed_classifier.py
```

The speed classifier analyzes temporal motion patterns to distinguish between:
- **Fast actions**: High-velocity movements (e.g., running, jumping)
- **Medium actions**: Moderate-velocity movements (e.g., walking, gesturing)
- **Slow actions**: Low-velocity movements (e.g., sitting, standing)

#### 2. Complexity Classification  
Categorizes actions based on spatial and temporal complexity:
```bash
# Update directory parameter in complexity_classifier.py (line 145)
# Note: Input should match output from speed_classifier.py
python3 complexity_classifier.py
```

The complexity classifier evaluates actions across multiple dimensions:
- **Simple actions**: Basic single-joint movements
- **Moderate actions**: Multi-joint coordinated movements
- **Complex actions**: Full-body coordinated sequences

This dual classification approach enables:
- **Targeted feature extraction** optimized for specific action characteristics
- **Improved model training** through homogeneous action groupings
- **Better evaluation metrics** by accounting for inherent action difficulty
- **Enhanced ensemble performance** through specialized model training

### Feature Extraction

#### 3. Pose Feature Extraction
```bash
# Update directories in PoseExtract.py (lines 178-179)
python3 PoseExtract.py
```

#### 4. Optical Flow Feature Extraction
```bash
# Update directories in OpticalFlowExtract.py (lines 163-165)
# Note: Requires landmark extraction output as input
python3 OpticalFlowExtract.py
```

### Model Training and Evaluation

#### 5. LSTM Model (Pose Estimation)
```bash
# Update feature directory path in model.py (line 349)
python3 model.py
```

#### 6. ResNet3D Model (Optical Flow)
```bash
# Update feature directory path in resnetModel.py (line 285)  
python3 resnetModel.py
```

#### 7. Ensemble Model
```bash
# Update model paths in ensembleModel.py (lines 272-273)
# Note: Requires trained LSTM and ResNet3D model paths
python3 ensembleModel.py
```

### Live Recognition Application

#### 8. Real-time Stream
```bash
# Update trained model paths in harStream.py (lines 265, 267)
python3 harStream.py
```

## Model Architecture

### LSTM with Attention Mechanism
- Processes sequential landmark data extracted from pose estimation
- Incorporates attention mechanism for improved temporal understanding
- Focuses on skeletal joint movements and positioning

### ResNet3D for Optical Flow
- Analyzes dense optical flow fields to capture motion patterns
- Utilizes 3D convolutional layers for spatiotemporal feature learning
- Processes motion information at pixel level

### Weighted Ensemble
- Combines predictions from both LSTM and ResNet3D models
- Optimized weighting scheme for improved overall performance
- Leverages complementary strengths of both approaches

## Key Features

- **Intelligent Preprocessing**: Novel speed and complexity classifiers for automatic action categorization
- **Multi-modal Analysis**: Combines skeletal pose data with motion flow information
- **Adaptive Training**: Specialized model training based on action characteristics
- **Comparative Evaluation**: Systematic comparison between pose estimation and optical flow methods
- **Real-time Capability**: Live action recognition for practical applications
- **Modular Design**: Separate components for easy experimentation and modification
- **Comprehensive Pipeline**: End-to-end workflow from intelligent data preprocessing to deployment

## Results and Performance

The comparative evaluation demonstrates the effectiveness of the ensemble approach across different action characteristics:

### Performance Summary

| Model | Simple | Complex | Slow | Medium | Fast | Overall |
|-------|--------|---------|------|--------|------|---------|
| LSTM | 66% | 53% | 73% | 58% | 45% | **59%** |
| ResNet3D | 83% | 70% | 76% | 80% | 73% | **77%** |
| **LSTM + ResNet3D** | **89%** | **80%** | **88%** | **86%** | **80%** | **84%** |

### Key Findings

**Ensemble Superiority**: The weighted ensemble achieved 84% overall accuracy, representing:
- 25-point improvement over LSTM alone (59% → 84%)
- 7-point improvement over ResNet3D alone (77% → 84%)
- Consistent performance gains across all action categories

**Model-Specific Insights**:
- **ResNet3D dominance**: Optical flow consistently outperformed pose estimation across all action types
- **LSTM limitations**: Struggled particularly with fast actions (45%), likely due to difficulty tracking rapid pose changes
- **Complexity impact**: Both individual models showed degraded performance on complex vs. simple actions

**Preprocessing Validation**: The speed and complexity classifications revealed meaningful performance patterns:
- **Speed sensitivity**: Fast actions proved most challenging for pose-based methods
- **Complexity correlation**: Performance decreased with action complexity for individual models
- **Ensemble robustness**: The combined approach maintained strong performance even for complex actions (80%)

**Practical Implications**: 
- 84% accuracy demonstrates real-world viability
- Complementary strengths of pose and optical flow methods validated
- Intelligent preprocessing enables targeted performance analysis

## Configuration

Each script requires directory path updates before execution. Ensure that:

1. Input/output directories exist and are accessible
2. Directory paths are consistent across dependent scripts
3. Trained model paths are correctly specified for ensemble and live recognition
4. All auxiliary files are downloaded and placed in appropriate directories

## References

- **LSTM Attention Mechanism**: [Revolutionizing Time Series Prediction with LSTM with the Attention Mechanism](https://drlee.io/revolutionizing-time-series-prediction-with-lstm-with-the-attention-mechanism-090833a19af9)
- **Optical Flow Implementation**: [OpenCV Optical Flow Tutorial](https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html)

## Academic Context

This work was completed as part of a dissertation research project investigating the effectiveness of different feature extraction and modeling approaches for human action recognition. The comparative analysis provides insights into the strengths and limitations of pose-based versus motion-based recognition methods.

---

**Note**: This repository contains the complete codebase for the dissertation project. For detailed methodology, experimental results, and theoretical analysis, please refer to the full dissertation document.
