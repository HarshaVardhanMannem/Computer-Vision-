# Computer Vision Portfolio

A hands-on computer vision portfolio showing how I solve image problems from simple baselines to modern deep learning systems.

## 🚀 Highlights

- **Built complete CV pipelines** from data prep to evaluation across multiple datasets.
- **Compared classical and deep learning approaches** (KNN, MLP, DenseNet, VGG16, ResNet50, YOLO, ViT).
- **Implemented transfer learning and representation learning** (DenseNet fine-tuning + convolutional autoencoders).
- **Worked across core vision tasks**: image classification, reconstruction, and object detection.
- **Used real experiment workflows** with train/validation/test splits, metric tracking, and model comparison.

## 📁 Project Work

### `EE623_Assignment_01.ipynb` — Foundations + Baselines
- Distance-based analysis (L1/L2) on MNIST and CIFAR-10.
- KNN classification with runtime/accuracy trade-offs.
- MLP experiments with hyperparameter tuning.
- Bonus benchmark: KNN vs MLP on Fashion-MNIST.

### `EE623_Assignment_02.ipynb` — Transfer Learning in Practice
- Built a **Mini CIFAR-10** dataset (150 images/class).
- Created train/validation/test splits.
- Trained and evaluated **DenseNet121**.
- Visualized learning curves and validation behavior.

### `EE623_Assignment_03.ipynb` — Autoencoders + Feature Learning
- Designed convolutional autoencoders on CIFAR-10.
- Trained models for image reconstruction quality.
- Tested architecture/training variations.

### `Final_Project_CV.ipynb` — End-to-End CV Showcase
- Baseline image classification with **KNN**.
- Deep classifiers with **VGG16** and **ResNet50**.
- Object detection inference using a **YOLO** pipeline.
- Vision Transformer implementation and model comparison.

## 🧠 Key Technical Strengths Demonstrated

- Model benchmarking and comparative analysis.
- Transfer learning with pretrained backbones.
- Deep network training and evaluation workflows.
- Representation learning with encoder-decoder architectures.
- Practical notebook-based experimentation and result reporting.

## 🛠️ Run Locally

```bash
pip install tensorflow keras numpy scikit-learn matplotlib seaborn opencv-python notebook
jupyter notebook
```

Then open any notebook and run cells top-to-bottom.

## ✅ Recommended Walkthrough Order

1. `EE623_Assignment_01.ipynb`
2. `EE623_Assignment_02.ipynb`
3. `EE623_Assignment_03.ipynb`
4. `Final_Project_CV.ipynb`

## Notes

- First run may download datasets/pretrained weights.
- Runtime depends on hardware (GPU strongly recommended for deep models).
