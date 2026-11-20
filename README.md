# Waste Classification Using CNN

## Project Team

**Vikram Rathod**
1st Year, B.Tech Computer Science and Engineering
National Institute of Technology Karnataka (NITK)

**Kinjal Barman**
1st Year, B.Tech Computer Science and Engineering
National Institute of Technology Karnataka (NITK)

---

## 1. Introduction

Waste management is one of the important environmental challenges in today's world. Proper identification and segregation of waste materials can improve recycling and reduce environmental pollution.

In this project, we developed a waste classification system using a **Convolutional Neural Network (CNN)**. The system takes an image of waste as input and predicts its category using a trained deep learning model.

The model classifies waste into six categories:

1. Cardboard
2. Glass
3. Metal
4. Paper
5. Plastic
6. Trash

We also created a web interface separately, through which users can upload waste images and view the predicted category.

## 2. Project Objectives

- To understand and implement image classification using deep learning.
- To develop a CNN model for classifying different types of waste.
- To prepare and preprocess an image dataset.
- To train and evaluate the model using PyTorch.
- To export the trained model into ONNX format.
- To use the trained model for waste image prediction through a web interface.
- To explore how AI can support waste segregation and recycling.

## 3. Dataset Description

We collected our waste image dataset from Kaggle. From the available images, we manually selected the required images and removed poor-quality images to improve the quality of our dataset.

**Dataset source:** [Kaggle Waste Classification Dataset](https://www.kaggle.com/datasets/kinjalbarman/waste-classification-cleaned-dataset)

The intended dataset contains 3,000 images divided into training and validation sets.

| Waste Category | Expected Training Images | Validation Images |
| -------------- | -----------------------: | ----------------: |
| Cardboard      |                      400 |               100 |
| Glass          |                      400 |               100 |
| Metal          |                      400 |               100 |
| Paper          |                      400 |               100 |
| Plastic        |                      400 |               100 |
| Trash          |                      400 |               100 |
| **Total**      |                **2,400** |           **600** |

**Dataset note:** The image counts above describe the intended dataset arrangement. Always run `download_data.py` to check the actual number of images available in your local folders before training. The scripts support JPG, JPEG, JFIF, PNG, BMP and WEBP image formats.

### Dataset Organization

The dataset is organized into separate training and validation folders. Each folder contains six subfolders corresponding to the waste categories.

## 4. Methodology

Our project follows these major steps:

**Step 1: Dataset Collection**

Collected waste images from Kaggle for six different waste categories.

**Step 2: Data Selection and Cleaning**

Manually selected images and removed poor-quality images from the collected dataset.

**Step 3: Data Preprocessing**

Prepared the images for model training by resizing them to 96 × 96 pixels, converting them into tensors, and applying normalization.

Data augmentation techniques were also used during training to improve the model's ability to generalize.

**Step 4: Model Development**

Developed a custom CNN architecture to extract important features from waste images and classify them into six categories.

**Step 5: Model Training**

Trained the CNN model using PyTorch with training and validation datasets.

**Step 6: Model Export**

Exported the trained model into ONNX format for model inference.

**Step 7: Web Interface**

Created a separate web interface where users can upload waste images and view the predicted category.

## 5. CNN Model Architecture

We used a custom lightweight CNN architecture for waste image classification.

The architecture includes:

- **Convolutional Layers:** Extract important features such as edges, shapes, and patterns from images.
- **Batch Normalization:** Helps stabilize and improve the training process.
- **SiLU Activation:** Introduces non-linearity into the network.
- **Depthwise Separable Convolution:** Reduces computational complexity.
- **Squeeze-and-Excitation Blocks:** Helps the model focus on important feature channels.
- **Adaptive Average Pooling:** Converts feature maps into a fixed-size representation.
- **Dropout:** Helps reduce overfitting.
- **Fully Connected Layers:** Perform the final classification.

### Model Input and Output

| Specification      | Details            |
| ------------------ | ------------------ |
| Input image        | RGB image          |
| Image dimensions   | 96 × 96 pixels     |
| Input tensor shape | (Batch, 3, 96, 96) |
| Output shape       | (Batch, 6)         |
| Number of classes  | 6                  |
| Framework          | PyTorch            |
| Export format      | ONNX               |

The model produces six output scores, corresponding to the six waste categories.

## 6. Technologies Used

| Technology   | Purpose                            |
| ------------ | ---------------------------------- |
| Python       | Main programming language          |
| PyTorch      | CNN development and model training |
| Torchvision  | Image loading and preprocessing    |
| NumPy        | Numerical operations               |
| Pillow       | Image processing                   |
| ONNX         | Model export format                |
| ONNX Runtime | Model inference                    |
| HTML         | Webpage structure                  |
| CSS          | Webpage styling                    |
| JavaScript   | Webpage functionality              |

## 7. Project Folder Structure

The main project folder is named `AI-DS-Waste-Classification`.

```text
AI-DS-Waste-Classification/
│
├── README.md
├── requirements.txt
├── download_data.py
├── train_template.py
├── verify_submission.py
├── model.onnx
│
└── data/
    └── public/
        ├── train/
        │   ├── cardboard/
        │   ├── glass/
        │   ├── metal/
        │   ├── paper/
        │   ├── plastic/
        │   └── trash/
        │
        └── val/
            ├── cardboard/
            ├── glass/
            ├── metal/
            ├── paper/
            ├── plastic/
            └── trash/
