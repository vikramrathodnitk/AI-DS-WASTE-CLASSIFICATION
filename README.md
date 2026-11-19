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

The final dataset contains **3,000 images** divided into training and validation sets.

| Waste Category | Training Images | Validation Images |
| -------------- | --------------: | ----------------: |
| Cardboard      |             400 |               100 |
| Glass          |             400 |               100 |
| Metal          |             400 |               100 |
| Paper          |             400 |               100 |
| Plastic        |             400 |               100 |
| Trash          |             400 |               100 |
| **Total**      |       **2,400** |           **600** |

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
```

*Note: `model.onnx` is generated after model training and export. The web interface source files are maintained separately and are not included in this project folder.*

## 8. Installation and Execution

Run the following commands from the `AI-DS-Waste-Classification` project directory.

### Step 1: Install Required Libraries

```bash
python -m pip install -r requirements.txt
```

### Step 2: Prepare the Dataset

Place the dataset in the following structure:

```text
data/public/train/
data/public/val/
```

Each folder must contain the six class folders with their respective images.

Run the dataset preparation script:

```bash
python download_data.py
```

### Step 3: Train the CNN Model

Run the training script:

```bash
python train_template.py --data_dir data/public --epochs 10 --batch_size 32 --lr 0.001 --output model.onnx
```

The script trains the CNN model using the training dataset, evaluates it using the validation dataset, and exports the trained model in ONNX format.

### Step 4: Verify the Exported Model

```bash
python verify_submission.py --model model.onnx --val_dir data/public/val
```

The verification script is used to examine the exported model, including its input and output shapes, model size, parameter count, CPU inference latency, and validation accuracy.

## 9. Results and Evaluation

The model is evaluated using the validation dataset.

The following measurements are used to understand its performance:

- **Validation Accuracy:** Measures how many validation images are classified correctly.
- **Model Size:** Shows the storage size of the exported ONNX model.
- **Parameter Count:** Represents the number of learnable parameters in the model.
- **CPU Inference Latency:** Measures the time required to make a prediction on the CPU.

### Performance Results

| Evaluation Metric    | Result |
| -------------------- | ------ |
| Validation Accuracy  |        |
| ONNX Model Size      |        |
| Number of Parameters |        |
| Average CPU Latency  |        |

## 10. Conclusion

This project demonstrates the application of deep learning in waste classification using a Convolutional Neural Network.

By collecting and preparing waste images, developing a CNN model, training it with PyTorch, and exporting it into ONNX format, we created a system capable of classifying waste into six categories.

The project also helped us understand image preprocessing, CNN architecture, model training, evaluation, and the use of AI in practical environmental applications.

## 11. Learning Outcomes

Through this project, we gained practical experience in:

- Understanding image classification and CNNs.
- Collecting, selecting, and organizing image datasets.
- Applying image preprocessing and data augmentation.
- Developing and training a deep learning model using PyTorch.
- Evaluating model performance using validation data.
- Exporting a trained model into ONNX format.
- Understanding model size, parameters, and inference latency.
- Connecting an AI-based classification task with a web interface.
