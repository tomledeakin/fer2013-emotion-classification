# FER-2013 Emotion Classification

This project uses a variety of models to classify emotions from facial expression images in the **FER-2013 dataset**. The images are **128x128 pixels** in size.

## Dataset
The **FER-2013** dataset contains grayscale images with 7 emotion classes:

- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

## Model Architectures

### 1. **Multilayer Perceptron (MLP)**

| Layer          | Input Dimensions   | Output Dimensions   | Activation Function |
|----------------|--------------------|---------------------|---------------------|
| Input Layer    | 128x128 (flattened)| 16384               | -                   |
| Hidden Layer 1 | 16384              | 1024                | ReLU                |
| Hidden Layer 2 | 1024               | 512                 | ReLU                |
| Hidden Layer 3 | 512                | 256                 | ReLU                |
| Output Layer   | 256                | 7                   | -                   |

- **Accuracy**: ~44% on the test set with the current MLP model.

### 2. **[Next Model Name]**

| Layer          | Input Dimensions   | Output Dimensions   | Activation Function |
|----------------|--------------------|---------------------|---------------------|
| Input Layer    | [input dims]       | [output dims]       | -                   |
| Hidden Layer 1 | [dims]             | [dims]              | [activation]        |
| Hidden Layer 2 | [dims]             | [dims]              | [activation]        |
| ...            | ...                | ...                 | ...                 |
| Output Layer   | [dims]             | [dims]              | -                   |

### Performance:

- **Model 1 (MLP)**: ~44% accuracy.
- **[Model Name]**: [accuracy]% accuracy (to be updated after implementation).
