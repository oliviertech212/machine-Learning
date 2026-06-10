# Image Classifier — Laptop / Food / Bag

A PyTorch CNN model that classifies images into 3 categories:
laptop, food, and bag.

---

## Folder Structure

```
image_classifier/
│
├── dataset/                  ← Put your images here
│   ├── laptop/               ← All laptop images go here
│   ├── food/                 ← All food images go here
│   └── bag/                  ← All bag images go here
│
├── models/
│   ├── model.py              ← CNN model definition
│   └── image_classifier.pth  ← Saved model (created after training)
│
├── utils/
│   └── data_loader.py        ← Data loading and transforms
│
├── train.py                  ← Run this to train the model
├── predict.py                ← Run this to predict a new image
├── requirements.txt          ← Python packages needed
└── README.md
```

---

## Step by Step Guide

### Step 1 — Install requirements

```bash
pip install -r requirements.txt
```

### Step 2 — Collect your images

- Collect at least 50 images per category (more = better)
- Put them in the correct folders:
  - Laptop images → dataset/laptop/
  - Food images   → dataset/food/
  - Bag images    → dataset/bag/
- Any image format works: .jpg .jpeg .png

### Step 3 — Train the model

```bash
python train.py
```

You will see output like:
```
Epoch 01/20 | Train Loss: 1.0842 | Test Loss: 0.9631 | Accuracy: 45.0%
Epoch 02/20 | Train Loss: 0.8214 | Test Loss: 0.7891 | Accuracy: 62.5%
...
Epoch 20/20 | Train Loss: 0.1823 | Test Loss: 0.2014 | Accuracy: 94.0%
```

A chart will also be saved as training_results.png

### Step 4 — Predict a new image

```bash
python predict.py path/to/your/image.jpg
```

Output example:
```
Image     : my_photo.jpg
Prediction: LAPTOP
Confidence: 96.3%

All probabilities:
  bag       2.1%  █
  food      1.6%
  laptop   96.3%  ████████████████████████████
```

---

## Tips for better accuracy

- Use at least 100 images per category
- Make sure images are clear and well lit
- Mix different angles, backgrounds, and lighting
- The more variety in your images, the stronger the model

---

## How it works (simply)

1. Images are converted to numbers (pixels)
2. CNN layers find patterns — edges, shapes, colours
3. Classifier layers decide which category it is
4. Loss measures how wrong the model is
5. Optimizer fixes the weights to reduce loss
6. Repeat until accuracy is high
