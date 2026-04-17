# Deepfake Detection Model - Overfitting Analysis & Solutions

## Problem Summary
Your model works well on personal images but fails on other real images because:
- **Overfitting**: Learned specific features of your personal images, not general deepfake indicators
- **Poor Generalization**: Can't handle variations in camera, lighting, image quality, etc.
- **Limited Training Data**: Too few samples with insufficient diversity

---

## Root Cause Analysis

### 1. **Overfitting to Personal Images** (Main Issue)
Your model isn't learning "What is a deepfake?" but rather "What does a personal image look like?"

**Evidence:**
- Works perfectly on training data (personal images)
- Fails on unfamiliar real images (classifies as fake)
- This is classic overfitting pattern

### 2. **Insufficient Data Augmentation**
Your current augmentation only does:
- Random horizontal flip
- Random rotation (±10°)

**This is too weak!** The model still sees mostly the same patterns.

### 3. **Weak Regularization**
- No dropout layers
- No weight decay
- Model can memorize instead of generalize

### 4. **Limited Layer Unfreezing**
- Frozen layers 1-2 of ResNet50
- Leaves model too constrained for your specific task

---

## Solutions Implemented in `train_model_v2.py`

### ✅ Solution 1: Aggressive Data Augmentation
```python
- Random flip/vertical flip
- Rotation (±15°)
- Affine transformations (translation)
- ColorJitter (brightness, contrast, saturation, hue)
- Gaussian blur (simulates different camera quality)
- Random perspective (simulates different angles)
```
**Why?** Forces model to learn robustness, not memorize patterns

### ✅ Solution 2: Dropout Regularization
```python
# Before: Just final linear layer
self.resnet.fc = nn.Linear(num_features, num_classes)

# After: With dropout
self.resnet.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_features, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, num_classes)
)
```
**Why?** Prevents neural network from relying on specific features

### ✅ Solution 3: Class Balancing
```python
sampler = WeightedRandomSampler(sample_weights, ...)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
```
**Why?** If you have more real/fake images, the model gets biased

### ✅ Solution 4: Better Hyperparameters
| Parameter | Old | New | Reason |
|-----------|-----|-----|--------|
| Batch Size | 32 | 16 | Smaller batches = more updates per epoch |
| Learning Rate | 0.001 | 0.0005 | Slower, more stable learning |
| Epochs | 50 | 100 | More training with early stopping |
| Scheduler | StepLR | ReduceLROnPlateau | Adaptive learning rate |
| Weight Decay | None | 0.0001 | L2 regularization |

### ✅ Solution 5: Unfroze More Layers
```python
# Old: Froze layers 1-2
for param in self.resnet.layer1.parameters():
    param.requires_grad = False
for param in self.resnet.layer2.parameters():
    param.requires_grad = False

# New: Only freeze layer1
for param in self.resnet.layer1.parameters():
    param.requires_grad = False
```
**Why?** Allows better adaptation to your specific task

---

## How to Use

### Step 1: Organize Your Data
```
data/
├── real/
│   ├── real_image1.jpg
│   ├── real_image2.jpg
│   └── ... (add diverse real images from different sources)
└── fake/
    ├── fake_image1.jpg
    ├── fake_image2.jpg
    └── ... (add diverse fake images)
```

### Step 2: Important - Diversify Your Training Data
To fix the generalization issue, you **MUST** include:
- Real images from different cameras/phones
- Different lighting conditions (bright, dim, indoor, outdoor)
- Different ethnicities and ages
- Different backgrounds
- Different image qualities (compression artifacts, etc.)

**Download datasets:**
- FaceForensics++: https://github.com/ondyari/FaceForensics
- Deepfake Detection Challenge: https://kaggle.com/c/deepfake-detection-challenge
- DFDC: https://github.com/deepfacelab/

### Step 3: Train the Improved Model
```bash
python backend/train_model_v2.py \
    --epochs 100 \
    --batch-size 16 \
    --lr 0.0005 \
    --data-dir data \
    --model-path models/deepfake_detector_v2.pth
```

### Step 4: Monitor Training
- Watch for **validation accuracy** (not training accuracy)
- If training accuracy >> validation accuracy = still overfitting
- Look for this pattern:
  - Training loss: continuously decreasing
  - Validation loss: decreases then plateaus
  - Validation accuracy: increases then plateaus

### Step 5: Test on Diverse Images
Create a test folder with images from different sources and evaluate:
```python
# Test on diverse images
for test_image in diverse_test_images:
    prediction = predict_image(test_image)
    print(f"{test_image}: {prediction}")
```

---

## Expected Improvements
With `train_model_v2.py`:
- ✅ Better generalization to unfamiliar images
- ✅ More robust to different lighting/cameras
- ✅ Lower false positive rate (fewer "fake" predictions on real images)
- ✅ More stable predictions

---

## Additional Recommendations

### 1. **Add Confidence Threshold**
Don't rely on binary classification. Use confidence scores:
```python
probabilities = torch.nn.functional.softmax(outputs, dim=1)
confidence = probabilities.max().item()

if confidence < 0.7:
    result = "Uncertain - Need manual review"
elif predicted_class == 1:
    result = f"Fake (Confidence: {confidence:.2%})"
else:
    result = f"Real (Confidence: {confidence:.2%})"
```

### 2. **Ensemble Multiple Models**
Train multiple models on different data splits:
```python
predictions = [model1.predict(image), model2.predict(image), model3.predict(image)]
final_prediction = majority_vote(predictions)
```

### 3. **Use More Advanced Architectures**
- Vision Transformer (ViT) - better generalization
- EfficientNet - more parameter efficient
- Multi-modal models (combine image + metadata)

### 4. **Regular Model Evaluation**
Test on completely unseen datasets regularly:
- FaceForensics test set
- Real images from internet
- Images from different devices

---

## Next Steps
1. ✅ Run `train_model_v2.py` with your current data
2. ✅ Collect more diverse real and fake images
3. ✅ Retrain with diverse dataset
4. ✅ Test on images from different sources
5. ✅ Implement confidence threshold in API

---

## Files Created
- `backend/train_model_v2.py` - Improved training script with all fixes
- `models/deepfake_detector_v2.pth` - Will be created after training
- `models/deepfake_detector_v2_history.json` - Training history
