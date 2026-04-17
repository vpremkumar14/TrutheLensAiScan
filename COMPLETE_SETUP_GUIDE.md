# 🚀 TruthLens AI - Complete Training & Setup Guide

## Quick Start (5 Minutes)

### Step 1: Generate Dataset
```bash
cd backend
python generate_dataset.py
```

This will:
- Download 150 AI-generated fake faces
- Create 100 synthetic real faces
- Guide you on adding more images

### Step 2: Train Model
```bash
python train_model_final.py --epochs 50 --batch-size 16 --lr 0.0005
```

Expected output:
```
🔥 Epoch 5/50
   Train Loss: 0.2345 | Train Acc: 87.23%
   Val Loss: 0.2856 | Val Acc: 84.91%
   ✅ Best model saved! (Val Acc: 84.91%)
```

### Step 3: Start Backend API
```bash
python app_updated.py
```

Should see:
```
🖥️  Using device: cuda  (or cpu)
✅ Model loaded successfully
🚀 TruthLens AI - Deepfake Detection API
```

### Step 4: Start Frontend
```bash
cd ../frontend
npm install
npm run dev
```

Open `http://localhost:5173` in your browser

---

## Detailed Setup Guide

### Prerequisites
- Python 3.9+
- Node.js 16+
- 10GB free space (for dataset)
- 4GB RAM minimum (8GB+ recommended)

### Backend Setup

#### 1. Install Python Dependencies
```bash
cd backend
pip install -r requirements.txt
```

If `requirements.txt` is missing, install manually:
```bash
pip install torch torchvision torchaudio
pip install flask flask-cors
pip install pillow opencv-python
pip install numpy scikit-learn
```

#### 2. Generate Training Dataset
```bash
python generate_dataset.py
```

**Output:**
```
📁 Dataset directory: /path/to/backend/data

🤖 Generating 150 AI-generated fake faces...
📸 Creating synthetic real face images...

📊 DATASET STATUS
Real images: 100
Fake images: 250
Total: 350

⚠️  CAUTION: Limited data (100+ recommended per class)
   Add more images for better accuracy
```

**To add more data manually:**
1. Copy real face images to `backend/data/real/`
2. Copy deepfake/fake images to `backend/data/fake/`
3. Supported formats: `.jpg`, `.jpeg`, `.png`

**Recommended sources:**
- Real faces: CelebA-HQ, personal photos, public portraits
- Fake faces: FaceForensics++, Kaggle deepfake datasets
- AI generated: thispersondoesnotexist.com (already automated)

#### 3. Train the Model
```bash
python train_model_final.py \
  --epochs 50 \
  --batch-size 16 \
  --lr 0.0005 \
  --data-dir data \
  --model-path models/deepfake_detector.pth
```

**Parameters:**
| Parameter | Default | Notes |
|-----------|---------|-------|
| `--epochs` | 50 | Higher = better (but longer) |
| `--batch-size` | 16 | Smaller = more stable but slower |
| `--lr` | 0.0005 | Learning rate (smaller is safer) |
| `--data-dir` | data | Path to real/fake folders |
| `--model-path` | models/deepfake_detector.pth | Where to save trained model |

**Expected training time:**
- GPU (CUDA): ~10-20 minutes
- CPU: ~1-2 hours

**Monitor training:**
The script will print updates every 5 epochs:
```
Epoch 1/50
   Train Loss: 0.6934 | Train Acc: 50.23%
   Val Loss: 0.6891 | Val Acc: 52.15%

Epoch 5/50
   Train Loss: 0.2451 | Train Acc: 87.60%
   Val Loss: 0.2789 | Val Acc: 85.23%
   ✅ Best model saved! (Val Acc: 85.23%)
```

**Good training signs:**
- ✅ Train and validation accuracy increase together
- ✅ Gap between train/val < 10%
- ✅ No overfitting (validation doesn't decrease while training increases)

**Bad training signs:**
- ❌ Training accuracy > 95%, Validation accuracy < 75% (overfitting)
- ❌ Validation accuracy plateaus early (not enough data or weak augmentation)
- ❌ Both accuracies stay around 50% (model can't learn)

**If training fails:**

1. **"No images found"**
   - Check that `backend/data/real/` and `backend/data/fake/` exist
   - Add images using `python generate_dataset.py`

2. **"CUDA out of memory"**
   - Reduce batch size: `--batch-size 8`
   - Use CPU: Add `import os; os.environ['CUDA_LAUNCH_BLOCKING'] = '1'`

3. **"Model validation accuracy very low"**
   - Add more diverse data
   - Increase epochs: `--epochs 100`
   - Run `generate_dataset.py` again to download more data

#### 4. Test the Trained Model
```bash
python test_model.py
```

#### 5. Start Flask API Server
```bash
python app_updated.py
```

Expected output:
```
============================================================
🚀 TruthLens AI - Deepfake Detection API
============================================================
🖥️  Using device: cuda
✅ Model loaded successfully
 * Running on http://0.0.0.0:5000
============================================================
```

**API Endpoints:**
- `POST /api/detect/image` - Analyze image
- `POST /api/detect/video` - Analyze video
- `POST /api/detect/url` - Analyze from URL
- `GET /api/health` - Health check
- `GET /api/info` - API information

### Frontend Setup

#### 1. Install Dependencies
```bash
cd ../frontend
npm install
```

#### 2. Configure API URL (Optional)
Create `.env.local`:
```
REACT_APP_API_URL=http://localhost:5000/api
```

#### 3. Start Development Server
```bash
npm run dev
```

Open `http://localhost:5173`

#### 4. Build for Production
```bash
npm run build
```

Output in `dist/` folder

---

## Testing the Full Pipeline

### Test 1: Quick Image Detection
```bash
# Use curl to test API
curl -X POST \
  -F "file=@test_image.jpg" \
  http://localhost:5000/api/detect/image
```

Response:
```json
{
  "success": true,
  "prediction": "Real",
  "confidence": 0.87,
  "explanation": "This image appears to be authentic...",
  "is_deepfake": false
}
```

### Test 2: Frontend UI
1. Open `http://localhost:5173`
2. Go to "Image Detection"
3. Upload a test image
4. Should see result with confidence score

### Test 3: API Health
```bash
curl http://localhost:5000/api/health
```

Response:
```json
{
  "status": "healthy",
  "device": "cuda",
  "model_ready": true
}
```

---

## Troubleshooting

### "Module not found" errors
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

### "Port 5000 already in use"
```bash
# Use different port
python app_updated.py --port 5001
```

### CORS errors in frontend
- Check backend is running on `http://localhost:5000`
- Check `.env.local` has correct API URL
- Try in incognito mode (no cache)

### Low accuracy predictions
1. **Check dataset diversity:**
   ```bash
   python generate_dataset.py
   ```

2. **Retrain with more data:**
   ```bash
   python train_model_final.py --epochs 100 --batch-size 16
   ```

3. **Test on different image types:**
   - Different cameras
   - Different lighting
   - Different resolutions

### GPU not detected
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If `False`, install CUDA:
- Windows: Download NVIDIA CUDA Toolkit
- Or use CPU (slower but works)

---

## File Structure After Setup

```
TruthLensAiScan/
├── backend/
│   ├── data/
│   │   ├── real/        (100+ real face images)
│   │   └── fake/        (100+ fake face images)
│   ├── models/
│   │   ├── deepfake_detector.pth         (trained model)
│   │   └── deepfake_detector_history.json (training history)
│   ├── uploads/         (temporary files)
│   ├── utils/
│   │   ├── model_handler.py
│   │   └── preprocessing.py
│   ├── generate_dataset.py
│   ├── train_model_final.py
│   ├── app_updated.py
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   ├── components/
│   │   └── utils/api.js
│   ├── package.json
│   └── .env.local
└── README.md
```

---

## Performance Optimization

### For Better Accuracy
1. **More data** (most important)
   - 500+ images per class
   - Diverse sources

2. **Better augmentation** (already enabled)
   - Color jitter
   - Geometric transforms
   - Blur effects

3. **Longer training**
   ```bash
   python train_model_final.py --epochs 100 --batch-size 8
   ```

### For Faster Inference
1. **Use GPU**
   - Install CUDA
   - Model automatically uses GPU if available

2. **Reduce image size**
   - Already optimized at 224×224
   - Can reduce to 112×112 for speed

3. **Batch processing**
   - Send multiple images at once

---

## Next Steps

1. ✅ Run `generate_dataset.py`
2. ✅ Run `train_model_final.py`
3. ✅ Start `app_updated.py`
4. ✅ Open frontend at `http://localhost:5173`
5. ✅ Test with sample images
6. ✅ Improve accuracy by adding more diverse data
7. ✅ Deploy to production (see DOCKER.md)

---

## Support & Debugging

**Check model status:**
```bash
python -c "from backend.utils.model_handler import load_model; m = load_model(); print('✅ Model OK')"
```

**Check dataset:**
```bash
python backend/generate_dataset.py
```

**View training history:**
```bash
# After training, check:
models/deepfake_detector_history.json
```

**Verify API:**
```bash
curl http://localhost:5000/api/info
```

---

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| Model accuracy low | Add more diverse training data |
| Training very slow | Use GPU (CUDA) or reduce batch size |
| API won't start | Check port 5000 is free, reinstall Flask |
| Images detected as fake incorrectly | Retrain with better data, increase epochs |
| Frontend can't reach API | Check backend running, verify .env.local |

---

For more details, see:
- [Backend README](./backend/README.md)
- [Frontend README](./frontend/README.md)
- [Docker Setup](./DOCKER.md)
