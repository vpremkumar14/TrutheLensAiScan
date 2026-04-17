# 🚀 TruthLens AI - Quick Start (5 Steps)

## Step 1️⃣ Generate Dataset (5 min)
```bash
cd backend
python generate_dataset.py
```
✅ Creates fake AI faces + synthetic real faces

## Step 2️⃣ Train Model (20-60 min)
```bash
python train_model_final.py --epochs 50 --batch-size 16 --lr 0.0005
```
✅ Trains with proper generalization fixes

## Step 3️⃣ Start API (Terminal 1)
```bash
python app_updated.py
```
✅ Runs on http://localhost:5000

## Step 4️⃣ Start Frontend (Terminal 2)
```bash
cd frontend
npm run dev
```
✅ Runs on http://localhost:5173

## Step 5️⃣ Test in Browser
- Open http://localhost:5173
- Upload an image
- See prediction with confidence! ✅

---

## What Was Fixed

| Issue | Solution |
|-------|----------|
| ❌ Overfitting | ✅ Added dropout + weight decay |
| ❌ Poor generalization | ✅ Aggressive data augmentation |
| ❌ Class imbalance | ✅ WeightedRandomSampler |
| ❌ Manual tuning | ✅ Adaptive learning rate |
| ❌ Sparse API | ✅ Comprehensive endpoints |
| ❌ No integration | ✅ Full frontend integration |

---

## Files Created

1. **`backend/generate_dataset.py`** - Auto-generate training data
2. **`backend/train_model_final.py`** - Improved trainer
3. **`backend/app_updated.py`** - New API with endpoints
4. **`COMPLETE_SETUP_GUIDE.md`** - Detailed documentation
5. **`IMPLEMENTATION_SUMMARY.md`** - What was done
6. **`verify_setup.py`** - System verification

---

## Expected Results

After training:
- ✅ Validation accuracy: 80-90%
- ✅ Model file: `models/deepfake_detector.pth` (~90MB)
- ✅ Training history: `models/deepfake_detector_history.json`
- ✅ API works: Can detect images with confidence scores

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| "No images found" | Run: `python generate_dataset.py` |
| Low accuracy | Add more diverse data |
| API won't start | Port 5000 busy, use `--port 5001` |
| GPU not working | Check: `python -c "import torch; print(torch.cuda.is_available())"` |
| Frontend can't reach API | Check backend running on port 5000 |

---

## Next Steps

1. ✅ Run the 5 steps above
2. ✅ Test with images
3. ✅ Add more training data (>500 images per class)
4. ✅ Retrain for better accuracy
5. ✅ Deploy to production

---

For detailed info: See `COMPLETE_SETUP_GUIDE.md`
