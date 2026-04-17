"""
Test script to verify TruthLens AI setup
Run this after training to check if everything works
"""

import os
import sys
from pathlib import Path

def check_backend_setup():
    """Check if backend is properly set up"""
    print("\n" + "=" * 60)
    print("🔍 CHECKING BACKEND SETUP")
    print("=" * 60)
    
    backend_dir = Path(__file__).parent / "backend"
    
    checks = {
        "✓ Backend directory exists": backend_dir.exists(),
        "✓ requirements.txt exists": (backend_dir / "requirements.txt").exists(),
        "✓ data/real exists": (backend_dir / "data" / "real").exists(),
        "✓ data/fake exists": (backend_dir / "data" / "fake").exists(),
        "✓ models directory exists": (backend_dir / "models").exists(),
        "✓ train_model_final.py exists": (backend_dir / "train_model_final.py").exists(),
        "✓ generate_dataset.py exists": (backend_dir / "generate_dataset.py").exists(),
        "✓ app_updated.py exists": (backend_dir / "app_updated.py").exists(),
        "✓ utils/model_handler.py exists": (backend_dir / "utils" / "model_handler.py").exists(),
    }
    
    all_ok = True
    for check, passed in checks.items():
        print(f"{check}: {'✅' if passed else '❌'}")
        if not passed:
            all_ok = False
    
    # Check data
    real_images = len(list((backend_dir / "data" / "real").glob("*"))) if (backend_dir / "data" / "real").exists() else 0
    fake_images = len(list((backend_dir / "data" / "fake").glob("*"))) if (backend_dir / "data" / "fake").exists() else 0
    
    print(f"\n📊 Dataset Status:")
    print(f"   Real images: {real_images}")
    print(f"   Fake images: {fake_images}")
    
    if real_images < 50 or fake_images < 50:
        print("   ⚠️  Not enough data! Run: python backend/generate_dataset.py")
        return False
    else:
        print("   ✅ Dataset looks good")
    
    # Check trained model
    model_path = backend_dir / "models" / "deepfake_detector.pth"
    if model_path.exists():
        model_size = model_path.stat().st_size / (1024 * 1024)
        print(f"\n✅ Trained model found ({model_size:.1f} MB)")
    else:
        print(f"\n❌ Trained model not found!")
        print("   Run: python backend/train_model_final.py")
        return False
    
    return all_ok

def check_frontend_setup():
    """Check if frontend is properly set up"""
    print("\n" + "=" * 60)
    print("🔍 CHECKING FRONTEND SETUP")
    print("=" * 60)
    
    frontend_dir = Path(__file__).parent / "frontend"
    
    checks = {
        "✓ Frontend directory exists": frontend_dir.exists(),
        "✓ package.json exists": (frontend_dir / "package.json").exists(),
        "✓ src directory exists": (frontend_dir / "src").exists(),
        "✓ API utility exists": (frontend_dir / "src" / "utils" / "api.js").exists(),
        "✓ vite.config.js exists": (frontend_dir / "vite.config.js").exists(),
    }
    
    all_ok = True
    for check, passed in checks.items():
        print(f"{check}: {'✅' if passed else '❌'}")
        if not passed:
            all_ok = False
    
    # Check node_modules
    if (frontend_dir / "node_modules").exists():
        print(f"\n✅ Dependencies installed")
    else:
        print(f"\n⚠️  Dependencies not installed")
        print("   Run: cd frontend && npm install")
        return False
    
    return all_ok

def check_python_imports():
    """Check if required Python packages are installed"""
    print("\n" + "=" * 60)
    print("🔍 CHECKING PYTHON PACKAGES")
    print("=" * 60)
    
    packages = {
        "torch": "PyTorch",
        "torchvision": "TorchVision",
        "flask": "Flask",
        "flask_cors": "Flask-CORS",
        "PIL": "Pillow",
        "cv2": "OpenCV",
        "numpy": "NumPy",
    }
    
    all_ok = True
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"✓ {name}: ✅")
        except ImportError:
            print(f"✓ {name}: ❌")
            all_ok = False
    
    if not all_ok:
        print("\n⚠️  Missing packages. Run: pip install -r backend/requirements.txt")
    
    return all_ok

def suggest_next_steps():
    """Suggest next steps based on checks"""
    print("\n" + "=" * 60)
    print("📋 NEXT STEPS")
    print("=" * 60)
    
    print("""
1️⃣  GENERATE DATASET (if not done):
    python backend/generate_dataset.py

2️⃣  TRAIN MODEL:
    python backend/train_model_final.py --epochs 50

3️⃣  START BACKEND API:
    python backend/app_updated.py

4️⃣  START FRONTEND (in new terminal):
    cd frontend
    npm run dev

5️⃣  OPEN IN BROWSER:
    http://localhost:5173

6️⃣  TEST DETECTION:
    Upload an image in the UI

For detailed guide, see: COMPLETE_SETUP_GUIDE.md
""")

def main():
    print("\n" + "=" * 60)
    print("🚀 TruthLens AI - Setup Verification")
    print("=" * 60)
    
    backend_ok = check_backend_setup()
    frontend_ok = check_frontend_setup()
    python_ok = check_python_imports()
    
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    print(f"Backend:  {'✅ OK' if backend_ok else '❌ Issues found'}")
    print(f"Frontend: {'✅ OK' if frontend_ok else '❌ Issues found'}")
    print(f"Python:   {'✅ OK' if python_ok else '❌ Issues found'}")
    
    if backend_ok and frontend_ok and python_ok:
        print("\n✅ Everything looks good! You're ready to go.")
    else:
        print("\n⚠️  Some issues found. Please address them above.")
    
    suggest_next_steps()

if __name__ == "__main__":
    main()
