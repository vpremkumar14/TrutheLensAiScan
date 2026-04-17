"""
Generate Training Dataset for Windows
Downloads diverse real and fake face images
"""

import os
import shutil
import urllib.request
import time
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "data"
REAL_DIR = DATASET_DIR / "real"
FAKE_DIR = DATASET_DIR / "fake"

# Create directories
REAL_DIR.mkdir(parents=True, exist_ok=True)
FAKE_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("🚀 TruthLens AI - Dataset Generator for Windows")
print("=" * 60)

# ============================================================
# 1. GENERATE FAKE FACES using thispersondoesnotexist.com
# ============================================================
def generate_fake_faces(num_images=150):
    """Download AI-generated fake faces"""
    print(f"\n🤖 Generating {num_images} AI-generated fake faces...")
    
    for i in range(num_images):
        try:
            url = "https://thispersondoesnotexist.com/image"
            filename = FAKE_DIR / f"ai_face_{i:04d}.jpg"
            
            if filename.exists():
                print(f"  ⏭️  Skipping {filename.name} (already exists)")
                continue
            
            print(f"  ⬇️  Downloading {i+1}/{num_images}...", end="\r")
            urllib.request.urlretrieve(url, filename)
            time.sleep(0.5)  # Rate limiting
            
        except Exception as e:
            print(f"  ⚠️  Error downloading fake face {i}: {e}")
            continue
    
    print(f"  ✅ Generated {len(list(FAKE_DIR.glob('*.jpg')))} fake faces")

# ============================================================
# 2. CREATE SYNTHETIC REAL FACES
# ============================================================
def create_synthetic_real_faces():
    """Create simple real face images from PIL"""
    print("\n📸 Creating synthetic real face images...")
    
    try:
        from PIL import Image, ImageDraw
        import random
    except ImportError:
        print("  ⚠️  PIL not installed. Skipping synthetic face generation.")
        print("  Install with: pip install Pillow")
        return
    
    for i in range(100):
        filename = REAL_DIR / f"synthetic_real_{i:04d}.jpg"
        
        if filename.exists():
            continue
        
        try:
            # Create random face-like image
            img = Image.new('RGB', (224, 224), color=(200, 150, 100))
            draw = ImageDraw.Draw(img)
            
            # Draw simple face
            draw.ellipse([40, 30, 100, 90], fill=(220, 180, 160))  # Head
            draw.ellipse([55, 45, 70, 60], fill=(50, 50, 50))      # Left eye
            draw.ellipse([85, 45, 100, 60], fill=(50, 50, 50))     # Right eye
            draw.arc([55, 100, 85, 130], 0, 180, fill=(150, 100, 80), width=3)  # Mouth
            
            img.save(filename)
        except Exception as e:
            print(f"  ⚠️  Error creating synthetic face {i}: {e}")
    
    print(f"  ✅ Created {len(list(REAL_DIR.glob('synthetic_*.jpg')))} synthetic real faces")

# ============================================================
# 3. PROVIDE INSTRUCTIONS FOR MANUAL DATA
# ============================================================
def print_manual_data_instructions():
    """Guide user to add manual data"""
    print("\n" + "=" * 60)
    print("📝 MANUAL DATA COLLECTION INSTRUCTIONS")
    print("=" * 60)
    
    print(f"""
For better model accuracy, add your own images:

1️⃣  REAL FACES - Copy to: {REAL_DIR}
   Sources:
   - Your personal photos (selfies, profile pics)
   - Screenshots from videos
   - Photos from friends/colleagues
   - Download from: https://www.celeba-hq.com/ or similar
   - Or search "celebrity faces" on Google Images

2️⃣  FAKE/DEEPFAKE IMAGES - Copy to: {FAKE_DIR}
   Sources:
   - AI-generated faces (already downloaded above)
   - Deepfake videos screenshots
   - Face-swapped images
   - Or search "deepfake" on GitHub/Kaggle

3️⃣  Minimum requirement: 100+ images in each folder
   Recommended: 500+ images in each folder

4️⃣  Supported formats: .jpg, .jpeg, .png

After adding images, run training:
   python backend/train_model_final.py
""")

# ============================================================
# 4. VERIFY DATASET
# ============================================================
def verify_dataset():
    """Check dataset status"""
    real_count = len(list(REAL_DIR.glob("*.jpg"))) + len(list(REAL_DIR.glob("*.png")))
    fake_count = len(list(FAKE_DIR.glob("*.jpg"))) + len(list(FAKE_DIR.glob("*.png")))
    
    print("\n" + "=" * 60)
    print("📊 DATASET STATUS")
    print("=" * 60)
    print(f"✓ Real images: {real_count}")
    print(f"✓ Fake images: {fake_count}")
    print(f"✓ Total: {real_count + fake_count}")
    
    if real_count < 50 or fake_count < 50:
        print("\n⚠️  WARNING: Not enough data!")
        print("   Add more images to data/real/ and data/fake/")
        return False
    elif real_count < 100 or fake_count < 100:
        print("\n⚠️  CAUTION: Limited data (100+ recommended per class)")
        print("   Add more images for better accuracy")
        return True
    else:
        print("\n✅ Dataset looks good! Ready to train.")
        return True

# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    print(f"\n📁 Dataset directory: {DATASET_DIR}")
    
    # Generate fake faces
    generate_fake_faces(150)
    
    # Create synthetic real faces
    create_synthetic_real_faces()
    
    # Print instructions
    print_manual_data_instructions()
    
    # Verify
    is_ready = verify_dataset()
    
    print("\n" + "=" * 60)
    if is_ready:
        print("🎯 Next step: Run training with:")
        print("   python backend/train_model_final.py")
    else:
        print("⚠️  Please add more images to data/ folders first")
    print("=" * 60)
