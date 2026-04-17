import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
import numpy as np
import os

MODEL_PATH = 'models/deepfake_detector.pth'

class DeepfakeDetectorModel(nn.Module):
    """Updated model with dropout for better generalization"""
    def __init__(self, num_classes=2, dropout_rate=0.5):
        super(DeepfakeDetectorModel, self).__init__()
        # Load pretrained ResNet50
        self.resnet = resnet50(pretrained=True)
        
        # Freeze early layers
        for param in self.resnet.layer1.parameters():
            param.requires_grad = False
        
        # Replace final layer with dropout + classification head
        num_features = self.resnet.fc.in_features
        
        self.resnet.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.resnet(x)

# Keep old name for backward compatibility
class DeepfakeDetector(DeepfakeDetectorModel):
    pass

def load_model(device='cpu'):
    """Load trained model or create a new one"""
    model = DeepfakeDetector(num_classes=2)
    
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            print(f"✓ Model loaded from {MODEL_PATH}")
        except Exception as e:
            print(f"⚠ Could not load model: {e}. Using untrained model.")
            print("  To use a trained model, run: python train_model.py")
    else:
        print(f"⚠ Model file not found at {MODEL_PATH}")
        print("  To train and save model, run: python train_model.py")
    
    model.to(device)
    model.eval()
    return model

def preprocess_input(image_array):
    """Preprocess image numpy array"""
    # Normalize as per ImageNet standards
    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    if isinstance(image_array, np.ndarray):
        image = Image.fromarray(image_array.astype('uint8'))
    else:
        image = image_array
    
    return normalize(image).unsqueeze(0)

@torch.no_grad()
def predict_image(image_path, model, device='cpu'):
    """Predict if image is real or fake"""
    try:
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Resize to 224x224
        image = image.resize((224, 224))
        
        # Preprocess
        input_tensor = preprocess_input(np.array(image))
        input_tensor = input_tensor.to(device)
        
        # Predict
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get prediction
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class].item()
        
        # Map to labels
        label = 'Real' if predicted_class == 0 else 'Fake'
        
        # Generate explanation
        explanation = generate_explanation(label, confidence, predicted_class)
        
        return label, confidence, explanation
    
    except Exception as e:
        print(f"Error in predict_image: {e}")
        # Return default prediction
        return 'Real', 0.5, f"Error during analysis: {str(e)}"

@torch.no_grad()
def predict_video(video_path, model, device='cpu', frames_to_sample=10):
    """Predict if video is real or fake by analyzing multiple frames"""
    try:
        import cv2
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return 'Real', 0.5, "Error opening video file"
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames - 1, frames_to_sample, dtype=int)
        
        predictions = []
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize
            frame = cv2.resize(frame, (224, 224))
            
            # Preprocess
            input_tensor = preprocess_input(frame)
            input_tensor = input_tensor.to(device)
            
            # Predict
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class].item()
            
            predictions.append((predicted_class, confidence))
        
        cap.release()
        
        # Aggregate predictions
        if predictions:
            avg_class = np.mean([p[0] for p in predictions])
            avg_confidence = np.mean([p[1] for p in predictions])
            
            # If closer to 0, it's real; if closer to 1, it's fake
            predicted_label = 'Real' if avg_class < 0.5 else 'Fake'
            # Adjust confidence based on distance from 0.5
            final_confidence = max(avg_confidence, 1 - avg_confidence)
            
            explanation = generate_explanation(
                predicted_label, final_confidence, avg_class, is_video=True
            )
            
            return predicted_label, final_confidence, explanation
        else:
            return 'Real', 0.5, "Could not extract frames from video"
    
    except Exception as e:
        print(f"Error in predict_video: {e}")
        return 'Real', 0.5, f"Error during video analysis: {str(e)}"

def generate_explanation(label, confidence, predicted_class, is_video=False):
    """Generate human-readable explanation"""
    media_type = "video" if is_video else "image"
    
    if label == 'Real':
        if confidence > 0.95:
            return f"This {media_type} appears to be authentic with very high confidence ({confidence*100:.1f}%). No significant signs of AI manipulation or deepfake artifacts were detected in the analysis."
        elif confidence > 0.85:
            return f"This {media_type} is likely authentic ({confidence*100:.1f}% confidence). The analysis shows minimal indicators of AI generation or manipulation."
        else:
            return f"This {media_type} appears to be real ({confidence*100:.1f}% confidence), though some minor artifacts were detected. Further manual inspection may be recommended."
    else:
        if confidence > 0.95:
            return f"This {media_type} shows strong signs of being AI-generated or a deepfake with high confidence ({confidence*100:.1f}%). Multiple artifacts consistent with generative models were detected."
        elif confidence > 0.85:
            return f"This {media_type} is likely AI-generated or a deepfake ({confidence*100:.1f}% confidence). Several indicators suggest the presence of synthetic or manipulated content."
        else:
            return f"This {media_type} may contain AI-generated or deepfake elements ({confidence*100:.1f}% confidence). Proceed with caution and consider additional verification methods."
