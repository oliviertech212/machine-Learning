import torch
import sys
from PIL import Image
from torchvision import transforms
from models.model import ImageClassifier

# ─────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────
MODEL_PATH   = "models/image_classifier.pth"
CLASS_NAMES  = ["bag", "food", "laptop"]  # must match your folder names

# ─────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"

model = ImageClassifier(num_classes=len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

print("Model loaded successfully!")

# ─────────────────────────────────────────
# IMAGE TRANSFORM
# ─────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ─────────────────────────────────────────
# PREDICT FUNCTION
# ─────────────────────────────────────────
def predict(image_path):
    # Load and prepare image
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Make prediction
    with torch.inference_mode():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        predicted_index = torch.argmax(probabilities).item()

    predicted_class = CLASS_NAMES[predicted_index]
    confidence      = probabilities[predicted_index].item() * 100

    print(f"\nImage     : {image_path}")
    print(f"Prediction: {predicted_class.upper()}")
    print(f"Confidence: {confidence:.1f}%")
    print("\nAll probabilities:")
    for name, prob in zip(CLASS_NAMES, probabilities):
        bar = "█" * int(prob.item() * 30)
        print(f"  {name:<8} {prob.item()*100:5.1f}%  {bar}")

    return predicted_class, confidence


# ─────────────────────────────────────────
# RUN
# ─────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py path/to/your/image.jpg")
    else:
        predict(sys.argv[1])
