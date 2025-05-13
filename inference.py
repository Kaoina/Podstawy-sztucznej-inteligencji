import torch
from torchvision import transforms
from PIL import Image
import sys
import os
from models import ArtStyleCNN

model_path = "CustomCNN_best_model.pt"  # Ścieżka do  modelu
image_path = "ścieżka/do/obrazu.jpg"    # <-- PODAJ ŚCIEŻKĘ DO OBRAZU

idx_to_label = {
    0: "Academic_Art",
    1: "Art_Nouveau",
    2: "Baroque",
    3: "Expressionism",
    4: "Japanese_Art",
    5: "Neoclassicism",
    6: "Primitivism",
    7: "Realism",
    8: "Renaissance",
    9: "Rococo",
    10: "Romanticism",
    11: "Symbolism",
    12: "Western_Medieval"
}

img_size = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Przygotowanie transformacji obrazu ===
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
])

# === Wczytanie i przygotowanie obrazu ===
try:
    image = Image.open(image_path).convert('RGB')
except Exception as e:
    print(f"❌ Błąd przy wczytywaniu obrazu: {e}")
    sys.exit(1)

input_tensor = transform(image).unsqueeze(0).to(device)

# === Wczytanie modelu ===
num_classes = len(idx_to_label)
model = ArtStyleCNN(num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# === Inferencja ===
with torch.no_grad():
    outputs = model(input_tensor)
    _, predicted = torch.max(outputs, 1)
    predicted_class = idx_to_label[predicted.item()]

print(f"📸 Obraz: {os.path.basename(image_path)}")
print(f"🔍 Przewidywana klasa: {predicted_class}")
