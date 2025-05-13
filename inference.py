import torch
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
from models import ArtStyleCNN
from pathlib import Path
import numpy as np
from preprocessing import create_data_loaders
from models import get_resnet_model
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import random
from torchvision.utils import make_grid


# === PARAMETRY ===
model_name = "ResNet18_pretrained"
model_path = f"Models/{model_name}_best_model.pt"
data_dir = r"C:\Users\Dominik\Desktop\outData\split"
output_dir = f"inference_outputs/{model_name}"
os.makedirs(output_dir, exist_ok=True)

img_size = 224
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    # === TWORZENIE DATALOADERA TESTOWEGO ===
    loaders = create_data_loaders(data_dir=data_dir, img_size=img_size)
    test_loader = loaders['test_loader']
    idx_to_label = test_loader.dataset.idx_to_label
    label_to_idx = test_loader.dataset.label_to_idx
    num_classes = len(idx_to_label)

    # === WCZYTYWANIE MODELU ===
    model = get_resnet_model(num_classes, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # === INFERENCJA ===
    all_preds = []
    all_labels = []
    all_paths = []

    with torch.no_grad():
        for inputs, labels, paths in tqdm(test_loader, desc="🔎 Inferencja"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_paths.extend(paths)

    # === MAPOWANIE I ZAPIS WYNIKÓW ===
    results_df = pd.DataFrame({
        "ImagePath": all_paths,
        "TrueLabel": [idx_to_label[i] for i in all_labels],
        "PredictedLabel": [idx_to_label[i] for i in all_preds]
    })
    results_csv_path = os.path.join(output_dir, "inference_results.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"✅ Wyniki zapisane w {results_csv_path}")

    # === RAPORT I MACIERZ POMYŁEK ===
    print("\n📋 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=[idx_to_label[i] for i in range(num_classes)]))

    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=[idx_to_label[i] for i in range(num_classes)],
                yticklabels=[idx_to_label[i] for i in range(num_classes)])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Normalized Confusion Matrix")
    plt.tight_layout()

    confusion_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(confusion_path)
    print(f"📊 Macierz pomyłek zapisana w {confusion_path}")

    # === WIZUALIZACJA PRZYKŁADOWYCH PREDYKCJI ===

    # Wybierz losowo N przykładów
    N = 9
    random_indices = random.sample(range(len(all_paths)), N)

    fig, axs = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle("🔍 Przykładowe predykcje", fontsize=16)

    for i, idx in enumerate(random_indices):
        img_path = all_paths[idx]
        true_label = idx_to_label[all_labels[idx]]
        pred_label = idx_to_label[all_preds[idx]]

        # Wczytaj obraz z dysku
        image = Image.open(img_path).convert("RGB")
        axs[i // 3, i % 3].imshow(image)
        axs[i // 3, i % 3].set_title(f"Prawda: {true_label}\nPred: {pred_label}",
                                    color="green" if true_label == pred_label else "red")
        axs[i // 3, i % 3].axis("off")

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    sample_vis_path = os.path.join(output_dir, "sample_predictions.png")
    plt.savefig(sample_vis_path)
    plt.show()
    print(f"🖼️ Przykładowe predykcje zapisane w {sample_vis_path}")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
