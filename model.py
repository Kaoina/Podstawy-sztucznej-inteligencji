import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from preprocessing import create_data_loaders, ArtStyleDataset
import time
from tqdm import tqdm
from collections import Counter


# # Własna architektura CNN
class ArtStyleCNN(nn.Module):
    def __init__(self, num_classes):
        super(ArtStyleCNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 112x112

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 56x56

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28x28

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # wynik 1x1x512
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x



# Funkcja walidacyjna
def evaluate_model(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0

    print("🔍 Rozpoczynam evaluate_model...")
    print(f"🧪 Liczba batchy w val_loader: {len(val_loader)}")

    with torch.no_grad():
        for batch_idx, (inputs, labels, img_paths) in enumerate(tqdm(val_loader, desc="🧪 Walidacja")):
            try:
                start = time.time()
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                duration = time.time() - start
                if duration > 5.0:
                    print(f"⚠️ Batch {batch_idx} zajął {duration:.2f}s — {img_paths[0]}")
            except Exception as e:
                print(f"\n❌ Błąd w batchu walidacyjnym {batch_idx}: {e}")
                continue

    acc = 100. * correct / total if total > 0 else 0.0
    print(f"🎯 Accuracy: {acc:.2f}%")

    pred_counts = Counter()
    with torch.no_grad():
        for inputs, labels, _ in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            pred_counts.update(predicted.cpu().numpy().tolist())

    print("\n🔢 Rozkład przewidywanych klas na zbiorze walidacyjnym:")

    for class_idx, count in sorted(pred_counts.items()):
        class_name = val_loader.dataset.idx_to_label[class_idx]
        print(f" - {class_name} ({class_idx}): {count} próbek")

    return acc

# Funkcja treningowa
def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10):
    torch.autograd.set_detect_anomaly(True)
    model = model.to(device)

    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        start_time = time.time()

        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", unit="batch")

        for i, (inputs, labels, _) in enumerate(pbar):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            try:
                #print(f"Rozmiar batcha: {inputs.shape}")
                outputs = model(inputs)
                #print(f"Rozmiar outputu: {outputs.shape}")
                #print(f"labels.shape: {labels.shape}, dtype: {labels.dtype}")
                
                loss = criterion(outputs, labels)
                #print(f"Loss: {loss.item()}")  

                loss.backward()
                #print("Backward OK")

                optimizer.step()
                #print("Optimizer step OK")

            except Exception as e:
                print(f"Błąd w batchu: {e}")
                continue

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Dynamiczny update paska tqdm
            avg_loss = running_loss / (i + 1)
            acc = 100. * correct / total
            pbar.set_postfix({"Loss": f"{avg_loss:.4f}", "Train Acc": f"{acc:.2f}%"})

        train_acc = 100. * correct / total
        val_acc = evaluate_model(model, val_loader, device)
        
        torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pt")
        print(f"💾 Zapisano model po epoce {epoch+1}")


        print(f"Epoch time: {time.time() - start_time:.2f}s")
        print(f"Train Loss: {running_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        torch.cuda.empty_cache()
        
        print(f"✔️ Zakończono epokę {epoch+1}")

if __name__ == "__main__":
    print("\nDane cuda:")
    print(torch.__version__)                     # np. 2.6.0+cu118
    print(torch.version.cuda)                    # np. '11.8' jeśli CUDA działa
    print(torch.cuda.is_available())             # True tylko jeśli działa GPU
    print("\n")

    import sys
    print(sys.executable)
    print()


    # ścieżki do katalogów
    input_directory = r"C:\Users\Dominik\Desktop\DataSet"
    output_directory = r"C:\Users\Dominik\Desktop\outData"

    # Parametry
    img_size = 224
    
    batch_size = 32
    min_class_size = 2000
    max_class_size = 3000

    # Ścieżki wyjściowe
    processed_dir = os.path.join(output_directory, "processed")
    split_dir = os.path.join(output_directory, "split")

    data_loaders = create_data_loaders(split_dir, batch_size=batch_size, img_size=img_size)

    # 🔍 Debug: sprawdzenie poprawnej liczby klas i ich mapowania
    print("Liczba klas:", len(data_loaders['train_dataset'].label_to_idx))
    print("Mapowanie klas:", data_loaders['train_dataset'].label_to_idx)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Używane urządzenie: {device}")

    num_classes = len(data_loaders['train_dataset'].label_to_idx)
    model = ArtStyleCNN(num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    try:
        train_model(model, 
                    data_loaders['train_loader'], 
                    data_loaders['val_loader'], 
                    criterion, optimizer, 
                    device, 
                    num_epochs=10)
        print("✅ Trenowanie zakończone poprawnie.")
    except Exception as e:
        print(f"❌ Błąd w train_model: {e}")