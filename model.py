import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from preprocessing import create_data_loaders, ArtStyleDataset
import time
from tqdm import tqdm
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
import seaborn as sns
import pandas as pd
import torchvision.models as models

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

def get_resnet_model(num_classes, pretrained=True):
    model = models.resnet18(pretrained=pretrained)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)  # zmieniamy ostatnią warstwę
    return model


# Funkcja do obliczania i wyświetlania macierzy pomyłek
def plot_confusion_matrix(y_true, y_pred, classes, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    
    # Normalizacja macierzy dla lepszej czytelności
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Używamy seaborn dla ładniejszego wykresu
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    
    plt.xlabel('Przewidywane klasy')
    plt.ylabel('Rzeczywiste klasy')
    plt.title('Znormalizowana macierz pomyłek')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"📊 Zapisano macierz pomyłek do {save_path}")
    
    #plt.show()


# Funkcja do rysowania wykresów strat i dokładności
def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path=None):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Wykres funkcji straty
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Strata treningowa')
    plt.plot(epochs, val_losses, 'r-', label='Strata walidacyjna')
    plt.title('Funkcje straty')
    plt.xlabel('Epoki')
    plt.ylabel('Strata')
    plt.legend()
    
    # Wykres dokładności
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-', label='Dokładność treningowa')
    plt.plot(epochs, val_accs, 'r-', label='Dokładność walidacyjna')
    plt.title('Dokładność modelu')
    plt.xlabel('Epoki')
    plt.ylabel('Dokładność [%]')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"📊 Zapisano wykresy do {save_path}")
    
    #plt.show()


# Rozszerzona funkcja ewaluacyjna z dodatkowymi metrykami
def evaluate_model(model, val_loader, device, calc_metrics=False):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    val_loss = 0
    criterion = nn.CrossEntropyLoss()

    print("🔍 Rozpoczynam evaluate_model...")
    print(f"🧪 Liczba batchy w val_loader: {len(val_loader)}")

    with torch.no_grad():
        for batch_idx, (inputs, labels, img_paths) in enumerate(tqdm(val_loader, desc="🧪 Walidacja")):
            try:
                start = time.time()
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                
                # Obliczanie straty
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Zbieranie predykcji i prawdziwych etykiet dla metryk
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                duration = time.time() - start
                if duration > 5.0:
                    print(f"⚠️ Batch {batch_idx} zajął {duration:.2f}s — {img_paths[0]}")
            except Exception as e:
                print(f"\n❌ Błąd w batchu walidacyjnym {batch_idx}: {e}")
                continue

    acc = 100. * correct / total if total > 0 else 0.0
    avg_val_loss = val_loss / len(val_loader)
    print(f"🎯 Accuracy: {acc:.2f}%")
    print(f"📉 Średnia strata walidacyjna: {avg_val_loss:.4f}")

    # Liczenie dystrybucji predykcji
    pred_counts = Counter(all_preds)
    print("\n🔢 Rozkład przewidywanych klas na zbiorze walidacyjnym:")
    
    for class_idx, count in sorted(pred_counts.items()):
        class_name = val_loader.dataset.idx_to_label[class_idx]
        print(f" - {class_name} ({class_idx}): {count} próbek")
    
    # Obliczanie dodatkowych metryk, jeśli wymagane
    if calc_metrics:
        # Uzyskanie nazw klas
        class_names = [val_loader.dataset.idx_to_label[i] for i in range(len(val_loader.dataset.idx_to_label))]
        
        # Precision, Recall, F1-Score
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted')
        
        # Wyświetlenie metryk
        print("\n📊 Metryki modelu:")
        print(f"🔹 Accuracy: {acc:.2f}%")
        print(f"🔹 Precision (ważona): {precision:.4f}")
        print(f"🔹 Recall (ważony): {recall:.4f}")
        print(f"🔹 F1-Score (ważony): {f1:.4f}")
        
        # Wyświetlenie pełnego raportu klasyfikacji
        print("\n📝 Szczegółowy raport klasyfikacji:")
        print(classification_report(all_labels, all_preds, target_names=class_names))
        
        # Tworzenie macierzy pomyłek
        print("\n🧩 Tworzę macierz pomyłek...")
        plot_confusion_matrix(all_labels, all_preds, class_names, save_path=f"{model_name}_confusion_matrix.png")
        
        # Tworzenie DataFrame z metrykami dla każdej klasy
        metrics_df = pd.DataFrame(
            precision_recall_fscore_support(all_labels, all_preds),
            index=['Precision', 'Recall', 'F1-Score', 'Support'],
            columns=class_names
        ).T
        
        print("\n📋 Metryki dla poszczególnych klas:")
        print(metrics_df.to_string())
        
        return acc, avg_val_loss, precision, recall, f1, metrics_df
    
    return acc, avg_val_loss


# Rozszerzona funkcja treningowa z zapisywaniem historii
def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10, model_name="model"):
    torch.autograd.set_detect_anomaly(True)
    model = model.to(device)
    
    # Listy do przechowywania historii treningu
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    best_val_acc = 0.0

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
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

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

        # Obliczenie średniej straty i dokładności treningowej
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # Zapisanie wartości do historii
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Ewaluacja na zbiorze walidacyjnym
        val_acc, val_loss = evaluate_model(model, val_loader, device)
        
        # Zapisanie wartości walidacyjnych do historii
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Zapisanie modelu
        torch.save(model.state_dict(), f"{model_name}_epoch_{epoch+1}.pt")
        print(f"💾 Zapisano model po epoce {epoch+1}")
        
        # Zapisanie najlepszego modelu
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{model_name}_best_model.pt")
            print(f"🏆 Nowy najlepszy model z dokładnością {val_acc:.2f}%")

        print(f"Epoch time: {time.time() - start_time:.2f}s")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        torch.cuda.empty_cache()
        
        # Rysowanie wykresów dla aktualnego stanu treningu
        if (epoch + 1) % 3 == 0 or (epoch + 1) == num_epochs:  # Co 3 epoki lub po ostatniej
            plot_training_history(train_losses, val_losses, train_accs, val_accs, 
                                  save_path=f"{model_name}_training_history_epoch_{epoch+1}.png")
        
        print(f"✔️ Zakończono epokę {epoch+1}")
    
    # Końcowa ewaluacja z obliczeniem wszystkich metryk
    print("\n🏁 Ewaluacja końcowa modelu...")
    # Ładowanie najlepszego modelu
    model.load_state_dict(torch.load(f"{model_name}_best_model.pt"))
    final_results = evaluate_model(model, val_loader, device, calc_metrics=True)
    
    # Rysowanie finalnych wykresów
    plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path=f"{model_name}_final_training_history.png")
    
    # Zapisanie historii treningu do pliku CSV
    history_df = pd.DataFrame({
        'Epoka': range(1, num_epochs + 1),
        'Strata_treningowa': train_losses,
        'Strata_walidacyjna': val_losses,
        'Dokładność_treningowa': train_accs,
        'Dokładność_walidacyjna': val_accs
    })
    history_df.to_csv(f'{model_name}_training_history.csv', index=False)
    print("📊 Zapisano historię treningu do training_history.csv")
    
    return history_df, final_results


if __name__ == "__main__":
    print("\nDane cuda:")
    print(torch.__version__)                     # np. 2.6.0+cu118
    print(torch.version.cuda)                    # np. '11.8' jeśli CUDA działa
    print(torch.cuda.is_available())             # True tylko jeśli działa GPU
    print("\n")

    # ścieżki do katalogów
    input_directory = r"C:\Users\Dominik\Desktop\DataSet"
    output_directory = r"C:\Users\Dominik\Desktop\outData"

    # Parametry
    img_size = 224
    batch_size = 32
    num_epochs = 10 


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

    models_to_train = {
        #"CustomCNN": ArtStyleCNN(num_classes),
        "ResNet18_pretrained": get_resnet_model(num_classes, pretrained=True),
        "ResNet18_scratch": get_resnet_model(num_classes, pretrained=False)
    }
    
    for model_name, model in models_to_train.items():
        print(f"\n🧠 Trenuję model: {model_name}")
    
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        try:
            history, final_metrics = train_model(model, 
                                                data_loaders['train_loader'], 
                                                data_loaders['val_loader'], 
                                                criterion, optimizer, 
                                                device, 
                                                num_epochs=10,
                                                model_name=model_name
                                                )
            print(f"✅ Trenowanie zakończone dla modelu {model_name}")
            # Wyświetlenie podsumowania końcowego
            print("\n📋 Podsumowanie treningu:")
            print(f"Najlepsza dokładność walidacyjna: {max(history['Dokładność_walidacyjna']):.2f}%")
            print(f"Finalne metryki:")
            print(f" - Accuracy: {final_metrics[0]:.2f}%")
            print(f" - Precision: {final_metrics[2]:.4f}")
            print(f" - Recall: {final_metrics[3]:.4f}")
            print(f" - F1-Score: {final_metrics[4]:.4f}")
        except Exception as e:
            print(f"❌ Błąd w modelu {model_name}: {e}")


