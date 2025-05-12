import torch
import torch.nn as nn
from tqdm import tqdm
from collections import Counter
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
import seaborn as sns
import pandas as pd
import os

# Funkcja do obliczania i wyświetlania macierzy pomyłek
def plot_confusion_matrix(y_true, y_pred, classes, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    
    # Normalizacja (lepsza czytelność)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
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

def evaluate_model(model, val_loader, device, model_name, calc_metrics=False):
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
                
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
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

    pred_counts = Counter(all_preds)
    print("\n🔢 Rozkład przewidywanych klas na zbiorze walidacyjnym:")
    
    for class_idx, count in sorted(pred_counts.items()):
        class_name = val_loader.dataset.idx_to_label[class_idx]
        print(f" - {class_name} ({class_idx}): {count} próbek")
    
    if calc_metrics:
        class_names = [val_loader.dataset.idx_to_label[i] for i in range(len(val_loader.dataset.idx_to_label))]
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted')
        
        print("\n📊 Metryki modelu:")
        print(f"🔹 Accuracy: {acc:.2f}%")
        print(f"🔹 Precision (ważona): {precision:.4f}")
        print(f"🔹 Recall (ważony): {recall:.4f}")
        print(f"🔹 F1-Score (ważony): {f1:.4f}")
        
        print("\n📝 Szczegółowy raport klasyfikacji:")
        print(classification_report(all_labels, all_preds, target_names=class_names))
        
        print("\n🧩 Tworzę macierz pomyłek...")
        plot_confusion_matrix(all_labels, all_preds, class_names, save_path=f"{model_name}_confusion_matrix.png")
        
        metrics_df = pd.DataFrame(
            precision_recall_fscore_support(all_labels, all_preds),
            index=['Precision', 'Recall', 'F1-Score', 'Support'],
            columns=class_names
        ).T
        
        print("\n📋 Metryki dla poszczególnych klas:")
        print(metrics_df.to_string())
        
        return acc, avg_val_loss, precision, recall, f1, metrics_df
    
    return acc, avg_val_loss

def train_model(model, train_loader, val_loader, criterion, optimizer, device, model_name, num_epochs=10):
    torch.autograd.set_detect_anomaly(True)
    model = model.to(device)
    
    plot_dir = os.path.join("training_plots", model_name)
    os.makedirs(plot_dir, exist_ok=True)

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

            avg_loss = running_loss / (i + 1)
            acc = 100. * correct / total
            pbar.set_postfix({"Loss": f"{avg_loss:.4f}", "Train Acc": f"{acc:.2f}%"})

        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        val_acc, val_loss = evaluate_model(model, val_loader, device, model_name)
        
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # torch.save(model.state_dict(), f"{model_name}_epoch_{epoch+1}.pt")
        # print(f"💾 Zapisano model po epoce {epoch+1}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{model_name}_best_model.pt")
            print(f"🏆 Nowy najlepszy model z dokładnością {val_acc:.2f}%")

        print(f"Epoch time: {time.time() - start_time:.2f}s")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        torch.cuda.empty_cache()
        
        if (epoch + 1) % 3 == 0 or (epoch + 1) == num_epochs:
            plot_training_history(
                train_losses, val_losses, train_accs, val_accs,
                save_path=os.path.join(plot_dir, f"training_history_epoch_{epoch+1}.png")
            )

        
        print(f"✔️ Zakończono epokę {epoch+1}")
    
    print("\n🏁 Ewaluacja końcowa modelu...")
    model.load_state_dict(torch.load(f"{model_name}_best_model.pt"))
    final_results = evaluate_model(model, val_loader, device, model_name, calc_metrics=True)

    plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path=os.path.join(plot_dir, "final_training_history.png"))

    
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
