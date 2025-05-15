import os
import torch
from preprocessing import create_data_loaders
from models import ArtStyleCNN, get_resnet_model
from train_utils import train_model
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    print("\nDane cuda:")
    print(torch.__version__)                     # np. 2.6.0+cu118
    print(torch.version.cuda)                    # np. '11.8' jeśli CUDA działa
    print(torch.cuda.is_available())             # True tylko jeśli działa GPU
    print("\n")

    # ścieżki do katalogów
    #Dominik
    input_directory = r"C:\Users\Dominik\Desktop\DataSet"
    output_directory = r"C:\Users\Dominik\Desktop\outData"
    
    #Karolina
    #input_directory = r"C:\Users\kkuro\Desktop\Studia\semestr_6\Podstawy_AI\DataSet"
    #output_directory = r"C:\Users\kkuro\Desktop\Studia\semestr_6\Podstawy_AI\outData"

    # Parametry
    img_size = 224
    batch_size = 32
    
    start_epoch = 15        # liczba epok już przeprowadzonych
    total_epochs = 17        # do ilu epok chcesz dociągnąć model
    continue_training = True 

    # Ścieżki wyjściowe
    processed_dir = os.path.join(output_directory, "processed")
    split_dir = os.path.join(output_directory, "split")

    data_loaders = create_data_loaders(split_dir, batch_size=batch_size, img_size=img_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Używane urządzenie: {device}")

    num_classes = len(data_loaders['train_dataset'].label_to_idx)

    models_to_train = {
        #"CustomCNN": ArtStyleCNN(num_classes),
        "ResNet18_pretrained": get_resnet_model(num_classes, pretrained=True),
        #"ResNet18_scratch": get_resnet_model(num_classes, pretrained=False)
    }
    
    for model_name, model in models_to_train.items():
        print(f"\n🧠 Trenuję model: {model_name}")

        # Ścieżki do modelu i historii treningu
        models_dir = "Models"  # Folder na modele
        metrics_dir = "Metrics"  # Poprawiony folder name typo "Matrics" -> "Metrics"
        
        # Upewnij się, że foldery istnieją
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(metrics_dir, exist_ok=True)
        
        # Ścieżki do plików
        model_path = os.path.join(models_dir, f"{model_name}_best_model.pt")
        history_path = os.path.join(metrics_dir, f'{model_name}_training_history.csv')
        
        # Inicjalizuj start_epoch
        start_epoch = 0
        
        # Wczytywanie historii treningu (jeśli istnieje)
        previous_history = None
        if continue_training and os.path.exists(history_path):
            previous_history = pd.read_csv(history_path)
            print(f"📊 Wczytano historię treningu z {history_path} ({len(previous_history)} epok)")

            if len(previous_history) > 0:
                # Zapisz najwyższą epokę z poprzedniej historii
                start_epoch = int(previous_history['Epoka'].max())+1
                max_previous_epoch = previous_history['Epoka'].max()
                last_val_acc = previous_history['Dokładność_walidacyjna'].iloc[-1]
                print(f"📈 Ostatnia zapisana epoka: {max_previous_epoch}, dokładność walidacyjna: {last_val_acc:.2f}%")
                # W przypadku wznowienia treningu, możemy zacząć od epoki 1, ale będziemy wiedzieć
                # że musimy przesunąć numerację przy łączeniu historii
        
        # Wczytywanie modelu (jeśli kontynuujemy trening)
        if continue_training:
            if os.path.exists(model_path):
                print(f"🔄 Wczytywanie modelu z {model_path}")
                model.load_state_dict(torch.load(model_path, map_location=device))
            else:
                print(f"⚠️ Nie znaleziono {model_path}. Trening rozpocznie się od zera.")
                previous_history = None  # Reset poprzedniej historii gdy nie ma zapisanego modelu

        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        try:
            # Trenuj model
            # Tutaj nie przekazujemy start_epoch, bo funkcja train_model najprawdopodobniej
            # zawsze zaczyna od epoki 1, co widać w danych CSV
            history_df, final_metrics = train_model(model, 
                                            data_loaders['train_loader'], 
                                            data_loaders['val_loader'], 
                                            criterion, optimizer, 
                                            device, 
                                            model_name=model_name,
                                            num_epochs=total_epochs - start_epoch,
                                            start_epoch=start_epoch 
                                            )
            
            print(f"✅ Trenowanie zakończone dla modelu {model_name}")
            
            # Zapisz nową historię jako CSV
            if history_df is not None:
                # Łączenie z poprzednią historią (jeśli istnieje)
                if previous_history is not None:
                    # Przesuń numerację epok w nowej historii, aby kontynuować po ostatniej epoce z poprzedniej historii
                    max_previous_epoch = previous_history['Epoka'].max()
                    print(f"🔢 Przesuwanie numeracji epok o {max_previous_epoch} (kontynuacja po epoce {max_previous_epoch})")
                    
                    # Dostosowujemy numerację nowych epok, aby kontynuować po poprzednich
                    history_df['Epoka'] = history_df['Epoka'] + max_previous_epoch
                    
                    # Łączymy historie
                    full_history = pd.concat([previous_history, history_df], ignore_index=True)
                    
                    # Dla pewności sortujemy po numerze epoki
                    full_history = full_history.sort_values('Epoka').reset_index(drop=True)
                else:
                    full_history = history_df
                
                # Zapisz pełną historię
                full_history.to_csv(history_path, index=False)
                print(f"💾 Zapisano pełną historię treningu do {history_path}")
                
                # Rysuj pełną historię treningu
                plot_dir = os.path.join("training_plots", model_name)
                os.makedirs(plot_dir, exist_ok=True)
                
                plt.figure(figsize=(12, 5))
                plt.suptitle(f'{model_name} – Pełna historia treningu ({len(full_history)} epok)', fontsize=16)
                
                # Wykres funkcji straty
                plt.subplot(1, 2, 1)
                plt.plot(full_history['Epoka'], full_history['Strata_treningowa'], 'b-', label='Strata treningowa')
                plt.plot(full_history['Epoka'], full_history['Strata_walidacyjna'], 'r-', label='Strata walidacyjna')
                plt.title('Funkcje straty')
                plt.xlabel('Epoki')
                plt.ylabel('Strata')
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
                
                # Wykres dokładności
                plt.subplot(1, 2, 2)
                plt.plot(full_history['Epoka'], full_history['Dokładność_treningowa'], 'b-', label='Dokładność treningowa')
                plt.plot(full_history['Epoka'], full_history['Dokładność_walidacyjna'], 'r-', label='Dokładność walidacyjna')
                plt.title('Dokładność')
                plt.xlabel('Epoki')
                plt.ylabel('Dokładność [%]')
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
                
                # Dodaj etykiety dla poszczególnych sesji treningowych
                if previous_history is not None:
                    # Dodaj pionową linię rozdzielającą sesje treningowe
                    max_previous_epoch = previous_history['Epoka'].max()
                    plt.subplot(1, 2, 1)
                    plt.axvline(x=max_previous_epoch, color='green', linestyle='--', alpha=0.7)
                    plt.text(max_previous_epoch+0.1, plt.ylim()[1]*0.9, 'Nowa sesja', 
                            rotation=90, color='green', verticalalignment='top')
                    
                    plt.subplot(1, 2, 2)
                    plt.axvline(x=max_previous_epoch, color='green', linestyle='--', alpha=0.7)
                    plt.text(max_previous_epoch+0.1, plt.ylim()[1]*0.9, 'Nowa sesja', 
                            rotation=90, color='green', verticalalignment='top')
                
                plt.tight_layout()
                
                full_history_plot_path = os.path.join(plot_dir, f"{model_name}_full_training_history.png")
                plt.savefig(full_history_plot_path)
                print(f"📊 Zapisano pełną historię treningu do {full_history_plot_path}")
            
            if final_metrics:
                print("\n📋 Podsumowanie treningu:")
                if history_df is not None and 'Dokładność_walidacyjna' in history_df:
                    # Znajdź najlepszą dokładność w nowej sesji treningowej
                    max_val_acc = max(history_df['Dokładność_walidacyjna'])
                    epoch_with_max = history_df.loc[history_df['Dokładność_walidacyjna'].idxmax(), 'Epoka']
                    print(f"Najlepsza dokładność walidacyjna w tej sesji: {max_val_acc:.2f}% (epoka {epoch_with_max})")
                    
                    if previous_history is not None:
                        # Znajdź najlepszą dokładność we wszystkich epokach
                        overall_max_val_acc = max(full_history['Dokładność_walidacyjna'])
                        overall_best_epoch = full_history.loc[full_history['Dokładność_walidacyjna'].idxmax(), 'Epoka']
                        print(f"Najlepsza dokładność walidacyjna ogółem: {overall_max_val_acc:.2f}% (epoka {overall_best_epoch})")
                
                print(f"Finalne metryki:")
                print(f" - Accuracy: {final_metrics[0]:.2f}%")
                print(f" - Precision: {final_metrics[2]:.4f}")
                print(f" - Recall: {final_metrics[3]:.4f}")
                print(f" - F1-Score: {final_metrics[4]:.4f}")
            
        except Exception as e:
            print(f"❌ Błąd w modelu {model_name}: {e}")
            import traceback
            traceback.print_exc()
