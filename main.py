import os
import torch
from preprocessing import create_data_loaders
from models import ArtStyleCNN, get_resnet_model
from train_utils import train_model
import torch.nn as nn
import torch.optim as optim

if __name__ == "__main__":
    print("\nDane cuda:")
    print(torch.__version__)                     # np. 2.6.0+cu118
    print(torch.version.cuda)                    # np. '11.8' jeśli CUDA działa
    print(torch.cuda.is_available())             # True tylko jeśli działa GPU
    print("\n")

    # ścieżki do katalogów
    #Dominik
    # input_directory = r"C:\Users\Dominik\Desktop\DataSet"
    # output_directory = r"C:\Users\Dominik\Desktop\outData"
    
    #Karolina
    input_directory = r"C:\Users\kkuro\Desktop\Studia\semestr_6\Podstawy_AI\DataSet"
    output_directory = r"C:\Users\kkuro\Desktop\Studia\semestr_6\Podstawy_AI\outData"

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
        "CustomCNN": ArtStyleCNN(num_classes),
        #"ResNet18_pretrained": get_resnet_model(num_classes, pretrained=True),
        #"ResNet18_scratch": get_resnet_model(num_classes, pretrained=False)
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
            print("\n📋 Podsumowanie treningu:")
            print(f"Najlepsza dokładność walidacyjna: {max(history['Dokładność_walidacyjna']):.2f}%")
            print(f"Finalne metryki:")
            print(f" - Accuracy: {final_metrics[0]:.2f}%")
            print(f" - Precision: {final_metrics[2]:.4f}")
            print(f" - Recall: {final_metrics[3]:.4f}")
            print(f" - F1-Score: {final_metrics[4]:.4f}")
        except Exception as e:
            print(f"❌ Błąd w modelu {model_name}: {e}")
