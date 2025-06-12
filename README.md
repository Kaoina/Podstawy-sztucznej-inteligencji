## **Podsumowanie projektu – Klasyfikacja stylów artystycznych za pomocą AI**

Celem projektu było stworzenie modelu sztucznej inteligencji zdolnego do klasyfikacji obrazów według stylu artystycznego, na podstawie danych z zestawu WikiArt (13 kategorii, \~30 GB danych). Dane zostały odpowiednio przetworzone – obrazy ujednolicono do rozmiaru 512×512 px z paddingiem, a następnie podzielono na zbiory treningowy, walidacyjny i testowy (70/15/15), dbając o zbalansowanie klas poprzez redukcję i augmentację danych.

Stworzono własny model CNN (**ArtStyleCNN**) oraz porównano go z dwoma wersjami modelu **ResNet18**: trenowaną od zera i pre-trained. Do treningu wykorzystano bibliotekę PyTorch i obliczenia na GPU. Modele oceniono za pomocą wykresów strat i dokładności, metryk (Accuracy, Precision, Recall, F1-score) oraz macierzy pomyłek.

**Najlepsze wyniki osiągnął model ResNet18 z wagami pre-trained**, uzyskując:

* **Accuracy:** 80.12%
* **F1-score:** 0.8021

Model ten skutecznie rozpoznaje większość stylów artystycznych, zwłaszcza **Primitivism** i **Western Medieval**, choć ma trudności z odróżnianiem klas wizualnie podobnych, takich jak **Romanticism** i **Baroque**.

Projekt pokazuje, że zastosowanie pretrenowanych architektur głębokich sieci neuronowych pozwala skutecznie rozpoznawać style malarskie, co może być użyteczne m.in. w muzeach, galeriach czy systemach rekomendacyjnych związanych ze sztuką.
