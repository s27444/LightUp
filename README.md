# Light Up (Akari) - Solver

## Opis Problemu
Light Up (znany również jako Akari) to łamigłówka logiczna rozgrywana na prostokątnej siatce. Siatka zawiera białe i czarne komórki. Celem jest umieszczenie żarówek na białych komórkach zgodnie z następującymi zasadami:

1. Każda żarówka oświetla cały swój wiersz i kolumnę, chyba że na drodze stoi czarna komórka
2. Żadna żarówka nie może oświetlać innej żarówki
3. Czarne komórki mogą zawierać liczby (0-4), wskazujące dokładnie ile żarówek musi być umieszczonych w sąsiedztwie tej komórki
4. Każda biała komórka musi być oświetlona przez co najmniej jedną żarówkę

Jest to problem NP-zupełny, co czyni go idealnym do testowania algorytmów optymalizacyjnych.

## Struktura Projektu

```
LightUp/
│
├── lightup/                # Główna logika gry
│   ├── board.py            # Reprezentacja planszy, ładowanie, walidacja
│   ├── solution.py         # Reprezentacja rozwiązania, generowanie sąsiedztwa
│   ├── objective.py        # Funkcja celu (ocena rozwiązania)
│   └── utils.py            # Funkcje pomocnicze
│
├── algorithms/             # Algorytmy optymalizacyjne
│   ├── brute_force.py      # Algorytm pełnego przeglądu
│   ├── hill_climbing.py    # Algorytm wspinaczkowy
│   ├── tabu_search.py      # Przeszukiwanie z zakazami
│   ├── simulated_annealing.py # Symulowane wyżarzanie
│   ├── genetic_algorithm.py   # Algorytm genetyczny
│   ├── parallel_genetic.py    # Równoległa wersja algorytmu genetycznego
│   ├── island_genetic.py      # Algorytm genetyczny w wersji wyspowej
│   └── evolutionary_strategy.py # Strategie ewolucyjne dla funkcji testowych
│
├── experiments/            # Skrypty do eksperymentów i porównań
│   └── compare_methods.py
│
├── data/                   # Przykładowe plansze wejściowe
│
├── main.py                 # Główny punkt wejścia (CLI)
└── requirements.txt
```

## Instalacja

1. Sklonuj repozytorium:
```
git clone https://github.com/yourusername/LightUp.git
cd LightUp
```

2. Utwórz wirtualne środowisko:
```
python -m venv venv
source venv/bin/activate  # W Windows: venv\Scripts\activate
```

3. Zainstaluj zależności:
```
pip install -r requirements.txt
```

## Format Planszy

Pliki wejściowe planszy mają następujący format:
- Pierwsza linia: `<szerokość> <wysokość>`
- Kolejne linie: reprezentacja planszy, gdzie:
  * `.`: Biała komórka (pusta)
  * `#`: Czarna komórka (bez liczby)
  * `1`, `2`, `3`, `4`: Czarna komórka z liczbą
  * `0`: Czarna komórka z liczbą 0 (żadna żarówka nie może być w sąsiedztwie)

Przykład:
```
5 5
.#...
.1#..
.....
..#2.
...#.
```

## Użycie

### Rozwiązywanie łamigłówki

```bash
# Rozwiązanie planszy metodą pełnego przeglądu
python main.py --algorithm brute_force --input data/example1.txt

# Rozwiązanie algorytmem wspinaczkowym
python main.py --algorithm hill_climbing --input data/example1.txt

# Rozwiązanie przeszukiwaniem z zakazami
python main.py --algorithm tabu_search --input data/example1.txt

# Rozwiązanie symulowanym wyżarzaniem
python main.py --algorithm simulated_annealing --input data/example1.txt

# Rozwiązanie algorytmem genetycznym
python main.py --algorithm genetic_algorithm --input data/example1.txt

# Rozwiązanie równoległym algorytmem genetycznym
python main.py --algorithm parallel_genetic --input data/example1.txt

# Rozwiązanie algorytmem genetycznym w wersji wyspowej
python main.py --algorithm island_genetic --input data/example1.txt

# Zapisanie rozwiązania do pliku
python main.py --algorithm hill_climbing --input data/example1.txt --output solution.txt

# Pokazanie szczegółowego postępu
python main.py --algorithm hill_climbing --input data/example1.txt --verbose
```

### Parametry Algorytmów

Każdy algorytm ma specyficzne parametry, które można dostosować:

#### Algorytm Wspinaczkowy (Hill Climbing)
```bash
python main.py --algorithm hill_climbing --input data/example1.txt \
  --stochastic --neighbor-count 20 --restart-count 10
```
- `--stochastic`: Używa stochastycznej wersji algorytmu (losowy wybór sąsiadów)
- `--neighbor-count`: Liczba generowanych sąsiadów w każdej iteracji
- `--restart-count`: Liczba restartów z losowego punktu startowego

#### Przeszukiwanie z Zakazami (Tabu Search)
```bash
python main.py --algorithm tabu_search --input data/example1.txt \
  --tabu-size 20 --aspiration --backtrack --neighbor-count 20 --restart-count 5
```
- `--tabu-size`: Rozmiar listy zakazów
- `--aspiration`: Włącza kryterium aspiracji (akceptuj zakazane ruchy, jeśli prowadzą do lepszego rozwiązania)
- `--backtrack`: Włącza możliwość powrotu do najlepszego znalezionego rozwiązania
- `--neighbor-count`: Liczba generowanych sąsiadów w każdej iteracji
- `--restart-count`: Liczba restartów z losowego punktu startowego

#### Symulowane Wyżarzanie (Simulated Annealing)
```bash
python main.py --algorithm simulated_annealing --input data/example1.txt \
  --initial-temp 200 --cooling-rate 0.005 --cooling-schedule exponential --restart-count 5
```
- `--initial-temp`: Początkowa temperatura
- `--cooling-rate`: Współczynnik chłodzenia
- `--cooling-schedule`: Schemat chłodzenia (linear, exponential, logarithmic)
- `--restart-count`: Liczba restartów z losowego punktu startowego

#### Algorytm Genetyczny (Genetic Algorithm)
```bash
python main.py --algorithm genetic_algorithm --input data/example1.txt \
  --population-size 100 --max-generations 200 --crossover-method uniform \
  --mutation-method random_flip --elite-size 10
```
- `--population-size`: Rozmiar populacji
- `--max-generations`: Maksymalna liczba pokoleń
- `--crossover-method`: Metoda krzyżowania (uniform, single_point)
- `--mutation-method`: Metoda mutacji (random_flip, swap)
- `--elite-size`: Liczba najlepszych osobników zachowywanych bez zmian

#### Równoległy Algorytm Genetyczny (Parallel Genetic Algorithm)
```bash
python main.py --algorithm parallel_genetic --input data/example1.txt \
  --population-size 100 --max-generations 200 --crossover-method uniform \
  --mutation-method random_flip --elite-size 10 --num-workers 4
```
- Wszystkie parametry jak w zwykłym algorytmie genetycznym, plus:
- `--num-workers`: Liczba równoległych procesów (jeśli None, używa liczby dostępnych rdzeni CPU)

#### Algorytm Genetyczny w Wersji Wyspowej (Island Genetic Algorithm)
```bash
python main.py --algorithm island_genetic --input data/example1.txt \
  --population-size 50 --max-generations 100 --crossover-method uniform \
  --mutation-method random_flip --elite-size 5 --num-islands 4 \
  --migration-interval 10 --migration-rate 0.1 --distributed
```
- Wszystkie parametry jak w zwykłym algorytmie genetycznym, plus:
- `--num-islands`: Liczba wysp (podpopulacji)
- `--migration-interval`: Liczba pokoleń między migracjami
- `--migration-rate`: Współczynnik migracji (procent populacji)
- `--distributed`: Flaga włączająca tryb rozproszony (obliczenia na wielu maszynach)

### Porównywanie Metod

Skrypt `compare_methods.py` uruchamia wiele algorytmów na tej samej planszy i porównuje ich wydajność:

```bash
# Podstawowe porównanie
python experiments/compare_methods.py --input data/example1.txt --output results

# Porównaj tylko wybrane algorytmy
python experiments/compare_methods.py --input data/example1.txt --output results \
  --algorithms hill_climbing tabu_search genetic_algorithm parallel_genetic island_genetic

# Ustaw limit czasu na algorytm
python experiments/compare_methods.py --input data/example1.txt --output results --time-limit 30

# Pokaż szczegółowy postęp
python experiments/compare_methods.py --input data/example1.txt --output results --verbose
```

Skrypt generuje:
- Wykres porównania czasu wykonania
- Wykres porównania jakości rozwiązania
- Krzywe zbieżności
- Wykres wykorzystania zasobów vs. jakość rozwiązania
- Plik CSV z podsumowaniem

## Zaimplementowane Algorytmy

### Pełny Przegląd (Brute Force)
Wyczerpujące przeszukiwanie wszystkich możliwych rozwiązań. Praktyczne tylko dla bardzo małych łamigłówek.

### Algorytm Wspinaczkowy (Hill Climbing)
Algorytm lokalnego przeszukiwania, który zawsze przechodzi do sąsiedniego rozwiązania z najlepszym wynikiem. Zawiera zarówno warianty deterministyczne, jak i stochastyczne.

### Przeszukiwanie z Zakazami (Tabu Search)
Zaawansowane lokalne przeszukiwanie, które utrzymuje "listę zakazów" ostatnio odwiedzonych rozwiązań, aby uniknąć cyklicznych powrotów i uciec z lokalnych optimów. Zawiera kryteria aspiracji i backtracking.

### Symulowane Wyżarzanie (Simulated Annealing)
Probabilistyczna technika, która czasami akceptuje gorsze rozwiązania, aby uciec z lokalnych optimów. Prawdopodobieństwo akceptacji maleje wraz ze spadkiem "temperatury" zgodnie z wybranym schematem chłodzenia.

### Algorytm Genetyczny (Genetic Algorithm)
Algorytm ewolucyjny inspirowany selekcją naturalną. Zawiera:
- Dwie metody krzyżowania: jednolite (uniform) i jednopunktowe (single-point)
- Dwie metody mutacji: losową zmianę (random_flip) i zamianę (swap)
- Zachowanie elity (elite preservation)
- Selekcję turniejową (tournament selection)

### Równoległy Algorytm Genetyczny (Parallel Genetic Algorithm)
Równoległa wersja algorytmu genetycznego, która dystrybuuje ocenę funkcji przystosowania na wiele rdzeni CPU dla szybszego wykonania na dużych populacjach. Wykorzystuje Python'owy ProcessPoolExecutor do obliczeń równoległych.

### Algorytm Genetyczny w Wersji Wyspowej (Island Genetic Algorithm)
Rozszerzenie algorytmu genetycznego, które utrzymuje wiele izolowanych subpopulacji (wysp), które ewoluują niezależnie, z okresowymi migracjami osobników między wyspami. Sprzyja to różnorodności i może pomóc uniknąć przedwczesnej zbieżności do lokalnych optimów.

### Demo Programowania Genetycznego (Genetic Programming Demo)
Demonstracja programowania genetycznego zastosowanego do problemu regresji symbolicznej. Ewoluuje wyrażenia matematyczne, aby dopasować się do funkcji docelowej, używając programowania genetycznego opartego na drzewach z operatorami krzyżowania i mutacji.

### Strategie Ewolucyjne (Evolutionary Strategy)
Implementacja strategii ewolucyjnych w wariantach (μ,λ)-ES i (μ+λ)-ES do optymalizacji numerycznej. Testy obejmują kilka funkcji wzorcowych z wieloma lokalnymi optimami, takich jak funkcje Rastrigina, Ackleya i Schwefela.


## Przykładowe Rozwiązanie

Dla przykładowej planszy:
```
5 5
.#...
.1#..
.....
..#2.
...#.
```

Poprawne rozwiązanie może wyglądać następująco:
```
*#L**
L1#**
***L*
**#2L
*L*#*
```

Gdzie:
- `L`: Żarówka
- `*`: Oświetlona komórka
- `#`: Czarna komórka bez liczby
- `1`, `2`, itd.: Czarna komórka z liczbą

## Licencja

Ten projekt jest udostępniony na licencji MIT - szczegóły znajdują się w pliku LICENSE.