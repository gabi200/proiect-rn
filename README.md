




# Recunoastere semne de circulatie

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Georgescu Gabriel
**Dată actualizare:** 09.01.2026

---

## Introducere

Acest proiect implementează un sistem de recunoaștere a semnelor de circulație implementat în Python, folosind în principal bibliotecile YOLO si OpenCV.

## Instrucțiuni de rulare

Aplicația a fost testată pe Python 3.12.10.

Daca aplicația este rulată pe **Windows**, se recomandă folosirea [Python Install Manager](https://www.python.org/downloads/release/pymanager-252/):

- Folosind `Python Install Manager`, instalați Python 3.12:

  `py install 3.12`

- Clonați repository-ul și schimbați directorul curent:
	`git clone https://github.com/gabi200/proiect-rn.git`
	`cd proiect-rn`

- Instalați dependențele:

   `py -V:3.12 -m pip install -r .\requirements.txt`

- Rulați aplicatia:

  `py -V:3.12 .\src\app\main.py`

	- **IMPORTANT:** Datorită numărului mare de fișiere din dataset, nu este fezabilă și nici best-practice încarcarea acestora pe GitHub. Înainte de orice altă operațiune, selectați `Download dataset and generate data` și asteptați finalizarea scriptului, pentru descarcarea dataset-ului (de pe Kaggle) și apoi generarea datelor originale. În caz că apar eventuale probleme la descărcare și/sau generarea dataset-ului, datele sunt disponibile la [acest link Google Drive.](https://drive.google.com/drive/folders/1R2kPJKzK182LXeOGBuusAnqU6W9ZeAEa?usp=sharing)
	- Pentru rularea intefaței web, selectați `Run web UI`, iar pentru evaluare selectați `Evaluate model`.

### Pentru antrenare:
- Deoarece acesta este un SIA care lucrează cu imagini, se recomandă folosirea unui **GPU** pentru antrenare (de ex. prin tehnologia CUDA pentru Nvidia). Antrenarea pe **CPU** este extrem de lentă.
- Este necesară instalarea versiunii PyTorch corespunzătoare pentru sistemul pe care este efectuată antrenarea, pentru suport CUDA/ROCm: [Download PyTorch](https://pytorch.org/)
- Modelul a fost antrenat folosind CUDA pe un GPU Nvidia GeForce RTX 5060, 8GB VRAM. Pentru seria **RTX 5000**, se poate folosi următoarea comandă pentru a instala PyTorch:

`py -V:3.12 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130`
  
  - Rulația aplicația (ca mai sus) și selectați `Train model`. La prompt-ul `Enter custom training parameters?` selectați `n` pentru a continua cu setările predefinite.
## Despre arhitectura RN

Am ales folosirea **YOLO** deoarece acest model este specializat pe detecția de obiecte/feature-uri și a fost folosit si in detecția de semne de circulație. A fost aleasă versiunea **YOLOv9**, deoarece aceasta oferă un echilibru între performanța detecției și resursele utilizate. Astfel, sistemul poate fi rulat si pe sisteme embedded, de exemplu un **calculator de bord** inclus într-un vehicul sau un **single-board computer** (SBC).

##  1. Structura Repository-ului Github 

```
proiect-rn-[prenume-nume]/
├── README.md                           # Overview general proiect (actualizat)
├── etapa3_analiza_date.md         # Din Etapa 3
├── etapa4_arhitectura_sia.md      # Din Etapa 4
├── etapa5_antrenare_model.md      # ← ACEST FIȘIER (completat)
│
├── docs/
│   ├── state_machine.png              # Din Etapa 4
│   ├── loss_curve.png                 # NOU - Grafic antrenare
│   ├── confusion_matrix.png           # (opțional - Nivel 3)
│   └── screenshots/
│       ├── inference_real.png         # NOU - OBLIGATORIU
│       └── ui_demo.png                # Din Etapa 4
│
├── data/                               # Din Etapa 3-4 (NESCHIMBAT)
│   ├── raw/
│   ├── generated/                     # Contribuția voastră 40%
│   ├── processed/
│   ├── train/
│   ├── validation/
│   └── test/
│
├── src/
│   ├── data_acquisition/              # Din Etapa 4
│   ├── preprocessing/                 # Din Etapa 3
│   │   └── combine_datasets.py        # NOU (dacă ați adăugat date în Etapa 4)
│   ├── neural_network/
│   │   ├── model.py                   # Din Etapa 4
│   │   ├── train.py                   # NOU - Script antrenare
│   │   └── evaluate.py                # NOU - Script evaluare
│   └── app/
│       └── main.py                    # ACTUALIZAT - încarcă model antrenat
│
├── models/
│   ├── untrained_model.h5             # Din Etapa 4
│   ├── trained_model.h5               # NOU - OBLIGATORIU
│   └── final_model.onnx               # (opțional - Nivel 3 bonus)
│
├── results/                            # NOU - Folder rezultate antrenare
│   ├── training_history.csv           # OBLIGATORIU - toate epoch-urile
│   ├── test_metrics.json              # Metrici finale pe test set
│   └── hyperparameters.yaml           # Hiperparametri folosiți
│
├── config/
│   └── preprocessing_params.pkl       # Din Etapa 3 (NESCHIMBAT)
│
├── requirements.txt                    # Actualizat
└── .gitignore
```


##  2. Descrierea Setului de Date


### 2.1 Sursa datelor

* **Origine:** Imagini de pe Google Maps, YouTube, alte surse publice
* **Modul de achiziție:** dataset public + generare
* 
### 2.2 Caracteristicile dataset-ului

* **Număr total de observații:** 7634
* **Număr de caracteristici (features):** 1
* **Tipuri de date:** Imagini/Categoriale
* **Format fișiere:** PNG

### 2.3 Descrierea fiecărei caracteristici

| **Caracteristică** | **Tip** | **Unitate** | **Descriere** | **Domeniu valori** |
|-------------------|---------|-------------|---------------|--------------------|
| trafficsign_name | categorial | – | Numele semnului de circulatie | ex.: forb_left |

---

##  3. Analiza Exploratorie a Datelor (EDA)

### 3.1 Statistici descriptive aplicate

* **Distribuții pe caracteristici** (histograme) - Se analizează distribuție în funcție de categoria semnului de circulație

##  4. Preprocesarea Datelor

###  4.1 Transformarea caracteristicilor

* **Augumentarea datelor:** generare de caracteristici random (linii, pătrate) pe imagini pentru a diversifica setul de date si a simula condiții reale. După augumentare, s-a dublat setul de date, jumătate din total fiind date augementate.


### 4.2 Structurarea seturilor de date

**Împărțirea datelor:**
* 65% – train
* 17.5% – validation
* 17.5% – test

**Principii respectate:**
* Stratificare pentru clasificare
* Fără scurgere de informație (data leakage)
* Statistici calculate DOAR pe train și aplicate pe celelalte seturi

## 5. Nevoile rezvolate de SIA


| **Nevoie reală concretă** | **Cum o rezolvă SIA-ul vostru** | **Modul software responsabil** |
|---------------------------|--------------------------------|--------------------------------|
| Detectarea semnelor de circulatie in conditii reale si variate | Model performant, date training variate si bine augumentate -> 30% înbunătățire recunoaștere în situații complexe| RN  |
|Rularea eficientă pe diferite platforme hardware și integrarea cu hardware fizic | Folosirea OpenCV și unui model optimizat pentru o utilizare redusă a resurselor | RN + App |

## 6. Contribuția originală la setul de date

**Total observații finale:** 7634 (după Etapa 3 + Etapa 4)
**Observații originale:** 3252 (42.6%)

**Tipul contribuției:**
[ ]  Date generate prin simulare fizică  
[ ] Date achiziționate cu senzori proprii  
[ ] Etichetare/adnotare manuală  
[ ] Date sintetice prin metode avansate  
[X] Date generate computațional

**Descriere detaliată:**
Am generat feature-uri random (linii, patrate) pe imaginile din dataset,
folosind OpenCV, reprezentand o augumentare complexa a datelor.

**Locația codului:** `src/data_acquisition/generate_img.py`

**Locația datelor:** `data/generated/`

**Dovezi:**

- Log generare: `docs/log_generare.txt`


## 7. Diagrama State Machine
### Justificarea State Machine-ului ales:


Am ales arhitectura de monitorizare continuă deoarece proiectul poate fi integrat într-un sistem 
de control al unui vehicul autonom, unde reacția în timp real este critică.

Stările principale sunt:
1. Start Web UI: Interfata este pornita de utilizator, porneste inferenta daca exista o camera web.
2. Get image from camera: Obtine o imagine de la camera web cu indexul 0 de pe sistem
3. Inference: Ruleaza reteaua neuronala pentru a identifica semnele de circulatie din imagine
4. Display inference output: se afiseaza clasele identificate pe imagine
5. Wait for user input: se asteapta ca utilizatorul sa faca o actiune (sa schimbe tab-ul, sa incarce o imagine)
6. Fetch and display histograms: se apeleaza modulul de analiza si afiseaza histograme relevante


Tranzițiile critice sunt:
- Operare -> STOP: Daca utilizatorul inchide interfata web.
- IDLE -> ERROR FRAME: Daca nu exista o camera video conectata la sistem/s-a pierdut conexiunea.

Starea ERROR este esențială pentru că exista posibilitatea ca, din cauza vibratiilor, sa se piarda conexiunea cu camera intr-un sistem mobil autonom. 

## 8. Hiperparametri și augumentări

| **Hiperparametru** | **Valoare Aleasă** | **Justificare** |
|--------------------|-------------------|-----------------|
| Learning rate | 0.1| Valoare standard YOLO, este adecvată pentru learning rate optimizer `cos_LR` |
| Batch size | 10 | Compromis memorie/stabilitate |
| Number of epochs |  50 | Cu early stopping după 5 epoci fără îmbunătățire |
| Optimizer | SGD (Stochastic Gradient Descent) | Oferă acuratețe sporită în task-urile de object detection |
| Loss function | Classification loss (binary cross-entropy), Box Loss | Metode standard YOLO. Parametri pentru classification loss: cls=1.5. Box loss: 7.5 (default) |
| Activation functions | SiLU (Sigmoid Linear Unit)| Adecvat pentru object detection, inclus in YOLO |


**Augumentări relevante domeniu**

Am aplicat următoarele augumentări:
- `hsv_h=0.015` (hue). Am setat această valoare la o valoare foarte scăzută pentru a nu schimba radical culorile, acestea fiind importante pentru identificarea tipului de acțiune (albastru = indicator de obligație, roșu = interzicere etc.)
- `hsv_s=0.6`(saturation). Valoarea de saturație ajută la simularea diferitelor condiții de lumină sau a semnelor murdare.
- `hsv_v=0.5`(value).  Această valoare reprezintă luminozitatea și ajută la simularea condițiilor de lumină variate.
- `scale=0.8` Această valoare simulează o variație relativ mare de dimeniuni, deoarece semnele de circulație pot fi la diferite distanțe față de vehicul.
- `shear=2.0`. Această valoare este considerată scăzută, deoarece fenomenul de "shear" nu este comun în această aplicație. Însă, a fost aleasă o val. non-zero, deoarece pot fi generate mici fenomene "shear" din cauza lentilei camerei sau a vibrațiilor.
- `perspective=0.001`. Această valoare este importantă, deoarece semnele de circulație sunt  deseori distorsionate. Această augumentare simulează diferite perspective.
- `fliplr=0`. Această augumentare este setată la **zero**, iar acest lucru este **critic**. Setarea default din YOLO este 0.5, ceea ce ar rezulta în imagini care ar fi flipped. Acest lucru este extrem de periculos, deoarece un indicator de *obligatoriu stânga*, ar putea deveni *obligatoriu dreapta*.
- `degrees=3`. Este simulată o variație a  înclinării de maxim 3 grade, simulând o mică înclinare a camerei sau a semnelor.

## Evaluare acuratețe și performanță

**mAP50 (toate clasele)**: 0.917

**F1 (toate clasele)**: 0.83

### Benchmark latență
S-a utilizat modulul de benchmark integrat în biblioteca Ultralytics. Codul de benchmark este disponibil la calea `src/app/latency_benchmark.py`, iar log-ul rulării este disponibil la `docs/demo_latency_test.txt`.


**Rezultat benchmark**: 15.47 ms
