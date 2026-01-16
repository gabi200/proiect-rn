



# 📘 README – Etapa 5: Configurarea și Antrenarea Modelului RN

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Georgescu Gabriel
**Link Repository GitHub:** [Link Github](https://github.com/gabi200/proiect-rn)
**Data predării:** 19.12.2025

---

## Scopul Etapei 5

Această etapă corespunde punctului **6. Configurarea și antrenarea modelului RN** din lista de 9 etape - slide 2 **RN Specificatii proiect.pdf**.

**Obiectiv principal:** Antrenarea efectivă a modelului RN definit în Etapa 4, evaluarea performanței și integrarea în aplicația completă.

**Pornire obligatorie:** Arhitectura completă și funcțională din Etapa 4:
- State Machine definit și justificat
- Cele 3 module funcționale (Data Logging, RN, UI)
- Minimum 40% date originale în dataset

---

## PREREQUISITE – Verificare Etapa 4 (OBLIGATORIU)

**Înainte de a începe Etapa 5, verificați că aveți din Etapa 4:**

- [X] **State Machine** definit și documentat în `docs/state_machine.*`
- [X] **Contribuție ≥40% date originale** în `data/generated/` (verificabil)
- [X] **Modul 1 (Data Logging)** funcțional - produce CSV-uri
- [X] **Modul 2 (RN)** cu arhitectură definită dar NEANTRENATĂ (`models/untrained_model.h5`)
- [X] **Modul 3 (UI/Web Service)** funcțional cu model dummy
- [X] **Tabelul "Nevoie → Soluție → Modul"** complet în README Etapa 4

** Dacă oricare din punctele de mai sus lipsește → reveniți la Etapa 4 înainte de a continua.**

---

## Pregătire Date pentru Antrenare 

### Dacă ați adăugat date noi în Etapa 4 (contribuția de 40%):

**TREBUIE să refaceți preprocesarea pe dataset-ul COMBINAT:**

Exemplu:
```bash
# 1. Combinare date vechi (Etapa 3) + noi (Etapa 4)
python src/preprocessing/combine_datasets.py

# 2. Refacere preprocesare COMPLETĂ
python src/preprocessing/data_cleaner.py
python src/preprocessing/feature_engineering.py
python src/preprocessing/data_splitter.py --stratify --random_state 42

# Verificare finală:
# data/train/ → trebuie să conțină date vechi + noi
# data/validation/ → trebuie să conțină date vechi + noi
# data/test/ → trebuie să conțină date vechi + noi
```

** ATENȚIE - Folosiți ACEIAȘI parametri de preprocesare:**
- Același `scaler` salvat în `config/preprocessing_params.pkl`
- Aceiași proporții split: 70% train / 15% validation / 15% test
- Același `random_state=42` pentru reproducibilitate

**Verificare rapidă:**
```python
import pandas as pd
train = pd.read_csv('data/train/X_train.csv')
print(f"Train samples: {len(train)}")  # Trebuie să includă date noi
```

---

##  Cerințe Structurate pe 3 Niveluri

### Nivel 1 – Obligatoriu pentru Toți (70% din punctaj)

Completați **TOATE** punctele următoare:

1. **Antrenare model** definit în Etapa 4 pe setul final de date (≥40% originale)
2. **Minimum 10 epoci**, batch size 8–32
3. **Împărțire stratificată** train/validation/test: 70% / 15% / 15%
4. **Tabel justificare hiperparametri** (vezi secțiunea de mai jos - OBLIGATORIU)
5. **Metrici calculate pe test set:**
   - **Acuratețe ≥ 65%**
   - **F1-score (macro) ≥ 0.60**
6. **Salvare model antrenat** în `models/trained_model.h5` (Keras/TensorFlow) sau `.pt` (PyTorch) sau `.lvmodel` (LabVIEW)
7. **Integrare în UI din Etapa 4:**
   - UI trebuie să încarce modelul ANTRENAT (nu dummy)
   - Inferență REALĂ demonstrată
   - Screenshot în `docs/screenshots/inference_real.png`

#### Tabel Hiperparametri și Justificări (OBLIGATORIU - Nivel 1)

Completați tabelul cu hiperparametrii folosiți și **justificați fiecare alegere**:

| **Hiperparametru** | **Valoare Aleasă** | **Justificare** |
|--------------------|-------------------|-----------------|
| Learning rate | 0.1| Valoare standard YOLO, este adecvată pentru learning rate optimizer `cos_LR` |
| Batch size | 10 | Compromis memorie/stabilitate |
| Number of epochs |  50 | Cu early stopping după 5 epoci fără îmbunătățire |
| Optimizer | SGD (Stochastic Gradient Descent) | Oferă acuratețe sporită în task-urile de object detection |
| Loss function | Classification loss (binary cross-entropy), Box Loss | Metode standard YOLO. Parametri pentru classification loss: cls=1.5. Box loss: 7.5 (default) |
| Activation functions | SiLU (Sigmoid Linear Unit)| Adecvat pentru object detection, inclus in YOLO |

**Justificare detaliată batch size**

Am ales `batch_size=10` pentru că avem N=7634 samples → 7634/11 = 694 iterații/epocă.
Aceasta oferă un echilibru între:
- Stabilitate gradient (batch prea mic → zgomot mare în gradient)
- Memorie GPU (batch prea mare → out of memory)

Batch size a fost determinat experimental. Modelul a fost antrenat pe un GPU cu 8 GB VRAM, TDP 150W. Au fost testate valori între 9 și 16, iar pentru a determina valoarea optimă am urmărit puterea consumată de GPU și utilizarea VRAM. 
Puterea electrică consumată este indicatorul optim pentru munca efectivă realizată de GPU. Procentajul de utilizare indicat de sistemul de operare este relativ și poate fi influențat de diferiți factor (ce nuclee din GPU sunt utilizate, frecvența curentă, etc.). Este important ca utilizarea VRAM să fie <8 GB în acest caz, iar în cazul depășirii, o parte din date este stocată în memoria RAM principală. Astfel, apare un bottleneck doarece datele trebuie transferate prin magistrala PCIe, și memoria RAM este mai lentă decât cea VRAM.

Pentru acest workload, puterea maximă atinsă a fost de aprox. **120W** (fluctuează 100-120W) pentru `batch_size=10`. 

**Justificare  parametri loss functions**
Classification loss gain (cls = 1.5). Există 55 de clase, dintre care multe sunt similare (de ex, semnele de limită de viteză). Valoarea default este 0.5, însă am crescut-o deoarece această reprezintă "penalizarea" claselor greșite. Este necesară o penalizare ridicată pentru a diferenția clar și clasele care arată foarte similar.

---

### Nivel 2 – Recomandat (85-90% din punctaj)

Includeți **TOATE** cerințele Nivel 1 + următoarele:

1. **Early Stopping** - oprirea antrenării dacă `val_loss` nu scade în 5 epoci consecutive
2. **Learning Rate Scheduler** - `ReduceLROnPlateau` sau `StepLR`
3. **Augmentări relevante domeniu:**
   - Vibrații motor: zgomot gaussian calibrat, jitter temporal
   - Imagini industriale: slight perspective, lighting variation (nu rotații simple!)
   - Serii temporale: time warping, magnitude warping
4. **Grafic loss și val_loss** în funcție de epoci salvat în `docs/loss_curve.png`
5. **Analiză erori context industrial** (vezi secțiunea dedicată mai jos - OBLIGATORIU Nivel 2)

**Indicatori țintă Nivel 2:**
- **Acuratețe ≥ 75%**
- **F1-score (macro) ≥ 0.70**

**Justificare learning rate scheduler**
Am folosit learning scheduler `cos_lr` (cosine annealing), deoarece acesta ajută în cazurile în care clasele sunt similare (de ex. un semn de limită de viteză 30 km/h vs. limită 50 km/h) și rezultă într-o acuratețe mai bună pentru această aplicație.

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

---

### Nivel 3 – Bonus (până la 100%)

**Punctaj bonus per activitate:**

| **Activitate** |  **Livrabil** |
|----------------|--------------|
| Comparare 2+ arhitecturi diferite | Tabel comparativ + justificare alegere finală în README |
| Export ONNX/TFLite + benchmark latență | Fișier `models/final_model.onnx` + demonstrație <50ms |
| Confusion Matrix + analiză 5 exemple greșite | `docs/confusion_matrix.png` + analiză în README |


### Benchmark latență
S-a utilizat modulul de benchmark integrat în biblioteca Ultralytics. Codul de benchmark este disponibil la calea `src/app/latency_benchmark.py`, iar log-ul rulării este disponibil la `docs/demo_latency_test.txt`.

**Rezultat benchmark**: 15.47 ms


---

## Verificare Consistență cu State Machine (Etapa 4)

Antrenarea și inferența trebuie să respecte fluxul din State Machine-ul vostru definit în Etapa 4.

**Exemplu pentru monitorizare vibrații lagăr:**

| **Stare din Etapa 4** | **Implementare în Etapa 5** |
|-----------------------|-----------------------------|
| `ACQUIRE_DATA` | Citire batch date din `data/train/` pentru antrenare |
| `PREPROCESS` | Aplicare scaler salvat din `config/preprocessing_params.pkl` |
| `RN_INFERENCE` | Forward pass cu model ANTRENAT (nu weights random) |
| `THRESHOLD_CHECK` | Clasificare Normal/Uzură pe baza output RN antrenat |
| `ALERT` | Trigger în UI bazat pe predicție modelului real |

**În `src/app/main.py` (UI actualizat):**

Verificați că **TOATE stările** din State Machine sunt implementate cu modelul antrenat:

```python
# ÎNAINTE (Etapa 4 - model dummy):
model = keras.models.load_model('models/untrained_model.h5')  # weights random
prediction = model.predict(input_scaled)  # output aproape aleator

# ACUM (Etapa 5 - model antrenat):
model = keras.models.load_model('models/trained_model.h5')  # weights antrenate
prediction = model.predict(input_scaled)  # predicție REALĂ și corectă
```

---

## Analiză Erori în Context Industrial (OBLIGATORIU Nivel 2)

**Nu e suficient să raportați doar acuratețea globală.** Analizați performanța în contextul aplicației voastre industriale:

### 1. Pe ce clase greșește cel mai mult modelul?

**Completați pentru proiectul vostru:**

Modelul confundă forb_speed_over_80 (limită de viteză 80 km/h) cu  forb_overtake (depășirea interzisă) în 33% din cazuri. Acest fenomen se întâmplă deoarece acestea au aceeași formă (circulară), aceeași culoare (margine roșie și fundal alb), singura diferență fiind simbolul din interior.

### 2. Ce caracteristici ale datelor cauzează erori?

Modelul are dificultăți în identificarea detaliilor fine (simboluri sau cifre), acesta punând prea mult accent pe forma și culoarea semnelor. De asemenea, apar probleme în special când semnul ocupă sub 5% din suprafața imaginii.

### 3. Ce implicații are pentru aplicația industrială?

FALSE NEGATIVES și FALSE POSITIVES: ambele pot fi **critice**, în funcție de semnul nedetectat sau fals detectat.

De exemplu, detectarea falsă a unui semn de "drum cu prioriatate" într-o situație în care vehiculul de fapt nu avea prioritate -> eroare catastrofică. 

În același mod, nedetectarea unui semn STOP rezultă într-o greșelă critică (neacordare de prioritate).

### 4. Ce măsuri corective propuneți?


1. Colectare imagini adiționale pentru clasele care generează confuzie (forb_overtake și semnele de limită de viteză)
2. Modificări arhitecturale pentru anumite categorii de semne (de exemplu, identificarea unui semn de limită de viteză generic și apoi rularea unui altui stage (OCR) pentru identificarea cifrelor)
3. Implementare filtru Gaussian blur
4. Creștere rezoluție imagini (de la 640px la 960px)

---

## Structura Repository-ului la Finalul Etapei 5

**Clarificare organizare:** Vom folosi **README-uri separate** pentru fiecare etapă în folderul `docs/`:

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

**Diferențe față de Etapa 4:**
- Adăugat `docs/etapa5_antrenare_model.md` (acest fișier)
- Adăugat `docs/loss_curve.png` (Nivel 2)
- Adăugat `models/trained_model.h5` - OBLIGATORIU
- Adăugat `results/` cu history și metrici
- Adăugat `src/neural_network/train.py` și `evaluate.py`
- Actualizat `src/app/main.py` să încarce model antrenat

---
## Instrucțiuni rulare

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
---

## Checklist Final – Bifați Totul Înainte de Predare

### Prerequisite Etapa 4 (verificare)
- [X] State Machine există și e documentat în `docs/state_machine.*`
- [X] Contribuție ≥40% date originale verificabilă în `data/generated/`
- [X] Cele 3 module din Etapa 4 funcționale

### Preprocesare și Date
- [X] Dataset combinat (vechi + nou) preprocesat (dacă ați adăugat date)
- [X] Split train/val/test: 70/15/15% (verificat dimensiuni fișiere)
- [X] Scaler din Etapa 3 folosit consistent (`config/preprocessing_params.pkl`)

### Antrenare Model - Nivel 1 (OBLIGATORIU)
- [X] Model antrenat de la ZERO (nu fine-tuning pe model pre-antrenat)
- [X] Minimum 10 epoci rulate (verificabil în `results/training_history.csv`)
- [X] Tabel hiperparametri + justificări completat în acest README
- [X] Metrici calculate pe test set: **Accuracy ≥65%**, **F1 ≥0.60**
- [X] Model salvat în `models/trained_model.h5` (sau .pt, .lvmodel)
- [X] `results/training_history.csv` există cu toate epoch-urile

### Integrare UI și Demonstrație - Nivel 1 (OBLIGATORIU)
- [X] Model ANTRENAT încărcat în UI din Etapa 4 (nu model dummy)
- [X] UI face inferență REALĂ cu predicții corecte
- [X] Screenshot inferență reală în `docs/screenshots/inference_real.png`
- [X] Verificat: predicțiile sunt diferite față de Etapa 4 (când erau random)

### Documentație Nivel 2 (dacă aplicabil)
- [X] Early stopping implementat și documentat în cod
- [X] Learning rate scheduler folosit (ReduceLROnPlateau / StepLR)
- [X] Augmentări relevante domeniu aplicate (NU rotații simple!)
- [X] Grafic loss/val_loss salvat în `docs/loss_curve.png`
- [X] Analiză erori în context industrial completată (4 întrebări răspunse)
- [X] Metrici Nivel 2: **Accuracy ≥75%**, **F1 ≥0.70**

### Documentație Nivel 3 Bonus (dacă aplicabil)
- [ ] Comparație 2+ arhitecturi (tabel comparativ + justificare)
- [X] Export ONNX/TFLite + benchmark latență (<50ms demonstrat)
- [ ] Confusion matrix + analiză 5 exemple greșite cu implicații

### Verificări Tehnice
- [ ] `requirements.txt` actualizat cu toate bibliotecile noi
- [X] Toate path-urile RELATIVE (nu absolute: `/Users/...` )
- [ ] Cod nou comentat în limba română sau engleză (minimum 15%)
- [X] `git log` arată commit-uri incrementale (NU 1 commit gigantic)
- [ ] Verificare anti-plagiat: toate punctele 1-5 respectate

### Verificare State Machine (Etapa 4)
- [X] Fluxul de inferență respectă stările din State Machine
- [X] Toate stările critice (PREPROCESS, INFERENCE, ALERT) folosesc model antrenat
- [X] UI reflectă State Machine-ul pentru utilizatorul final

### Pre-Predare
- [X] `docs/etapa5_antrenare_model.md` completat cu TOATE secțiunile
- [X] Structură repository conformă: `docs/`, `results/`, `models/` actualizate
- [ ] Commit: `"Etapa 5 completă – Accuracy=X.XX, F1=X.XX"`
- [ ] Tag: `git tag -a v0.5-model-trained -m "Etapa 5 - Model antrenat"`
- [ ] Push: `git push origin main --tags`
- [X] Repository accesibil (public sau privat cu acces profesori)

---

## Livrabile Obligatorii (Nivel 1)

Asigurați-vă că următoarele fișiere există și sunt completate:

1. **`docs/etapa5_antrenare_model.md`** (acest fișier) cu:
   - Tabel hiperparametri + justificări (complet)
   - Metrici test set raportate (accuracy, F1)
   - (Nivel 2) Analiză erori context industrial (4 paragrafe)

2. **`models/trained_model.h5`** (sau `.pt`, `.lvmodel`) - model antrenat funcțional

3. **`results/training_history.csv`** - toate epoch-urile salvate

4. **`results/test_metrics.json`** - metrici finale:

Exemplu:
```json
{
  "test_accuracy": 0.7823,
  "test_f1_macro": 0.7456,
  "test_precision_macro": 0.7612,
  "test_recall_macro": 0.7321
}
```

5. **`docs/screenshots/inference_real.png`** - demonstrație UI cu model antrenat

6. **(Nivel 2)** `docs/loss_curve.png` - grafic loss vs val_loss

7. **(Nivel 3)** `docs/confusion_matrix.png` + analiză în README

---

## Predare și Contact

**Predarea se face prin:**
1. Commit pe GitHub: `"Etapa 5 completă – Accuracy=X.XX, F1=X.XX"`
2. Tag: `git tag -a v0.5-model-trained -m "Etapa 5 - Model antrenat"`
3. Push: `git push origin main --tags`

---
