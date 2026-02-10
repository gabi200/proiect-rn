
## 1. Identificare Proiect

| Câmp | Valoare |
|------|---------|
| **Student** | Georgescu Gabriel |
| **Grupa / Specializare** | 632AB / Informatică Industrială |
| **Disciplina** | Rețele Neuronale |
| **Instituție** | POLITEHNICA București – FIIR |
| **Link Repository GitHub** | https://github.com/gabi200/proiect-rn |
| **Acces Repository** | Public |
| **Stack Tehnologic** | Python |
| **Domeniul Industrial de Interes (DII)** | Automotive |
| **Tip Rețea Neuronală** | CNN |

### Rezultate Cheie (Versiunea Finală vs Etapa 6)

| Metric | Țintă Minimă |  Rezultat Final | Status |
|--------|--------------|------------------|----------------|
| Accuracy (Test Set) | ≥70%  | 95.7% | ✓ |
| F1-Score (Macro) | ≥0.65 | 0.91  | ✓ |
| Latență Inferență | <50 ms  | 17.6 ms | ✓ |
| Contribuție Date Originale | ≥40% |  42.6% | ✓|
| Nr. Experimente Optimizare | ≥4 | 5 | ✓ |

### Declarație de Originalitate & Politica de Utilizare AI

**Acest proiect reflectă munca, gândirea și deciziile mele proprii.**

Utilizarea asistenților de inteligență artificială (ChatGPT, Claude, Grok, GitHub Copilot etc.) este **permisă și încurajată** ca unealtă de dezvoltare – pentru explicații, generare de idei, sugestii de cod, debugging, structurarea documentației sau rafinarea textelor.

**Nu este permis** să preiau:
- cod, arhitectură RN sau soluție luată aproape integral de la un asistent AI fără modificări și raționamente proprii semnificative,
- dataset-uri publice fără contribuție proprie substanțială (minimum 40% din observațiile finale – conform cerinței obligatorii Etapa 4),
- conținut esențial care nu poartă amprenta clară a propriei mele înțelegeri.

**Confirmare explicită (bifez doar ce este adevărat):**

| Nr. | Cerință                                                                 | Confirmare |
|-----|-------------------------------------------------------------------------|------------|
| 1   | Modelul RN a fost antrenat **de la zero** (weights inițializate random, **NU** model pre-antrenat descărcat) | [X] DA     |
| 2   | Minimum **40% din date sunt contribuție originală** (generate/achiziționate/etichetate de mine) | [X] DA     |
| 3   | Codul este propriu sau sursele externe sunt **citate explicit** în Bibliografie | [X] DA     |
| 4   | Arhitectura, codul și interpretarea rezultatelor reprezintă **muncă proprie** (AI folosit doar ca tool, nu ca sursă integrală de cod/dataset) | [X] DA     |
| 5   | Pot explica și justifica **fiecare decizie importantă** cu argumente proprii | [X] DA     |

**Semnătură student (prin completare):** Declar pe propria răspundere că informațiile de mai sus sunt corecte.

---

## 2. Descrierea Nevoii și Soluția SIA

### 2.1 Nevoia Reală / Studiul de Caz

Proiectul abordează problema recunoașterii semnelor de circulație, de obicei ca parte a unui sistem de self-driving sau ADAS (Advanced Driver-Assistance System). Sistemele curente ADAS nu pot face față la toate condițiile de drum și de vreme, și sunt limitate de hardware-ul de procesare instalat pe vehicule.

Scopul proiectului este de a crea un sistem de recunoaștere a semnelor de circulație cu acuratețe ridicată  (95.7%), fară a face un compromis la timpul de răspuns (17.6 ms). Modelul poate rula pe un NPU (Neural Processing Unit), care poate fi integrat într-un vehicul/robot autonom.

### 2.2 Beneficii Măsurabile Urmărite

1. Asigurarea procesării în timp real cu latență scăzută (<20 ms). Latența scăzută permite sistemului să reacționeze și la viteze de autostradă (>130 km/h)
2. Detecția semnelor de circulație într-un sistem de asistență poate duce la o reducere de 25% a accidentelor (de ex. oprirea la STOP în timp util activată de sistemul autonom)
3. Detectarea semnelor de limitare a vitezei și aplicarea limitei rezultă în reduce cu o reducere de 95% a amenzilor de limită depășită

### 2.3 Tabel: Nevoie → Soluție SIA → Modul Software

| **Nevoie reală concretă** | **Cum o rezolvă SIA-ul vostru** | **Modul software responsabil** |
|---------------------------|--------------------------------|--------------------------------|
| Detectarea semnelor de circulatie in conditii reale si variate | Model performant, date training variate si bine augmentate -> 30% înbunătățire recunoaștere în situații complexe| RN  |
|Rularea eficientă pe diferite platforme hardware și integrarea cu hardware fizic | Folosirea OpenCV și unui model optimizat pentru o utilizare redusă a resurselor | RN + App |
| Fiabilitatea sistemului, necesară în industria automotive | Acuratete ridicata RN (>95%), detectie situatii critice (camera acoperita/deconectata) | RN + App |


---

## 3. Dataset și Contribuție Originală

### 3.1 Sursa și Caracteristicile Datelor

| Caracteristică | Valoare |
|----------------|---------|
| **Origine date** |Mixt (Dataset public + generare OpenCV) |
| **Sursa concretă** | Kaggle - https://www.kaggle.com/datasets/raduoprea/traffic-signs/code |
| **Număr total observații finale (N)** | 7634 |
| **Număr features** | 55 |
| **Tipuri de date** | Categoriale |
| **Format fișiere** | JPG |
| **Perioada colectării/generării** | Noiembrie 2025 - Ianuarie 2026 |

### 3.2 Contribuția Originală (minim 40% OBLIGATORIU)

| Câmp | Valoare |
|------|---------|
| **Total observații finale (N)** |7634  |
| **Observații originale (M)** | 3252 |
| **Procent contribuție originală** | 42.6% |
| **Tip contribuție** | Generare date cu OpenCV |
| **Locație cod generare** | `src/data_acquisition/generate_img.py` |
| **Locație date originale** | `data/generated/` |

**Descriere metodă generare/achiziție:**

Am generat feature-uri random (linii, patrate) pe imaginile din dataset, folosind OpenCV, reprezentand o augmentare complexa a datelor. Datele sunt relevante deoarece simulează condiții reale în care semnele sunt murdare, acoperite, prezinta desene etc.

### 3.3 Preprocesare și Split Date

| Set | Procent | Număr Observații |
|-----|---------|------------------|
| Train | 85% | 6488 |
| Validation | 7.5% | 572 |
| Test | 7.5% | 572 |

**Referințe fișiere:** `data/README.md`

---

## 4. Arhitectura SIA și State Machine

### 4.1 Cele 3 Module Software

| Modul | Tehnologie | Funcționalitate Principală | Locație în Repo |
|-------|------------|---------------------------|-----------------|
| **Data Logging / Acquisition** | Python | Descărcare dataset de pe Kaggle, generare date adiționale cu OpenCV | `src/data_acquisition/` |
| **Neural Network** | YOLO (PyTorch) | Clasificare multi-clasă cu CNN | `src/neural_network/`, `src/app/` |
| **Web Service / UI** | Flask | Interfață upload imagine/inferență pe input webcam, simulare dashboard vehicul | `src/app/` |

### 4.2 State Machine

**Locație diagramă:** `docs/state_machine.drawio` *(sau `state_machine_v2.png` dacă actualizată în Etapa 6)*

**Stări principale și descriere:**
1. **Start Web UI:** Interfata este pornita de utilizator, porneste inferenta daca exista o camera web.
2. **Get image from camera:** Obtine o imagine de la camera web cu indexul 0 de pe sistem
3. **Inference:** Ruleaza reteaua neuronala pentru a identifica semnele de circulatie din imagine
4. **Display inference output:** se afiseaza clasele identificate pe imagine
5. **Wait for user input:** se asteapta ca utilizatorul sa faca o actiune (sa schimbe tab-ul, sa incarce o imagine)
6. **Fetch and display histograms:** se apeleaza modulul de analiza si afiseaza histograme relevante
7. **Calculate speed, signal and steering actions:** se determina schimbarea de stare necesara in functie de semnul detectat (ex. semn STOP -> oprire vehicul)
8. **Enable alarm, hazard signal and stop vehicle:** se activeaza alarma, semnalele de avarie si se opreste vehiculul daca camera este acoperita

**Justificare alegere arhitectură State Machine:**

Am ales arhitectura de monitorizare continuă deoarece proiectul poate fi integrat într-un sistem de control al unui vehicul autonom, unde reacția în timp real este critică.


### 4.3 Actualizări State Machine în Etapa 6 (dacă este cazul)

1. **Model înlocuit:** `models/trained_model.pt` → `models/optimized_model.pt`
   - Îmbunătățire: Accuracy +4%, F1 +9%
   - Motivație: aplicația are cerințe ridicate de siguranță și fiabilitate

2. **UI îmbunătățit:**
   - Adăugat FPS counter, opțiune export snapshot, opțiune export logs
   - Screenshot-uri: `docs/screenshots/ui_optimized_1.png, ui_optimized_2.png, ui_optimized_3.png, ui_optimized_4.png`

3. **Pipeline end-to-end re-testat:**
   - Test complet: input → preprocess → inference → decision → output
   - Timp total: 17.6 ms (vs 17.5 ms în Etapa 5)
   
4. **Adaugare simulare stare vehicul:**
	- simulare stare vehicul in Web UI
	- conexiune prin serial la un sistem hardware

---

## 5. Modelul RN – Antrenare și Optimizare

### 5.1 Arhitectura Rețelei Neuronale

```
backbone:
  - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
  - [-1, 1, RepNCSPELAN4, [256, 128, 64, 1]] # 2
  - [-1, 1, ADown, [256]] # 3-P3/8
  - [-1, 1, RepNCSPELAN4, [512, 256, 128, 1]] # 4
  - [-1, 1, ADown, [512]] # 5-P4/16
  - [-1, 1, RepNCSPELAN4, [512, 512, 256, 1]] # 6
  - [-1, 1, ADown, [512]] # 7-P5/32
  - [-1, 1, RepNCSPELAN4, [512, 512, 256, 1]] # 8
  - [-1, 1, SPPELAN, [512, 256]] # 9
 
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]] # cat backbone P4
  - [-1, 1, RepNCSPELAN4, [512, 512, 256, 1]] # 12

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]] # cat backbone P3
  - [-1, 1, RepNCSPELAN4, [256, 512, 256, 1]] # 15 (P3/8-small)

  - [-1, 1, ADown, [256]]
  - [[-1, 12], 1, Concat, [1]] # cat head P4
  - [-1, 1, RepNCSPELAN4, [512, 512, 256, 1]] # 18 (P4/16-medium)

  - [-1, 1, ADown, [512]]
  - [[-1, 9], 1, Concat, [1]] # cat head P5
  - [-1, 1, RepNCSPELAN4, [512, 512, 256, 1]] # 21 (P5/32-large)

  - [[15, 18, 21], 1, Detect, [nc]] # Detect(P3, P4, P5)
```

**Justificare alegere arhitectură:**

Am ales această arhitectură (YOLOv9c), deoarece oferă performanță solidă în recunoașterea semnelor de circulație, și un compromis bun între acuratețe și performanță, Am testat și arhitectura YOLO26s, aceasta oferind performanță ușor mai slabă (acuratețe 94%), însă fiind un model mai mic și mai ușor de rulat. 

Am considerat că în această aplicație critică acuratețea este mai importantă decât îmbunătățirile de eficiență, astfel păstrând YOLOv9c.

### 5.2 Hiperparametri Finali (Model Optimizat - Etapa 6)

| Hiperparametru | Valoare Finală | Justificare Alegere |
|----------------|----------------|---------------------|
| Learning Rate |0.001| Valoare adecvată pentru learning rate optimizer `cos_LR` și pentru optimizer `Adam` |
| Batch Size | [ex: 32] | [ex: Compromis memorie/stabilitate pentru N=15000 samples] |
| Epochs |  8 | Compromis memorie/stabilitate  |
| Optimizer |  Adam | Este adecvat în task-urile cu multe clase, în acest caz 55 de clase, dintre care unele apar mai rar. Optimizer-ul Adam modifică dinamic learning rate-ul pentru semnele rar intalnite|
| Loss function | Classification loss (binary cross-entropy), Box Loss | Metode standard YOLO. Parametri pentru classification loss: cls=1.5. Box loss: 7.5 (default) |
| Regularizare | label_smoothing=0.1 |Prevenire overfitting și îmbunătățește generalizarea |
| Early Stopping | patience=5 | Oprire după 5 epoci fără îmbunătățire |

### 5.3 Experimente de Optimizare (minim 4 experimente)

| **Exp#** | **Modificare față de Baseline** | **Accuracy** | **F1-score** | **Timp antrenare** | **Observații** |
|----------|------------------------------------------|--------------|--------------|-------------------|----------------|
| Baseline | Configurația din Etapa 5 | 0.917 | 0.83 | 331 min (5.5 h) | Referință |
| 1 | Schimbare `cls=2.0, optimizer='AdamW', batch=8` | 0.937 | 0.86 | 212 min (3.5 h) | Imbunatatire minora la mAP50, scadere cu 36% a timpului de antrenare |
| 2 | Schimbare `label_smoothing=0.1, batch=8, optimizer='Adam', lr0=0.001` | 0.957 | 0.91 | 235 min (3.9 h) | Imbunatatiri semnificative in mAP50 si F1, cu un timp decent de antrenare |
| 3 | Schimbare `label_smoothing=0.1, batch=8, optimizer='Adam', lr0=0.001, close_mosaic=10, epochs=60` | 0.952 | 0.91 | 287 min (4.8 h) | Rezultate in marja de eroare comparativ cu exp. anterior, cu un timp mai lung de antrenare |
| 4 | Schimbare `label_smoothing=0.1, batch=8, optimizer='Adam', lr0=0.001, copy_paste=0.3` | 0.957 | 0.91 | 206 min (3.4 h)  | Cea mai buna acuratete, timp de antrenare bun |
| 5 | Schimbare arhitectura `YOLO26s`, `label_smoothing=0.1, batch=10, optimizer='Adam', lr0=0.001, copy_paste=0.3, imgsz=960` | 0.94 | 0.88 | 113 min (1.9 h)  | Un echilibru foarte bun intre timp de antrenare si acuratete |
| **FINAL** |  Schimbare `label_smoothing=0.1, batch=8, optimizer='Adam', lr0=0.001, copy_paste=0.3`  | **0.957** | **0.91** | 206 min | **Modelul folosit în producție** |

**Justificare alegere model final:**
Am ales Experimentul 4 ca model final deoarece:
- oferă valorile mAP50=0.957 și scor F1=0.91 excelente, care sunt importante pentru o recunoașterea semnelor de circulație, o aplicație safety-critical
- timp de antrenare suficient de mic (206 min)
- îmbunătățirea în performanță este dată de schimbarea parametrilor, în special `label_smoothing=0.1`, care ajută la diferențierea semnelor asemănătoare (de exemplu cele de limitare de viteză)
- timpul de antrenare este redus datorită în principal datorită `batch=8`

**Referințe fișiere:** `results/optimization_experiments.csv`, `models/optimized_model.pt`

---

## 6. Performanță Finală și Analiză Erori

### 6.1 Metrici pe Test Set (Model Optimizat)

| Metric | Valoare | Target Minim | Status |
|--------|---------|--------------|--------|
| **Accuracy** | 95.7% | ≥70% | ✓ |
| **F1-Score (Macro)** | 0.91 | ≥0.65 | ✓ |
| **Precision (Macro)** | 1.00 | - | - |
| **Recall (Macro)** | 0.98 | - | - |

**Îmbunătățire față de Baseline (Etapa 5):**

| Metric | Etapa 5 (Baseline) | Etapa 6 (Optimizat) | Îmbunătățire |
|--------|-------------------|---------------------|--------------|
| Accuracy | 91.7% | 95.7% | +4% |
| F1-Score | 0.83 | 0.91 | +9% |

**Referință fișier:** `results/final_metrics.json`

### 6.2 Confusion Matrix

**Locație:** `docs/confusion_matrix_optimized.png`

**Interpretare:**

**Clasa cu cea mai bună performanță:** `mand_straight`
- Precision: 99-100%
- Recall: 99-100%
- Explicație: Semnul _„Mergi înainte”_ prezintă o geometrie clară și unică (săgeată verticală), cu contrast ridicat și variabilitate redusă. Aceste caracteristici vizuale stabile, împreună cu o reprezentare consistentă în setul de date, conduc la o separabilitate excelentă față de celelalte clase direcționale.

**Clasa cu cea mai slabă performanță:** `forb_speed_over_30`
- Precision: 93%
- Recall: 91%
- Explicație: Semnul de _limită de viteză 30 km/h_ este frecvent confundat cu alte semne de limită de viteză apropiate (`forb_speed_over_20`, `forb_speed_over_40`). Diferențierea se bazează exclusiv pe cifre, care devin dificil de distins la rezoluții mici, în condiții de blur sau variații de iluminare.

**Confuzii principale:**
1. Clasa `forb_speed_over_30` confundată cu `forb_speed_over_40` în ~6–8% din cazuri
   - Cauză: Similaritate vizuală ridicată (formă circulară, culori identice), diferența fiind doar valoarea numerică din interior.
   - Impact industrial: O limită de viteză interpretată greșit poate conduce la setarea incorectă a vitezei țintă a vehiculului autonom, cu implicații directe asupra siguranței și conformității cu legislația rutieră.
   
2. Clasa `warn_slippery_road` confundată cu `warn_poor_road_surface` în ~5–7% din cazuri
   - Cauză: Ambele sunt semne de avertizare cu pictograme similare, având aceeași formă triunghiulară și simboluri legate de aderența carosabilului.
   - Impact industrial: Această confuzie poate duce la reacții de control suboptime (de exemplu, activarea sau dezactivarea necorespunzătoare a strategiilor de limitare a accelerației sau de control al stabilității).
### 6.3 Analiza Top 5 Erori


| **Path** | **True Label** | **Predicted** | **Confidence** | **Cauză probabilă** | **Soluție propusă** |
|-----------|----------------|---------------|----------------|---------------------|---------------------|
|  `docs/fail_examples/1.png`  | N/A|` mand_roundabout `| 0.5 | Logo cu geoemtrie si culoare similara | Cresterea valorii minime de confidence |
|  `docs/fail_examples/2.png`  | N/A |` warn_two_way_traffic `| 0.4 | Semnul real nu e in dataset, dar e tot de avertizare si are simbol similar| Cresterea valorii minime de confidence, cresterea rezolutiei la training |
| `docs/fail_examples/3.jpg` |` mand_pass_left_right` |` mand_bike_lane` | 0.71 | Semnul este murdar si are stickere pe el | Cresterea numarului datelor de training, cresterea rezolutiei imaginilor |
| `docs/fail_examples/4.jpg` | N/A |` prio_stop` | 0.70 | Culoare si forma similiara | Diversificarea background-ului imaginilor prin augmentare |
|`docs/fail_examples/5.jpg` | N/A|`info_one_way_traffic` | 0.41 | Culoare si forma similara |  Cresterea valorii minime de confidence|

### 6.4 Validare în Context Industrial

**Ce înseamnă rezultatele pentru aplicația reală:**

Când sistemul de recunoaștere a semnelor de circulație este folosit ca parte a unui sistem de asistenta pentru sofer, rezulta o reducere a accidentelor cu pana la 25%. O eroare a sistemului poate avea consecinte grave (accident).

**Pragul de acceptabilitate pentru domeniu:** F1-score ≥0.98
**Status:** Neatins (valoare curenta 0.91)
**Plan de îmbunătățire (dacă neatins):** Marire rezolutie imagini training, marire nr. date training, augmentari aditionale

---

## 7. Aplicația Software Finală

### 7.1 Modificări Implementate în Etapa 6

| **Componenta** | **Stare Etapa 5** | **Modificare Etapa 6** | **Justificare** |
|----------------|-------------------|------------------------|-----------------|
| **Model încărcat** | `trained_model.pt` | `optimized_model.pt` | +4% accuracy, +9% F1 score, -37% timp antrenare|
|**Logging** | Doar log-uri de sistem (stare aplicație)| Log-uri sistem + detecție (clasă detectată + confidence). Opțiune export log-uri |Audit trail complet |
|**Preview cameră** | Stream cameră cu overlay detecție| Adăugat FPS counter | Monitorizare performanță sistem în timp real |
|**Snapshots** | N/A| Adăugat opțiune de captura snapshot cameră (cu overlay detecție și FPS) | Captura output sistem pentru logging/debugging |
|**Simulare sistem de control vehicul** | N/A| Adăugat simulare stare vehicul (vitezometru, semnalizare) care reactioneaza la semnele detectate |Demonstratie aplicatie reala a sistemului |
|**Detectie camera acoperita** | N/A| Detectare camera acoperita si atentionare prin alarma audio si vizuala | Asigurarea sigurantei sistemului |
|**Simulare cu sistem hardware** | N/A| Transmiterea datelor legate de starea vehiculului la un sistem hardware | Evidentierea legaturii dintre software si hardware intr-o aplicatie reala |

### 7.2 Screenshot UI cu Model Optimizat

**Locație:** `docs/screenshots/inference_optimized_1.png, inference_optimized_2.png,  inference_optimized_3.png,  inference_optimized_4.png`

Screenshot-urile demonstreaza functionalitatea aplicatiei web, cu cele 4 sectiuni: detectie live simpla (prin webcam), simulator sistem control vehicul, incarcare imagine, log-uri sistem.

### 7.3 Demonstrație Funcțională End-to-End

**Locație dovadă:** `docs/demo/demo_vid.mp4`

**Fluxul demonstrat:**

| Pas | Acțiune | Rezultat Vizibil |
|-----|---------|------------------|
| 1 | Input |Stream live de la webcam |
| 2 | Inferență | Predicție afișată pe stream-ul video (mand_pass_left) |
| 3 | Decizie | Este afișată acțiunea luată de sistem (semnalizare stanga) |
| 4 | Decizie | Este afisata o situatie speciala (camera acoperita) -> alarma operator |

**Latență măsurată end-to-end:** 17.6 ms  
**Data și ora demonstrației:** 10.02.2026 23:56

---

## 8. Structura Repository-ului Final

```
proiect-rn-[nume-prenume]/
│
├── README.md                               # ← ACEST FIȘIER (Overview Final Proiect - Pe moodle la Evaluare Finala RN > Upload Livrabil 1 - Proiect RN (Aplicatie Sofware) - trebuie incarcat cu numele: NUME_Prenume_Grupa_README_Proiect_RN.md)
│
├── docs/
│   ├── etapa3_analiza_date.md              # Documentație Etapa 3
│   ├── etapa4_arhitectura_SIA.md           # Documentație Etapa 4
│   ├── etapa5_antrenare_model.md           # Documentație Etapa 5
│   ├── etapa6_optimizare_concluzii.md      # Documentație Etapa 6
│   │
│   ├── state_machine.png                   # Diagrama State Machine inițială
│   ├── state_machine_v2.png                # (opțional) Versiune actualizată Etapa 6
│   ├── confusion_matrix_optimized.png      # Confusion matrix model final
│   │
│   ├── screenshots/
│   │   ├── ui_demo.png                     # Screenshot UI schelet (Etapa 4)
│   │   ├── inference_real.png              # Inferență model antrenat (Etapa 5)
│   │   └── inference_optimized.png         # Inferență model optimizat (Etapa 6)
│   │
│   ├── demo/                               # Demonstrație funcțională end-to-end
│   │   └── demo_end_to_end.gif             # (sau .mp4 / secvență screenshots)
│   │
│   ├── results/                            # Vizualizări finale
│   │   ├── loss_curve.png                  # Grafic loss/val_loss (Etapa 5)
│   │   ├── metrics_evolution.png           # Evoluție metrici (Etapa 6)
│   │   └── learning_curves_final.png       # Curbe învățare finale
│   │
│   └── optimization/                       # Grafice comparative optimizare
│       ├── accuracy_comparison.png         # Comparație accuracy experimente
│       └── f1_comparison.png               # Comparație F1 experimente
│
├── data/
│   ├── README.md                           # Descriere detaliată dataset
│   ├── raw/                                # Date brute originale
│   ├── processed/                          # Date curățate și transformate
│   ├── generated/                          # Date originale (contribuția ≥40%)
│   ├── train/                              # Set antrenare (70%)
│   ├── validation/                         # Set validare (15%)
│   └── test/                               # Set testare (15%)
│
├── src/
│   ├── data_acquisition/                   # MODUL 1: Generare/Achiziție date
│   │   ├── README.md                       # Documentație modul
│   │   ├── generate.py                     # Script generare date originale
│   │   └── [alte scripturi achiziție]
│   │
│   ├── preprocessing/                      # Preprocesare date (Etapa 3+)
│   │   ├── data_cleaner.py                 # Curățare date
│   │   ├── feature_engineering.py          # Extragere/transformare features
│   │   ├── data_splitter.py                # Împărțire train/val/test
│   │   └── combine_datasets.py             # Combinare date originale + externe
│   │
│   ├── neural_network/                     # MODUL 2: Model RN
│   │   ├── README.md                       # Documentație arhitectură RN
│   │   ├── model.py                        # Definire arhitectură (Etapa 4)
│   │   ├── train.py                        # Script antrenare (Etapa 5)
│   │   ├── evaluate.py                     # Script evaluare metrici (Etapa 5)
│   │   ├── optimize.py                     # Script experimente optimizare (Etapa 6)
│   │   └── visualize.py                    # Generare grafice și vizualizări
│   │
│   └── app/                                # MODUL 3: UI/Web Service
│       ├── README.md                       # Instrucțiuni lansare aplicație
│       └── main.py                         # Aplicație principală
│
├── models/
│   ├── untrained_model.h5                  # Model schelet neantrenat (Etapa 4)
│   ├── trained_model.h5                    # Model antrenat baseline (Etapa 5)
│   ├── optimized_model.h5                  # Model FINAL optimizat (Etapa 6) ← FOLOSIT
│   └── final_model.onnx                    # (opțional) Export ONNX pentru deployment
│
├── results/
│   ├── training_history.csv                # Istoric antrenare - toate epocile (Etapa 5)
│   ├── test_metrics.json                   # Metrici baseline test set (Etapa 5)
│   ├── optimization_experiments.csv        # Toate experimentele optimizare (Etapa 6)
│   ├── final_metrics.json                  # Metrici finale model optimizat (Etapa 6)
│   └── error_analysis.json                 # Analiza detaliată erori (Etapa 6)
│
├── config/
│   ├── preprocessing_params.pkl            # Parametri preprocesare salvați (Etapa 3)
│   └── optimized_config.yaml               # Configurație finală model (Etapa 6)
│
├── requirements.txt                        # Dependențe Python (actualizat la fiecare etapă)
└── .gitignore                              # Fișiere excluse din versionare
```

### Legendă Progresie pe Etape

| Folder / Fișier | Etapa 3 | Etapa 4 | Etapa 5 | Etapa 6 |
|-----------------|:-------:|:-------:|:-------:|:-------:|
| `data/raw/`, `processed/`, `train/`, `val/`, `test/` | ✓ Creat | - | Actualizat* | - |
| `data/generated/` | - | ✓ Creat | - | - |
| `src/preprocessing/` | ✓ Creat | - | Actualizat* | - |
| `src/data_acquisition/` | - | ✓ Creat | - | - |
| `src/neural_network/model.py` | - | ✓ Creat | - | - |
| `src/neural_network/train.py`, `evaluate.py` | - | - | ✓ Creat | - |
| `src/neural_network/optimize.py`, `visualize.py` | - | - | - | ✓ Creat |
| `src/app/` | - | ✓ Creat | Actualizat | Actualizat |
| `models/untrained_model.*` | - | ✓ Creat | - | - |
| `models/trained_model.*` | - | - | ✓ Creat | - |
| `models/optimized_model.*` | - | - | - | ✓ Creat |
| `docs/state_machine.*` | - | ✓ Creat | - | (v2 opțional) |
| `docs/etapa3_analiza_date.md` | ✓ Creat | - | - | - |
| `docs/etapa4_arhitectura_SIA.md` | - | ✓ Creat | - | - |
| `docs/etapa5_antrenare_model.md` | - | - | ✓ Creat | - |
| `docs/etapa6_optimizare_concluzii.md` | - | - | - | ✓ Creat |
| `docs/confusion_matrix_optimized.png` | - | - | - | ✓ Creat |
| `docs/screenshots/` | - | ✓ Creat | Actualizat | Actualizat |
| `results/training_history.csv` | - | - | ✓ Creat | - |
| `results/optimization_experiments.csv` | - | - | - | ✓ Creat |
| `results/final_metrics.json` | - | - | - | ✓ Creat |
| **README.md** (acest fișier) | Draft | Actualizat | Actualizat | **FINAL** |

*\* Actualizat dacă s-au adăugat date noi în Etapa 4*

### Convenție Tag-uri Git

| Tag | Etapa | Commit Message Recomandat |
|-----|-------|---------------------------|
| `v0.3-data-ready` | Etapa 3 | "Etapa 3 completă - Dataset analizat și preprocesat" |
| `v0.4-architecture` | Etapa 4 | "Etapa 4 completă - Arhitectură SIA funcțională" |
| `v0.5-model-trained` | Etapa 5 | "Etapa 5 completă - Accuracy=X.XX, F1=X.XX" |
| `v0.6-optimized-final` | Etapa 6 | "Etapa 6 completă - Accuracy=X.XX, F1=X.XX (optimizat)" |

---

## 9. Instrucțiuni de Instalare și Rulare

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

## 10. Concluzii și Discuții

### 10.1 Evaluare Performanță vs Obiective Inițiale

| Obiectiv Definit (Secțiunea 2) | Target | Realizat | Status |
|--------------------------------|--------|----------|--------|
| Latenta pentru operarea in timp real la viteze mari ale vehiculului | <20ms | 17.6ms | ✓ |
| Recall (macro) | ≥0.97 | 0.98 | ✓ |
| Accuracy pe test set | ≥99% | 95.7% | ✗ |
| F1-Score pe test set | ≥0.98 | 0.93 | ✗ |

### 10.2 Ce NU Funcționează – Limitări Cunoscute

  - Dataset dezechilibrat - clasa 'info_crosswalk' apare de mult mai multe ori decat celelalte (>1200 imagini)
   - Datele sunt colectate de pe Street View in conditii meteo normale si cu iluminare buna
 - Probleme cu false-positives
  - Confuzie intre anumite clase, in special semnele de limita de viteza
   - Model prea mare pentru deployment pe edge device low-end (Raspberry Pi)
   - Test set nu acoperă toate condițiile din situațiile reale


### 10.3 Lecții Învățate (Top 5)

1. Augmentările specifice domeniului sunt esentiale
2. Alegerea valorilor hiperparametrilor pot creea diferente semnificative
3. Trebuie gasit un compromis intre timpul de antrenare si performanta
4. Testarea pe module si end-to-end este esentiala
5. Documentatia incrementala a economisit timp

### 10.4 Retrospectivă

**Ce ați schimba dacă ați reîncepe proiectul?**

Daca as reincepe proiectul, as face rost de mai multe date de antrenare si m-as asigura ca dataset-ul este mai echilibrat. As adauga date din mai multe surse si in capturate in diferite conditii, si as urmari in principal metode pentru reducerea fals-pozitivelor. De asemenea, as urmari posibilitatea antrenarii pe date la o rezolutie mai mare.

### 10.5 Direcții de Dezvoltare Ulterioară

| Termen | Îmbunătățire Propusă | Beneficiu Estimat |
|--------|---------------------|-------------------|
| **Short-term** (1-2 săptămâni) | Colectare mai multe date pentru anumite clase| Accuracy crescut pe clasele respective |
| **Medium-term** (1-2 luni) | Crestere rezolutie date antrenare | Accuracy crescut |
| **Long-term** | Deployment pe edge device (Raspberry Pi)  | Cost hardware redus |
| **Long-term** |  Integrare cu sistem de control vehicul complet  | Testare si dezvoltare integrata a sistemului |

---

## 11. Bibliografie

1. Radu Oprea, Traffic Signs Detection, 2024, https://www.kaggle.com/datasets/raduoprea/traffic-signs/code
2. YOLO Documentation, 2026, Data Augmentation using Ultralytics YOLO, https://docs.ultralytics.com/guides/yolo-data-augmentation/
3. Wong Kin-Yiu, YOLOv9, 2024, https://github.com/WongKinYiu/yolov9
4. Paolo Bubisutti, 2023, The 5 major challenges of ADAS and autonomous driving, https://www.eurotech.com/blog/the-5-major-challenges-of-adas-and-autonomous-driving/
5. P.  Kuppusamy et al, 2023, Traffic Sign Recognition for Autonomous Vehicle Using Optimized YOLOv7 and Convolutional Block Attention Module, https://www.sciencedirect.com/org/science/article/pii/S1546221823001297
6. Christian Szegedy et al, 2015, Rethinking the Inception Architecture for Computer Vision, https://doi.org/10.48550/arXiv.1512.00567
7. Wikipedia, 2026, Indicatoarele rutiere din România, https://ro.wikipedia.org/wiki/Indicatoarele_rutiere_din_Rom%C3%A2nia
8. Flask Documentation, 2026, https://flask.palletsprojects.com/en/stable/

---

## 12. Checklist Final (Auto-verificare înainte de predare)

### Cerințe Tehnice Obligatorii

- [X] **Accuracy ≥70%** pe test set (verificat în `results/final_metrics.json`)
- [X] **F1-Score ≥0.65** pe test set
- [X] **Contribuție ≥40% date originale** (verificabil în `data/generated/`)
- [X] **Model antrenat de la zero** (NU pre-trained fine-tuning)
- [X] **Minimum 4 experimente** de optimizare documentate (tabel în Secțiunea 5.3)
- [X] **Confusion matrix** generată și interpretată (Secțiunea 6.2)
- [X] **State Machine** definit cu minimum 4-6 stări (Secțiunea 4.2)
- [X] **Cele 3 module funcționale:** Data Logging, RN, UI (Secțiunea 4.1)
- [X] **Demonstrație end-to-end** disponibilă în `docs/demo/`

### Repository și Documentație

- [X] **README.md** complet (toate secțiunile completate cu date reale)
- [X] **4 README-uri etape** prezente în `docs/` (etapa3, etapa4, etapa5, etapa6)
- [X] **Screenshots** prezente în `docs/screenshots/`
- [X] **Structura repository** conformă cu Secțiunea 8
- [X] **requirements.txt** actualizat și funcțional
- [X] **Cod comentat** (minim 15% linii comentarii relevante)
- [X] **Toate path-urile relative** (nu absolute: `/Users/...` sau `C:\...`)

### Acces și Versionare

- [X] **Repository accesibil** cadrelor didactice RN (public sau privat cu acces)
- [X] **Tag `v0.6-optimized-final`** creat și pushed
- [X] **Commit-uri incrementale** vizibile în `git log` (nu 1 commit gigantic)
- [ ] **Fișiere mari** (>100MB) excluse sau în `.gitignore`

### Verificare Anti-Plagiat

- [X] Model antrenat **de la zero** (weights inițializate random, nu descărcate)
- [X] **Minimum 40% date originale** (nu doar subset din dataset public)
- [X] Cod propriu sau clar atribuit (surse citate în Bibliografie)

---

## Note Finale

**Versiune document:** FINAL pentru examen  
**Ultima actualizare:** 11.02.2026
**Tag Git:** `v0.6-optimized-final`

---

*Acest README servește ca documentație principală pentru Livrabilul 1 (Aplicație RN). Pentru Livrabilul 2 (Prezentare PowerPoint), consultați structura din RN_Specificatii_proiect.pdf.*
