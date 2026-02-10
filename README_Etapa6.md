



# README – Etapa 6: Analiza Performanței, Optimizarea și Concluzii Finale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Georgescu  Gabriel
**Link Repository GitHub:** https://github.com/gabi200/proiect-rn
**Data predării:** 16.01.2026


## Scopul Etapei 6

Această etapă corespunde punctelor **7. Analiza performanței și optimizarea parametrilor**, **8. Analiza și agregarea rezultatelor** și **9. Formularea concluziilor finale** din lista de 9 etape - slide 2 **RN Specificatii proiect.pdf**.

**Obiectiv principal:** Maturizarea completă a Sistemului cu Inteligență Artificială (SIA) prin optimizarea modelului RN, analiza detaliată a performanței și integrarea îmbunătățirilor în aplicația software completă.

**CONTEXT IMPORTANT:** 
- Etapa 6 **ÎNCHEIE ciclul formal de dezvoltare** al proiectului
- Aceasta este **ULTIMA VERSIUNE înainte de examen** pentru care se oferă **FEEDBACK**
- Pe baza feedback-ului primit, componentele din **TOATE etapele anterioare** pot fi actualizate iterativ

**Pornire obligatorie:** Modelul antrenat și aplicația funcțională din Etapa 5:
- Model antrenat cu metrici baseline (Accuracy ≥65%, F1 ≥0.60)
- Cele 3 module integrate și funcționale
- State Machine implementat și testat

---

## MESAJ CHEIE – ÎNCHEIEREA CICLULUI DE DEZVOLTARE ȘI ITERATIVITATE

**ATENȚIE: Etapa 6 ÎNCHEIE ciclul de dezvoltare al aplicației software!**

**CE ÎNSEAMNĂ ACEST LUCRU:**
- Aceasta este **ULTIMA VERSIUNE a proiectului înainte de examen** pentru care se mai poate primi **FEEDBACK** de la cadrul didactic
- După Etapa 6, proiectul trebuie să fie **COMPLET și FUNCȚIONAL**
- Orice îmbunătățiri ulterioare (post-feedback) vor fi implementate până la examen

**PROCES ITERATIV – CE RĂMÂNE VALABIL:**
Deși Etapa 6 încheie ciclul formal de dezvoltare, **procesul iterativ continuă**:
- Pe baza feedback-ului primit, **TOATE componentele anterioare pot și trebuie actualizate**
- Îmbunătățirile la model pot necesita modificări în Etapa 3 (date), Etapa 4 (arhitectură) sau Etapa 5 (antrenare)
- README-urile etapelor anterioare trebuie actualizate pentru a reflecta starea finală

**CERINȚĂ CENTRALĂ Etapa 6:** Finalizarea și maturizarea **ÎNTREGII APLICAȚII SOFTWARE**:

1. **Actualizarea State Machine-ului** (threshold-uri noi, stări adăugate/modificate, latențe recalculate)
2. **Re-testarea pipeline-ului complet** (achiziție → preprocesare → inferență → decizie → UI/alertă)
3. **Modificări concrete în cele 3 module** (Data Logging, RN, Web Service/UI)
4. **Sincronizarea documentației** din toate etapele anterioare

**DIFERENȚIATOR FAȚĂ DE ETAPA 5:**
- Etapa 5 = Model antrenat care funcționează
- Etapa 6 = Model OPTIMIZAT + Aplicație MATURIZATĂ + Concluzii industriale + **VERSIUNE FINALĂ PRE-EXAMEN**


**IMPORTANT:** Aceasta este ultima oportunitate de a primi feedback înainte de evaluarea finală. Profitați de ea!

---

## PREREQUISITE – Verificare Etapa 5 (OBLIGATORIU)

**Înainte de a începe Etapa 6, verificați că aveți din Etapa 5:**

- [X] **Model antrenat** salvat în `models/trained_model.h5` (sau `.pt`, `.lvmodel`)
- [X] **Metrici baseline** raportate: Accuracy ≥65%, F1-score ≥0.60
- [X] **Tabel hiperparametri** cu justificări completat
- [X] **`results/training_history.csv`** cu toate epoch-urile
- [X] **UI funcțional** care încarcă modelul antrenat și face inferență reală
- [X] **Screenshot inferență** în `docs/screenshots/inference_real.png`
- [X] **State Machine** implementat conform definiției din Etapa 4

**Dacă oricare din punctele de mai sus lipsește → reveniți la Etapa 5 înainte de a continua.**

---

## Cerințe

Completați **TOATE** punctele următoare:

1. **Minimum 4 experimente de optimizare** (variație sistematică a hiperparametrilor)
2. **Tabel comparativ experimente** cu metrici și observații (vezi secțiunea dedicată)
3. **Confusion Matrix** generată și analizată
4. **Analiza detaliată a 5 exemple greșite** cu explicații cauzale
5. **Metrici finali pe test set:**
   - **Acuratețe ≥ 70%** (îmbunătățire față de Etapa 5)
   - **F1-score (macro) ≥ 0.65**
6. **Salvare model optimizat** în `models/optimized_model.h5` (sau `.pt`, `.lvmodel`)
7. **Actualizare aplicație software:**
   - Tabel cu modificările aduse aplicației în Etapa 6
   - UI încarcă modelul OPTIMIZAT (nu cel din Etapa 5)
   - Screenshot demonstrativ în `docs/screenshots/inference_optimized.png`
8. **Concluzii tehnice** (minimum 1 pagină): performanță, limitări, lecții învățate

#### Tabel Experimente de Optimizare

Documentați **minimum 4 experimente** cu variații sistematice:

| **Exp#** | **Modificare față de Baseline (Etapa 5)** | **Accuracy** | **F1-score** | **Timp antrenare** | **Observații** |
|----------|------------------------------------------|--------------|--------------|-------------------|----------------|
| Baseline | Configurația din Etapa 5 | 0.917 | 0.83 | 331 min (5.5 h) | Referință |
| 1 | Schimbare `cls=2.0, optimizer='AdamW', batch=8` | 0.937 | 0.86 | 212 min (3.5 h) | Imbunatatire minora la mAP50, scadere cu 36% a timpului de antrenare |
| 2 | Schimbare `label_smoothing=0.1, batch=8, optimizer='Adam', lr0=0.001` | 0.957 | 0.91 | 235 min (3.9 h) | Imbunatatiri semnificative in mAP50 si F1, cu un timp decent de antrenare |
| 3 | Schimbare `label_smoothing=0.1, batch=8, optimizer='Adam', lr0=0.001, close_mosaic=10, epochs=60` | 0.952 | 0.91 | 287 min (4.8 h) | Rezultate in marja de eroare comparativ cu exp. anterior, cu un timp mai lung de antrenare |
| 4 | Schimbare `label_smoothing=0.1, batch=8, optimizer='Adam', lr0=0.001, copy_paste=0.3` | 0.957 | 0.91 | 206 min (3.4 h)  | Cea mai buna acuratete, timp de antrenare bun |
| 5 | Schimbare arhitectura `YOLO26s`, `label_smoothing=0.1, batch=10, optimizer='Adam', lr0=0.001, copy_paste=0.3, imgsz=960` | 0.94 | 0.88 | 113 min (1.9 h)  | Un echilibru foarte bun intre timp de antrenare si acuratete |


**Justificare alegere configurație finală:**

Am ales Experimentul 4 ca model final:
- oferă valorile mAP50=0.957 și scor F1=0.91 excelente, care sunt importante pentru o recunoașterea semnelor de circulație, o aplicație safety-critical
- timp de antrenare suficient de mic (206 min)
- îmbunătățirea în performanță este dată de schimbarea parametrilor, în special `label_smoothing=0.1`, care ajută la diferențierea semnelor asemănătoare (de exemplu cele de limitare de viteză)
- timpul de antrenare este redus datorită în principal datorită `batch=8`

---

## 1. Actualizarea Aplicației Software în Etapa 6 

**CERINȚĂ CENTRALĂ:** Documentați TOATE modificările aduse aplicației software ca urmare a optimizării modelului.

### Tabel Modificări Aplicație Software

| **Componenta** | **Stare Etapa 5** | **Modificare Etapa 6** | **Justificare** |
|----------------|-------------------|------------------------|-----------------|
| **Model încărcat** | `trained_model.pt` | `optimized_model.pt` | +4% accuracy, +9% F1 score, -37% timp antrenare|
 |**Logging** | Doar log-uri de sistem (stare aplicație)| Log-uri sistem + detecție (clasă detectată + confidence). Opțiune export log-uri |Audit trail complet |
 |**Preview cameră** | Stream cameră cu overlay detecție| Adăugat FPS counter | Monitorizare performanță sistem în timp real |
  |**Snapshots** | N/A| Adăugat opțiune de captura snapshot cameră (cu overlay detecție și FPS) | Captura output sistem pentru logging/debugging |
  |**Simulare sistem de control vehicul** | N/A| Adăugat simulare stare vehicul (vitezometru, semnalizare) care reactioneaza la semnele detectate |Demonstratie aplicatie reala a sistemului |
   |**Detectie camera acoperita** | N/A| Detectare camera acoperita si atentionare prin alarma audio si vizuala | Asigurarea sigurantei sistemului |
   |**Simulare cu sistem hardware** | N/A| Transmiterea datelor legate de starea vehiculului la un sistem hardware | Evidentierea legaturii dintre software si hardware intr-o aplicatie reala |

### Modificări concrete aduse în Etapa 6:

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

## 2. Analiza Detaliată a Performanței

### 2.1 Confusion Matrix și Interpretare

**Locație:** `docs/confusion_matrix_optimized.png`

**Analiză obligatorie (completați):**

### Interpretare Confusion Matrix:

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


### 2.2 Analiza Detaliată a 5 Exemple Greșite

Selectați și analizați **minimum 5 exemple greșite** de pe test set:

| **Path** | **True Label** | **Predicted** | **Confidence** | **Cauză probabilă** | **Soluție propusă** |
|-----------|----------------|---------------|----------------|---------------------|---------------------|
|  `docs/fail_examples/1.png`  | N/A|` mand_roundabout `| 0.5 | Logo cu geoemtrie si culoare similara | Cresterea valorii minime de confidence |
|  `docs/fail_examples/2.png`  | N/A |` warn_two_way_traffic `| 0.4 | Semnul real nu e in dataset, dar e tot de avertizare si are simbol similar| Cresterea valorii minime de confidence, cresterea rezolutiei la training |
| `docs/fail_examples/3.jpg` |` mand_pass_left_right` |` mand_bike_lane` | 0.71 | Semnul este murdar si are stickere pe el | Cresterea numarului datelor de training, cresterea rezolutiei imaginilor |
| `docs/fail_examples/4.jpg` | N/A |` prio_stop` | 0.70 | Culoare si forma similiara | Diversificarea background-ului imaginilor prin augmentare |
|`docs/fail_examples/5.jpg` | N/A|`info_one_way_traffic` | 0.41 | Culoare si forma similara |  Cresterea valorii minime de confidence|

**Analiză detaliată per exemplu (scrieți pentru fiecare):**

### Exemplu 1 - Fals pozitiv pentru `mand_roundabout`

**Context:** Logo pe o masina care are o forma circulara si contur albastru
**Output RN:**` mand_roundabout 0.5 `

**Analiză:**
Imaginea surprinde un logo pe o masina, care este clar vizibil, avand forma circulara si culoare albastra.

**Implicație industrială:**
Detectarea fals-pozitivă a unui semn de sens giratoriu poate duce la încetinirea fără motiv a vehicului. 

**Soluție:**
1. Mărirea valorii minime de confidence
2. Diversificarea background-urilor pentru context adițional

### Exemplu 2 - Fals pozitiv pentru `warn_two_way_traffic`

**Context:** Semn de ingustarea drumului pe partea dreapta, care nu e in dataset
**Output RN:**` warn_two_way_traffic 0.4 `

**Analiză:**
Imaginea surprinde un semn de circulatie de avertizare, care totusi nu e in dataset (ingustarea drumului pe partea dreapta). Acesta a fost confundat cu semnul de avertizare pentru circulație in ambele sensuri, deoarece ambele sunt indicatoare de avertizare, si au geometria simbolurilor similara.

**Implicație industrială:**
Detectarea fals-pozitivă a unui semn de circulație în ambele sensuri poate cauza probleme dacă este detectat într-o zonă cu sens unic, încetinind vehiculul deoarece acesta se poate aștepta la trafic din sens opus.

**Soluție:**
1. Mărirea valorii minime de confidence
2. Cresterea rezolutiei datelor de trainining 

### Exemplu 3 - `mand_pass_left_right` confundat cu ` mand_bike_lane`

**Context:** Semn de ocolire prin stanga sau dreapta confundat cu semnul pentru pista de biciclete
**Output RN:**` mand_bike_line 0.71 `

**Analiză:**
Imaginea surprinde un semn de ocolire prin stanga sau dreapta care este acoperit de desene si stickere, astfel geometria lui nefiind clara, creeand o vaga forma similara cu cea a semnului de pista de biciclete.

**Implicație industrială:**
Nedetectarea corectă a semnului de ocolire poate avea consecințe severe doarece acesta marcheaza de obicei un obstacol, iar confuzia cu indicatorul pentru pista de biciclete poate cauza vehiculul sa schimbe banda fara motiv,  evitand "banda pentru biciclete".

**Soluție:**
1. Marirea numarului datelor de training 
2. Cresterea rezolutiei datelor de trainining 

### Exemplu 4 - Fals pozitiv pentru ` prio_stop `

**Context:** Logo cu culoare si geometrie similara cu semnul STOP
**Output RN:**` prio_stop 0.70 `

**Analiză:**
Imaginea surprinde un logo cu geometrie similara (hexagon) si schema culori (rosu si alb) cu un semn STOP. 

**Implicație industrială:**
Detectarea fals-pozitiva a semnului STOP este o eroare critica deoarece poate cauza vehiculul sa se opreasca brusc in timp ce merge la viteza mare, existand riscul de cauzarea accidentelor.

**Soluție:**
1. Diversificarea background-ului imaginilor, augmentare suplimentara
2. Cresterea rezolutiei datelor de trainining 

### Exemplu 5 - Fals pozitiv pentru ` info_one_way_traffic `

**Context:** Logo cu culoare si geometrie similara cu semnul de sens unic
**Output RN:**` info_one_way_traffic 0.41 `

**Analiză:**
Imaginea surprinde un logo cu geometrie similara (patrat) si culori similare (albastru si alb). 

**Implicație industrială:**
Detectarea fals-pozitiva a semnului de sens unic poate avea consecinte severe doarece vehiculul nu se va astepta sa existe vehicule care vin din sensul opus, astfel crescand potentialul de accidente.

**Soluție:**
1. Diversificarea background-ului imaginilor, augmentare suplimentara
2. Cresterea rezolutiei datelor de trainining 

---

## 3. Optimizarea Parametrilor și Experimentare

### 3.1 Strategia de Optimizare

Descrieți strategia folosită pentru optimizare:


### Strategie de optimizare adoptată:

**Abordare:** Manual

**Axe de optimizare explorate:**
1. **Arhitectură:** schimbare de la `YOLO11m` la `YOLO26s`
2. **Regularizare:** `label_smoothing=0.1`
3. **Learning rate:** schimbare `lr0=0.001` (initial learning rate), optimizer Adam
4. **Augmentări:** `copy_paste=0.3` (context augumentation)
5. **Batch size:** micșorare batch size la 8

**Criteriu de selecție model final:** F1-score si mAP50 maxim, cu timp de antrenare rezonabil

**Buget computațional:** 5 experimente, 23 ore GPU


### 3.2 Grafice Comparative

Generați și salvați în `docs/optimization/`:
- `accuracy_comparison.png` - Accuracy per experiment
- `f1_comparison.png` - F1-score per experiment
- `learning_curves_best.png` - Loss și Accuracy pentru modelul final

### 3.3 Raport Final Optimizare

### Raport Final Optimizare

**Model baseline (Etapa 5):**
- Accuracy: 0.917
- F1-score: 0.83
- Latență: 17.5 ms

**Model optimizat (Etapa 6):**
- Accuracy: 0.957 (+4%)
- F1-score: 0.91 (+9%)
- Latență: 17.6 ms (diferență neglijabilă, margin of error)

**Configurație finală aleasă:**
- Arhitectură: YOLO11m
- Learning rate: 0.001 cu `cos_lr`
- Batch size: 8
- Regularizare: `label_smoothing=0.1`
- Augmentări: `hsv_h=0.015, hsv_s=0.6, hsv_v=0.5, scale=0.8, shear=2.0, perspective=0.001, fliplr=0, degrees=3, copy_paste=0.3`
- Epoci: 50 

**Îmbunătățiri cheie:**
1. Schimbare learning rate și optimizer
2. Adăugare parametri de regualizare (`label_smoothing`)
3. Scădere batch size

---

## 4. Agregarea Rezultatelor și Vizualizări

### 4.1 Tabel Sumar Rezultate Finale

| **Metrică** | **Etapa 4** | **Etapa 5** | **Etapa 6** | **Target Industrial** | **Status** |
|-------------|-------------|-------------|-------------|----------------------|------------|
| Accuracy | N/A | 91.7% | 95.7% | ≥99% | Aproape |
| F1-score (macro) |N/A | 0.83 | 0.91 | ≥0.98 | Aproape |
| Recall (macro) | N/A | 0.97 | 0.98 | ≥0.97 | OK |
| Latență inferență | N/A | 17.5ms | 17.6ms | ≤50ms | OK |

### 4.2 Vizualizări Obligatorii

Salvați în `docs/results/`:

- [X] `confusion_matrix_optimized.png` - Confusion matrix model final
- [X] `learning_curves_final.png` - Loss și accuracy vs. epochs
- [X] `metrics_evolution.png` - Evoluție metrici Etapa 4 → 5 → 6
- [X] `example_predictions.png` - Grid cu 9+ exemple (correct + greșite)

---

## 5. Concluzii Finale și Lecții Învățate

**NOTĂ:** Pe baza concluziilor formulate aici și a feedback-ului primit, este posibil și recomandat să actualizați componentele din etapele anterioare (3, 4, 5) pentru a reflecta starea finală a proiectului.

### 5.1 Evaluarea Performanței Finale


### Evaluare sintetică a proiectului

**Obiective atinse:**
- [X] Model RN funcțional cu accuracy 95.7% pe test set
- [X] Integrare completă în aplicație software (3 module)
- [X] State Machine implementat și actualizat
- [X] Pipeline end-to-end testat și documentat
- [X] UI demonstrativ cu inferență reală
- [X] Documentație completă pe toate etapele
- [X] Demo cu sistem hardware pentru simulare vehicul

**Obiective parțial atinse:**
- Fals pozitive în diferite situații, care pot cauza situații periculoase
- Potențială confuzie între anumite semne (limite de viteză)

**Obiective neatinse:**
- Deployment pe un sistem embedded (de ex. Raspberry Pi)


### 5.2 Limitări Identificate


### Limitări tehnice ale sistemului

1. **Limitări date:**
   - Dataset dezechilibrat - clasa 'info_crosswalk' apare de mult mai multe ori decat celelalte (>1200 imagini)
   - Datele sunt colectate de pe Street View in conditii meteo normale si cu iluminare buna

2. **Limitări model:**
   - Probleme cu false-positives
   - Confuzie intre anumite clase, in special semnele de limita de viteza

3. **Limitări infrastructură:**
   - Model prea mare pentru deployment pe edge device

4. **Limitări validare:**
   - Test set nu acoperă toate condițiile din situațiile reale

### 5.3 Direcții de Cercetare și Dezvoltare


### Direcții viitoare de dezvoltare

**Pe termen scurt (1-3 luni):**
1. Colectare mai multe date pentru anumite clase
2. Folosire rezoluție mai mare pentru training

**Pe termen mediu (3-6 luni):**
3. Deployment pe sistem edge (Raspberry Pi, Jetson)
4. Integrare cu sistem de control vehicul

### 5.4 Lecții Învățate


### Lecții învățate pe parcursul proiectului

**Tehnice:**
1. Augmentările specifice domeniului sunt esentiale
2. Alegerea valorilor hiperparametrilor pot creea diferente semnificative

**Proces:**
3. Trebuie gasit un compromis intre timpul de antrenare si performanta
4. Testarea pe module si end-to-end este esentiala
5. Documentatia incrementala a economisit timp

**Colaborare:**
6. Feedback de la experți domeniu a ghidat selecția features

### 5.5 Plan Post-Feedback (ULTIMA ITERAȚIE ÎNAINTE DE EXAMEN)

### Plan de acțiune după primirea feedback-ului

**ATENȚIE:** Etapa 6 este ULTIMA VERSIUNE pentru care se oferă feedback!
Implementați toate corecțiile înainte de examen.

După primirea feedback-ului de la evaluatori:
- Adaugare simulare sistem de control vehicul in Web UI
- Sistem hardware de simulare

**Timeline:** Implementare corecții până la data examen
**Commit final:** `"Versiune finală examen - toate corecțiile implementate"`
**Tag final:** `git tag -a v1.0-final-exam -m "Versiune finală pentru examen"`

---

## Structura Repository-ului la Finalul Etapei 6

**Structură COMPLETĂ și FINALĂ:**

```
proiect-rn-[prenume-nume]/
├── README.md                               # Overview general proiect (FINAL)
├── etapa3_analiza_date.md                  # Din Etapa 3
├── etapa4_arhitectura_sia.md               # Din Etapa 4
├── etapa5_antrenare_model.md               # Din Etapa 5
├── etapa6_optimizare_concluzii.md          # ← ACEST FIȘIER (completat)
│
├── docs/
│   ├── state_machine.png                   # Din Etapa 4
│   ├── state_machine_v2.png                # NOU - Actualizat (dacă modificat)
│   ├── loss_curve.png                      # Din Etapa 5
│   ├── confusion_matrix_optimized.png      # NOU - OBLIGATORIU
│   ├── results/                            # NOU - Folder vizualizări
│   │   ├── metrics_evolution.png           # NOU - Evoluție Etapa 4→5→6
│   │   ├── learning_curves_final.png       # NOU - Model optimizat
│   │   └── example_predictions.png         # NOU - Grid exemple
│   ├── optimization/                       # NOU - Grafice optimizare
│   │   ├── accuracy_comparison.png
│   │   └── f1_comparison.png
│   └── screenshots/
│       ├── ui_demo.png                     # Din Etapa 4
│       ├── inference_real.png              # Din Etapa 5
│       └── inference_optimized.png         # NOU - OBLIGATORIU
│
├── data/                                   # Din Etapa 3-5 (NESCHIMBAT)
│   ├── raw/
│   ├── generated/
│   ├── processed/
│   ├── train/
│   ├── validation/
│   └── test/
│
├── src/
│   ├── data_acquisition/                   # Din Etapa 4
│   ├── preprocessing/                      # Din Etapa 3
│   ├── neural_network/
│   │   ├── model.py                        # Din Etapa 4
│   │   ├── train.py                        # Din Etapa 5
│   │   ├── evaluate.py                     # Din Etapa 5
│   │   └── optimize.py                     # NOU - Script optimizare/tuning
│   └── app/
│       └── main.py                         # ACTUALIZAT - încarcă model OPTIMIZAT
│
├── models/
│   ├── untrained_model.h5                  # Din Etapa 4
│   ├── trained_model.h5                    # Din Etapa 5
│   ├── optimized_model.h5                  # NOU - OBLIGATORIU
│
├── results/
│   ├── training_history.csv                # Din Etapa 5
│   ├── test_metrics.json                   # Din Etapa 5
│   ├── optimization_experiments.csv        # NOU - OBLIGATORIU
│   ├── final_metrics.json                  # NOU - Metrici model optimizat
│
├── config/
│   ├── preprocessing_params.pkl            # Din Etapa 3
│   └── optimized_config.yaml               # NOU - Config model final
│
├── requirements.txt                        # Actualizat
└── .gitignore
```

**Diferențe față de Etapa 5:**
- Adăugat `etapa6_optimizare_concluzii.md` (acest fișier)
- Adăugat `docs/confusion_matrix_optimized.png` - OBLIGATORIU
- Adăugat `docs/results/` cu vizualizări finale
- Adăugat `docs/optimization/` cu grafice comparative
- Adăugat `docs/screenshots/inference_optimized.png` - OBLIGATORIU
- Adăugat `models/optimized_model.h5` - OBLIGATORIU
- Adăugat `results/optimization_experiments.csv` - OBLIGATORIU
- Adăugat `results/final_metrics.json` - metrici finale
- Adăugat `src/neural_network/optimize.py` - script optimizare
- Actualizat `src/app/main.py` să încarce model OPTIMIZAT
- (Opțional) `docs/state_machine_v2.png` dacă s-au făcut modificări

---

## Checklist Final – Bifați Totul Înainte de Predare

### Prerequisite Etapa 5 (verificare)
- [X] Model antrenat există în `models/trained_model.h5`
- [X] Metrici baseline raportate (Accuracy ≥65%, F1 ≥0.60)
- [X] UI funcțional cu model antrenat
- [X] State Machine implementat

### Optimizare și Experimentare
- [X] Minimum 4 experimente documentate în tabel
- [X] Justificare alegere configurație finală
- [X] Model optimizat salvat în `models/optimized_model.h5`
- [X] Metrici finale: **Accuracy ≥70%**, **F1 ≥0.65**
- [X] `results/optimization_experiments.csv` cu toate experimentele
- [X] `results/final_metrics.json` cu metrici model optimizat

### Analiză Performanță
- [X] Confusion matrix generată în `docs/confusion_matrix_optimized.png`
- [X] Analiză interpretare confusion matrix completată în README
- [X] Minimum 5 exemple greșite analizate detaliat
- [X] Implicații industriale documentate (cost FN vs FP)

### Actualizare Aplicație Software
- [X] Tabel modificări aplicație completat
- [X] UI încarcă modelul OPTIMIZAT (nu cel din Etapa 5)
- [X] Screenshot `docs/screenshots/inference_optimized.png`
- [X] Pipeline end-to-end re-testat și funcțional
- [X] (Dacă aplicabil) State Machine actualizat și documentat

### Concluzii
- [X] Secțiune evaluare performanță finală completată
- [X] Limitări identificate și documentate
- [X] Lecții învățate (minimum 5)
- [X] Plan post-feedback scris

### Verificări Tehnice
- [X] `requirements.txt` actualizat
- [X] Toate path-urile RELATIVE
- [X] Cod nou comentat (minimum 15%)
- [X] `git log` arată commit-uri incrementale
- [X] Verificare anti-plagiat respectată

### Verificare Actualizare Etape Anterioare (ITERATIVITATE)
- [X] README Etapa 3 actualizat (dacă s-au modificat date/preprocesare)
- [X] README Etapa 4 actualizat (dacă s-a modificat arhitectura/State Machine)
- [X] README Etapa 5 actualizat (dacă s-au modificat parametri antrenare)
- [X] `docs/state_machine.*` actualizat pentru a reflecta versiunea finală
- [X] Toate fișierele de configurare sincronizate cu modelul optimizat

### Pre-Predare
- [X] `etapa6_optimizare_concluzii.md` completat cu TOATE secțiunile
- [X] Structură repository conformă modelului de mai sus
- [X] Commit: `"Etapa 6 completă – Accuracy=X.XX, F1=X.XX (optimizat)"`
- [X] Tag: `git tag -a v0.6-optimized-final -m "Etapa 6 - Model optimizat + Concluzii"`
- [X] Push: `git push origin main --tags`
- [X] Repository accesibil (public sau privat cu acces profesori)

---

## Livrabile Obligatorii

Asigurați-vă că următoarele fișiere există și sunt completate:

1. **`etapa6_optimizare_concluzii.md`** (acest fișier) cu:
   - Tabel experimente optimizare (minimum 4)
   - Tabel modificări aplicație software
   - Analiză confusion matrix
   - Analiză 5 exemple greșite
   - Concluzii și lecții învățate

2. **`models/optimized_model.h5`** (sau `.pt`, `.lvmodel`) - model optimizat funcțional

3. **`results/optimization_experiments.csv`** - toate experimentele
```

4. **`results/final_metrics.json`** - metrici finale:

Exemplu:
```json
{
  "model": "optimized_model.h5",
  "test_accuracy": 0.8123,
  "test_f1_macro": 0.7734,
  "test_precision_macro": 0.7891,
  "test_recall_macro": 0.7612,
  "false_negative_rate": 0.05,
  "false_positive_rate": 0.12,
  "inference_latency_ms": 35,
  "improvement_vs_baseline": {
    "accuracy": "+9.2%",
    "f1_score": "+9.3%",
    "latency": "-27%"
  }
}
```

5. **`docs/confusion_matrix_optimized.png`** - confusion matrix model final

6. **`docs/screenshots/inference_optimized.png`** - demonstrație UI cu model optimizat

---

## Predare și Contact

**Predarea se face prin:**
1. Commit pe GitHub: `"Etapa 6 completă – Accuracy=X.XX, F1=X.XX (optimizat)"`
2. Tag: `git tag -a v0.6-optimized-final -m "Etapa 6 - Model optimizat + Concluzii"`
3. Push: `git push origin main --tags`

---

**REMINDER:** Aceasta a fost ultima versiune pentru feedback. Următoarea predare este **VERSIUNEA FINALĂ PENTRU EXAMEN**!
