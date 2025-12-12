

# Recunoastere semne de circulatie

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Georgescu Gabriel
**Dată actualizare:** 10.12.2025

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
  
## Despre arhitrctura RN

Am ales folosirea **YOLO** deoarece acest model este specializat pe detecția de obiecte/feature-uri și a fost folosit si in detecția de semne de circulație. A fost aleasă versiunea **YOLOv9**, deoarece aceasta oferă un echilibru între performanța detecției și resursele utilizate. Astfel, sistemul poate fi rulat si pe sisteme embedded, de exemplu un **calculator de bord** inclus într-un vehicul sau un **single-board computer** (SBC).

**NOTA:** La finalul Etapei 4, modelul NU are training pe datasetul propriu, astfel incat nu recunoaste semne de circulatie. Acesta va recunoaste doar obiecte generice, modelul fiind cel furnizat de Ultralytics.
 
##  1. Structura Repository-ului Github 

```
proiect-rn-[nume-prenume]/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── generated/  # Date originale
│   ├── train/
│   ├── validation/
│   └── test/
├── src/
│   ├── data_acquisition/
│   ├── preprocessing/  # Din Etapa 3
│   ├── neural_network/
│   └── app/  # UI schelet
├── docs/
│   ├── state_machine.*           #(state_machine.png sau state_machine.pptx sau state_machine.drawio)
│   └── [alte dovezi]
├── models/  # Untrained model
├── config/
├── README.md
├── README_Etapa3.md              # (deja existent)
├── README_Etapa4_Arhitectura_SIA.md              # ← acest fișier completat (în rădăcină)
└── requirements.txt  # Sau .lvproj
```

---

##  2. Descrierea Setului de Date


### 2.1 Sursa datelor

* **Origine:** Imagini de pe Google Maps, YouTube, alte surse publice
* **Modul de achiziție:** dataset public + generare
* **Perioada / condițiile colectării:** Generat pe 19.02.2024 (dataset sursa)

### 2.2 Caracteristicile dataset-ului

* **Număr total de observații:** 4381
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

### 4.3 Salvarea rezultatelor preprocesării

* Datele preprocesate sunt salvate direct în folderul train
* Seturi train/val/test în foldere dedicate

## 5. Nevoile rezvolate de SIA


| **Nevoie reală concretă** | **Cum o rezolvă SIA-ul vostru** | **Modul software responsabil** |
|---------------------------|--------------------------------|--------------------------------|
| Detectarea semnelor de circulatie in conditii reale si variate | Model performant, date training variate si bine augumentate -> 30% înbunătățire recunoaștere în situații complexe| RN  |
|Rularea eficientă pe diferite platforme hardware și integrarea cu hardware fizic | Folosirea OpenCV și unui model optimizat pentru o utilizare redusă a resurselor | RN + App |

## 6. Contribuția originală la setul de date

**Total observații finale:** 7634 (după Etapa 3 + Etapa 4)
**Observații originale:** 3252 (42.6%)

***Tipul contribuției:**
[ ] Date generate prin simulare fizică  
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
4. Control action based on identified sign: In functie de semnul identificat, se poate transmite un semnal de stop, viraj etc.

Tranzițiile critice sunt:
- Operare -> STOP: Daca utilizatorul inchide interfata web.
- IDLE -> ERROR: Daca nu exista o camera video conectata la sistem/s-a pierdut conexiunea.

Starea ERROR este esențială pentru că exista posibilitatea ca, din cauza vibratiilor, sa se piarda conexiunea cu camera intr-un sistem mobil autonom.
