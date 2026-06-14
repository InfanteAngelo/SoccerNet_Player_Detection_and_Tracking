# Soccer Video Understanding

In questo README si analizza la struttura della cartella contenente il progetto realizzato. Per il download del dataset principale utilizzato fare riferimento alle istruzioni presenti nel file dataset_creation_script.ipynb e nella documentazione progettuale.

---

## Struttura cartella

```text
.
├── detetction_models/                                
│   ├── others/
│   └── ft_1920_yolo11m.pt
│
├── reid_models/
│   ├── others/                             
│   └── ft_osnet_ain_x1_0_imagenet.pt                             
│
├── SIMULATOR             
│   ├── lecture_example_from_training/ 
│   │   ├── Prediction_folder/ 
│   │   └── test_set/videos/
│   │
│   ├── evaluation_helper.py                       
│   └── simulator.py
│
├── behavior_gt_generator.py           
│
├── contest_setup.ipynb             
│
├── dataset_creation_script.ipynb              
│
├── osnet_ain.py          
│
├── test_detector.ipynb           
│
├── tracker_params.yaml           
│
├── train.ipynb            
│
└── Project_Documentation.pdf
```
---

## Dettaglio struttura

### detetction_models

Cartella contenente i pesi dei modelli utlizzati per la detection. Il file ft_1920_yolo11m.pt contiene i pesi del modello migliore scelto per la competizione, mentre nella cartella others sono presenti gli altri tentativi di detection.

### reid_models

Cartella contenente i pesi dei modelli utlizzati per la re-identification. Il file ft_osnet_ain_x1_0_imagenet.pt contiene i pesi del modello migliore scelto per la competizione, mentre nella cartella others sono presenti gli altri tentativi di re-identification.

### SIMULATOR

Cartella contenente il simulatore fornito per la valutazione delle metriche finali.

### behavior_gt_generator.py

File contenente lo script per la generazione dei file behavior_gt per i vari video di test per permettere la valutazione della metrica PTBS.

### contest_setup.ipynb 

File di setup di gara che permette di fare inferenza sui video presenti all'interno della cartella del SIMULATOR.

### dataset_creation_script.ipynb  

File contenente gli script per la generazione/campionamento dei vari dataset utilizzati per addestramento/validazione dei modelli di detection/re-identification.

### osnet_ain.py 

File della libreria torchreid contenente la struttura OSNet utilizzata per addestrare il modello per la re-identification.

### test_detector.ipynb     

File contenente gli script utilizzati per ottenere metriche di valutazione relative alle varie tipologie di detector.

### tracker_params.yaml 

File contenente gli iperparametri relativi al tracker selezionato per la competizione.

### train.ipynb

File contenente gli script utilizzati per l'addestramento dei modelli di detection/re-identification.

### Project_Documentation.pdf

File contenente la documentazione del progetto: analisi del dataset e delle task, soluzioni proposte e conclusioni sulla competizione.

---