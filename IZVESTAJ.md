# Izveštaj: PTB-XL Multi-Lead EKG Klasifikacija

## 1. Cilj projekta

Multi-label klasifikacija 12-kanalnih EKG snimaka iz PTB-XL skupa podataka u
5 dijagnostičkih superklasa: **NORM** (normalan nalaz), **MI** (infarkt
miokarda), **STTC** (ST/T promene), **CD** (poremećaji provođenja) i **HYP**
(hipertrofija). Pacijent može imati više dijagnoza istovremeno, pa je zadatak
multi-label (5 nezavisnih binarnih odluka), ne multi-class.

Projekat je nastavak ranije MIT-BIH faze - single-lead,
single-beat, multi-class klasifikacija otkucaja - proširen na 12-kanalne,
10-sekundne snimke sa više istovremenih dijagnoza.

## 2. Podaci

- **Izvor:** PTB-XL v1.0.3 (PhysioNet), ~21 400 snimaka sa prepoznatom
  dijagnostičkom superklasom.
- **Distribucija klasa** (multi-label, procenti sumiraju >100%): NORM 44.5%,
  MI 25.6%, STTC 24.5%, CD 22.9%, HYP 12.4% — HYP je izraženo najređa klasa.
- **Zvanični split** (kolona `strat_fold`, 1-10): foldovi 1-8 trening, fold 9
  validacija, fold 10 test - bez nasumičnog mešanja, da se izbegne curenje
  podataka istog pacijenta između skupova.
- **Sampling rate:** podržano 100Hz (1000 tačaka) i 500Hz (5000 tačaka),
  podesivo u `config.yaml`. Keš predobrađenih podataka je odvojen po
  sampling rate-u (`ptbxl_cache_{sr}hz.npz`) da promena u config-u ne bi
  tiho učitala keš napravljen za drugi rate.

## 3. Pipeline obrade podataka (`src/data_loading.py`, `src/preprocessing.py`)

- Mapiranje SCP-ECG dijagnostičkih kodova u 5 superklasa preko
  `scp_statements.csv`.
- Predobrada signala: uklanjanje baseline wander-a (high-pass 0.5Hz),
  uklanjanje visokofrekventnog šuma (low-pass 40Hz), z-score normalizacija
  po kanalu.
- **Meke (confidence-weighted) labele:** pored standardnih tvrdih 0/1
  labela, računaju se i "meke" labele na osnovu PTB-XL `likelihood`
  anotacije (0-100%, pouzdanost dijagnoze). Konvencija: `likelihood=0` znači
  "pouzdanost nije navedena" (ne "odsutno") - potvrđeno na podacima da ~47.5%
  svih anotacija ima `likelihood=0`, pa se tretira kao puna pouzdanost.
  **Meke labele se koriste samo za trening loss** (label smoothing prema
  stvarnoj pouzdanosti anotacije); validacija i test ostaju na tvrdim 0/1
  labelama, jer metrike (AUC, F1) zahtevaju diskretan ground truth.
- Memorijska optimizacija: signali se učitavaju direktno u unapred alocirane
  `float32` nizove (ne Python liste pa konverzija) - `wfdb.rdsamp` vraća
  `float64`, pa bi lista + naknadna konverzija privremeno držale dve kopije
  celog dataseta u memoriji (posebno bitno na 500Hz).

## 4. Arhitekture modela (`src/models.py`)

Sve dele `ResidualBlock1D` (rezidualni 1D konv. blok, nastavak MIT-BIH
`ResidualBlock1D`-a) sa ugrađenom **Squeeze-and-Excitation (SE)** kanalskom
pažnjom - mreža uči koji feature kanal da pojača/priguši.

| Model | Opis |
|---|---|
| `PTBXL_CNN` |  3 rezidualna bloka + max-pool između, `AdaptiveAvgPool1d` na kraju. Najjednostavniji, trenutno najbolji pojedinačni model. |
| `PTBXL_BiGRU` | 3 rezidualna bloka (stride=2) + max-pool, pa BiGRU, pa **max pooling** kroz vreme 
| `PTBXL_Transformer` | 4 rezidualna bloka (stride=2) → Transformer encoder (self-attention, `num_layers=2`) → **naučeni attention-pooling** (`AttentionPooling1D`, zamena za prost prosek) kroz vreme. |

Svi modeli rade i na 100Hz i 500Hz ulazu (arhitektura je dužinski agnostička
preko adaptivnog/global poolinga).

## 5. Loss funkcija i strategije za disbalans klasa

- **Focal loss** (`src/losses.py`) sa **per-klasa alpha vektorom** (ne
  skalar) - `alpha_t = alpha*targets + (1-alpha)*(1-targets)`, gde je
  `alpha` tenzor registrovan kao buffer (prati `.to(device)`). Trenutne
  vrednosti (`[0.1363, 0.2371, 0.2477, 0.2647, 0.4895]` za
  `[NORM,MI,STTC,CD,HYP]`) prate obrazac `alpha_i ∝ 1/prevalenca_i`.
- **WeightedRandomSampler** (`napravi_weighted_sampler`) - svaki uzorak
  dobija težinu jednaku inverznoj frekvenciji svoje **najređe pozitivne**
  klase (multi-label: uzorak sa i NORM i HYP dobija HYP-ovu težinu).
  Komplementaran sa alpha vektorom, ne zamena.

## 6. Trening infrastruktura (`src/train.py`)

- **Zvanični split trening** (`train.py`) i **k-fold cross-validation**
  (`train_kfold.py`) dele istu `treniraj_model()` funkciju.
- Config-driven izbor: **optimizer** (`adam`/`adamw`/`sgd`), **LR scheduler**
  (toggle-abilan, `CosineAnnealingLR` opciono sa **linear warmup** fazom
  preko `SequentialLR` - motivisano Transformer arhitekturama koje su
  osetljivije na visok LR na početku treninga), **gradient clipping**
  (`clip_grad_norm_`, protiv exploding gradijenata kod BiGRU/Transformer).
- Checkpoint se čuva po najboljem **validacionom macro AUC** (threshold-
  independent metrika - robusnija od F1@fiksni-prag, koji se pokazao
  nepouzdanim usled degenerisanih rešenja tokom ranog debagovanja).
- **K-fold protokol:** foldovi 1-9 rotiraju kao validacija (`kfold.foldovi`
  u config-u, podesiva lista - može se skratiti za jeftiniji ansambl), fold
  10 nikad nije viđen u treningu, koristi se isključivo kao zajednički test
  skup za ansambl.

## 7. Evaluacija (`src/evaluate.py`, `src/evaluate_kfold.py`)

- **Per-klasa tuning praga:** umesto fiksnog praga 0.5, `precision_recall_curve`
  na **validacionom** setu nalazi prag koji maksimizuje F1 za svaku klasu
  posebno - značajno poboljšava F1 bez ikakve promene modela (npr. macro F1
  0.57 → 0.70 samo od boljeg praga).
- **K-fold ansambl** (`evaluate_kfold.py`): soft-voting (usrednjavanje
  verovatnoća) svih fold modela na test setu; pragovi po klasi biraju se na
  **out-of-fold** predikcijama (svaki trening uzorak ocenjen samo modelom
  koji ga nije video), izbegavajući curenje i iz test i iz trening podataka.

## 8. Ključni eksperimenti 

| Eksperiment | Rezultat |
|---|---|
| Fiksiranje `alpha` bug-ova (0 i 1 su degenerisani slučajevi) | Kritičan preduslov - bez ovoga model uči da uvek predviđa isto |
| SE kanalska pažnja | Zadržano, nije izolovano škodilo |
| Kernel widening (5→11/15) | Bez jasnog dobitka na CNN-u |
| Dilated convolution (dilation 2/4/8) | Mešoviti rezultati, CD (ciljana klasa) se nije pomerio - vraćeno na baseline |
| Weighted sampler | Pomaže CNN-u (+0.003 macro AUC) |
| Transformer poboljšanja (attention pool, dublji encoder, LR warmup) | Nije promenilo rang - Transformer ostaje najslabiji pojedinačni model, verovatno zbog nedovoljno podataka za self-attention (slabija induktivna pristrasnost od CNN/GRU) |
| Grad-CAM analiza (CNN) | Model se fokusira na fiziološki smislene delove signala - QRS kompleks za MI/CD (kod CD-a šire, u skladu sa proširenim QRS-om), ST/T segment za STTC i HYP; potvrđuje da mreža ne uči artefakte |
| Lead occlusion analiza (HYP, 262 primera) | Model dominantno koristi **V1** (deo Sokolow-Lyon kriterijuma), dok **I, V4, V5** deluju kao suzbijajući (ne pozitivan) dokaz, a **V6** je zanemaren - model ne primenjuje pun bilateralni kriterijum (V1 + V5/V6), verovatan uzrok slabog HYP recall-a |
| `KlasneGlave` (per-klasa MLP glave umesto deljenog fc sloja) | Testirano na sve tri arhitekture - **nije dalo bolje rezultate**, vraćeno na deljeni finalni sloj |
| Random lead masking augmentacija | `p=1.0` (uvek aktivna) je **prejaka** - model nikad ne vidi pun 12-kanalni signal, pogoršala HYP recall; treba umerenija verovatnoća (~0.3), još nije potvrđeno da pomaže |
| Analiza CSV trening logova | Transformer ima upadljivo niži final train_loss od CNN-a (0.016 vs ~0.04) ali i niži val AUC - klasičan potpis overfitting-a, potvrđuje "gladan podataka" nalaz; jedan CNN k-fold trening (fold6) je udario u limit od 40 epoha pre nego što se early stopping aktivirao - nije stigao da se potpuno konvergira |
| **K-fold ansambl (CNN, 6 foldova)** | Macro AUC **0.9170** - **novi najbolji rezultat u projektu**, bolji od pojedinačnog CNN-a na SVIH 5 klasa (+0.009 do +0.015 AUC po klasi), i bolji od ranijeg Transformer ansambla (0.9062) |
| K-fold ansambl (Transformer, 5 foldova) | Macro AUC 0.9062 - nadmašen CNN ansamblom, i dalje bolji od bilo kog pojedinačnog modela |
| CD klasa | Kroz SVE eksperimente (kernel, dilatacija, alpha, sampler, k-fold ansambl) ostala relativno najotpornija na pomeranje - sumnja na inherentno šumovitije/heterogenije labele (CD objedinjuje više različitih pod-dijagnoza) |

## 9. Trenutno najbolji rezultati (macro AUC na test setu)

- **CNN, k-fold ansambl (6 foldova): 0.9170** - trenutno najbolji rezultat u projektu
- Transformer, k-fold ansambl (5 foldova): 0.9062
- CNN, pojedinačni model: 0.9048
- Transformer, pojedinačni model: 0.8878

**HYP ostaje najteža klasa** kroz sve eksperimente - AUC 0.78-0.83 pojedinačno,
0.844 u CNN ansamblu (najbolji dosadašnji HYP rezultat, F1 0.506). **CD je
najotpornija na arhitektonske/loss intervencije** (kernel, dilatacija, alpha,
sampler je nisu značajno pomerili), ali je i ona profitirala od k-fold
ansambliranja (AUC 0.914 → 0.929).

## 10. Alati i reproduktivnost

- Git repozitorijum inicijalizovan tokom projekta (`git log`: inicijalni
  commit + SE blok kao poseban commit za čistu ablaciju).
- `.gitignore` isključuje `data/`, `venv/`, checkpoint-e i `.npz` keševe
  (prevelike/izvedene datoteke).
- Sve trening/evaluacione skripte su `config.yaml`-driven - arhitektura,
  loss, optimizer, scheduler, sampler i k-fold obim se menjaju bez dodirivanja
  koda.

## 11. Mogući sledeći koraci

1. **Grad-CAM / attention vizualizacija** - interpretabilnost (šta model
   "gleda" po klasi);

2. **Augmentacija signala** (random shift, Gaussian/baseline-wander šum,
   random lead masking) - diskutovano, može direktno adresirati
   sampler-overfitting problem kod BiGRU-a (ponovljeni uzorci bi se svaki
   put drugačije augmentovali).

3. **Istraga CD klase** - koji SCP kodovi je čine, prosečna pouzdanost
   anotacija, da li je inherentno teža/šumovitija.
