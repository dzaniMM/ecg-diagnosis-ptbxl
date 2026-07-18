# PTB-XL Multi-Lead EKG Klasifikacija

Multi-label klasifikacija 12-kanalnih EKG snimaka (PTB-XL) u 5 dijagnostičkih
superklasa: **NORM, MI, STTC, CD, HYP**. Detaljan opis problema, arhitektura i
eksperimenata: [`IZVESTAJ.md`](IZVESTAJ.md).

## Struktura projekta

```
ptbxl-ekg-projekat/
├── config.yaml                  # Svi hiperparametri na jednom mestu
├── requirements.txt
├── data/
│   ├── raw/                     # Ovde ide preuzeti PTB-XL (korak 2)
│   └── processed/                # Keš predobrađenih podataka (pravi se automatski)
├── checkpoints/                  # Sačuvani najbolji modeli (po treningu/foldu)
├── gradcam_izlaz/                 # Izlazne slike iz interpretability.py
├── scripts/
│   └── download_ptbxl.sh         # Preuzimanje sa PhysioNet-a
├── notebooks/
│   └── PREZENTACIJA.ipynb        # Prezentacija projekta sa vizualizacijama
├── src/
│   ├── data_loading.py           # Učitavanje + mapiranje SCP kodova -> 5 superklasa
│   ├── preprocessing.py          # Filtriranje, normalizacija, PyTorch Dataset
│   ├── models.py                 # PTBXL_CNN, PTBXL_BiGRU, PTBXL_Transformer
│   ├── losses.py                 # FocalLossMultiLabel (per-klasa alpha)
│   ├── train.py                  # Trening (zvanični split)
│   ├── train_kfold.py            # Trening (k-fold cross-validation)
│   ├── evaluate.py               # Evaluacija jednog checkpoint-a
│   ├── evaluate_kfold.py         # Evaluacija k-fold ansambla (soft-voting)
│   └── interpretability.py       # Grad-CAM i lead occlusion analiza
└── IZVESTAJ.md                   # Opis problema, arhitektura, eksperimenata i nalaza
```

## Kako pokrenuti — korak po korak

### 1. Instaliraj zavisnosti

```bash
pip install -r requirements.txt
```


### 2. Preuzmi PTB-XL podatke

Lokalno (physionet.org nije dostupan iz Claude sandbox okruženja):

```bash
chmod +x scripts/download_ptbxl.sh
./scripts/download_ptbxl.sh
```

### 3. (Opciono) Proveri da učitavanje radi

```bash
python -m src.data_loading
```

Ispisuje distribuciju klasa i oblike nizova (train/val/test).

### 4. Podesi `config.yaml`

Ključna polja:

| Polje | Opcije |
|---|---|
| `podaci.sampling_rate` | `100` ili `500` (Hz) |
| `model.tip` | `"cnn"`, `"bigru"` ili `"transformer"` |
| `loss.tip` | `"focal"` ili `"bce"` |
| `trening.optimizer` | `"adam"`, `"adamw"` ili `"sgd"` |
| `trening.weighted_sampler` | `true`/`false` — retki uzorci (HYP, CD) češće u batch-u |
| `trening.lead_masking_p` | `0.0`-`1.0` — verovatnoća da se nasumični odvodi ugase (augmentacija) |
| `kfold.foldovi` | lista foldova (1-9) koji rotiraju kao validacija u k-fold treningu |

### 5. Treniraj model (zvanični split)

```bash
python -m src.train --config config.yaml
```

Prvi put predobrada + učitavanje svih 12-kanalnih signala potraje — keš se čuva
u `data/processed/ptbxl_cache_{100|500}hz.npz`, naredni treninzi kreću odmah.
Checkpoint se čuva u `checkpoints/best_{Model}_{loss}.pth`, po najboljem
validacionom macro AUC.

### 6. Evaluiraj na test setu

```bash
python -m src.evaluate --config config.yaml --checkpoint checkpoints/best_PTBXL_CNN_focal.pth
```

`--prag` (podrazumevano `0.5`) menja fiksni prag za izveštaj — skripta uvek i
sama pronalazi optimalan prag po klasi (na validacionom setu) i prijavljuje oba
rezultata. Generiše i `ptbxl_matrice_konfuzije.png` (5 binarnih matrica
konfuzije, po jedna za svaku superklasu).

**Napomena:** `model.tip` u config-u mora da odgovara arhitekturi checkpoint-a
koji učitavaš (npr. `"cnn"` za `best_PTBXL_CNN_*.pth`), inače puca greška
pri učitavanju težina.

### 7. K-fold cross-validation + ansambl

```bash
python -m src.train_kfold --config config.yaml    # trenira po jedan model za svaki fold iz kfold.foldovi
python -m src.evaluate_kfold --config config.yaml  # soft-voting ansambl na test setu
```


### 8. Interpretabilnost — Grad-CAM i lead occlusion (samo za `model.tip: "cnn"`)

```bash
# Grad-CAM - PNG po primeru/klasi
python -m src.interpretability --config config.yaml --checkpoint checkpoints/best_PTBXL_CNN_focal.pth

# koji odvodi su najbitniji za datu klasu
python -m src.interpretability --config config.yaml --checkpoint checkpoints/best_PTBXL_CNN_focal.pth \
    --analiza lead_occlusion --klasa HYP --n-primera 262

python -m src.interpretability --config config.yaml --analiza lead_occlusion_kfold --klasa HYP
```

Izlazne slike idu u `gradcam_izlaz/` (podesivo preko `--izlaz`).


