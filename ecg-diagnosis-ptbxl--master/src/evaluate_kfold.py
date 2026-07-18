"""
evaluate_kfold.py
==================
Ansambl evaluacija preko k-fold checkpoint-a iz train_kfold.py: učitava svih
9 fold modela, usrednjava njihove verovatnoće (soft-voting) na test setu
(fold 10), i poredi ansambl sa pojedinačnim fold modelima.

Pragovi po klasi se biraju na OUT-OF-FOLD predikcijama - svaki uzorak iz
foldova 1-9 dobija predikciju samo od modela koji ga NIJE video u treningu
(modela kome je taj fold bio validacija) - da izbor praga ne bi zavisio ni
od test seta ni od podataka koje je model već video.

Pokretanje:
    python -m src.evaluate_kfold --config config.yaml
"""

import argparse
import glob
import os

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader

from src.data_loading import SUPERKLASE
from src.evaluate import detaljne_metrike, nacrtaj_matrice_konfuzije
from src.preprocessing import PTBXLDataset
from src.train import (
    evaluiraj,
    macro_f1_bez_norm,
    napravi_model,
    naziv_modela_iz_configa,
    pronadji_optimalne_pragove,
    ucitaj_config,
)


def glavna(cfg_putanja: str):
    cfg = ucitaj_config(cfg_putanja)
    device = torch.device(cfg["trening"]["device"] if torch.cuda.is_available() else "cpu")

    sr = cfg["podaci"]["sampling_rate"]
    keš_putanja = os.path.join(cfg["podaci"]["putanja_processed"], f"ptbxl_kfold_cache_{sr}hz.npz")
    if not os.path.exists(keš_putanja):
        raise FileNotFoundError(
            f"Nije pronađen keš {keš_putanja}. Prvo pokreni src/train_kfold.py."
        )

    d = np.load(keš_putanja)
    X_trainval, y_trainval, foldovi = d["X_trainval"], d["y_trainval"], d["foldovi_trainval"]
    X_test, y_test = d["X_test"], d["y_test"]

    naziv_modela = naziv_modela_iz_configa(cfg)
    obrazac = os.path.join(cfg["putanje"]["checkpoints"], f"best_{naziv_modela}_fold*.pth")
    checkpoint_putanje = sorted(glob.glob(obrazac))
    if not checkpoint_putanje:
        raise FileNotFoundError(
            f"Nije pronađen nijedan k-fold checkpoint po obrascu {obrazac}. "
            f"Prvo pokreni src/train_kfold.py."
        )
    print(f"[INFO] Pronađeno {len(checkpoint_putanje)} fold checkpoint-a.")

    test_ds = PTBXLDataset(X_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=cfg["trening"]["batch_size"], shuffle=False)

    oof_probs = np.zeros_like(y_trainval, dtype=np.float32)
    pokriveni_foldovi = set()
    svi_test_probs = []

    for putanja in checkpoint_putanje:
        fold = int(putanja.rsplit("fold", 1)[1].split(".")[0])
        pokriveni_foldovi.add(fold)
        model = napravi_model(cfg).to(device)
        model.load_state_dict(torch.load(putanja, map_location=device))

        val_maska = foldovi == fold
        val_ds = PTBXLDataset(X_trainval[val_maska], y_trainval[val_maska])
        val_loader = DataLoader(val_ds, batch_size=cfg["trening"]["batch_size"], shuffle=False)
        val_auc, val_probs, _ = evaluiraj(model, val_loader, device)
        oof_probs[val_maska] = val_probs

        test_auc, test_probs, _ = evaluiraj(model, test_loader, device)
        svi_test_probs.append(test_probs)
        print(f"  Fold {fold}: val_auc={val_auc:.4f}  test_auc={test_auc:.4f}  -> {putanja}")

    ensemble_test_probs = np.mean(svi_test_probs, axis=0)

    # Pragovi po klasi biraju se na out-of-fold predikcijama, ograničeno na
    # foldove koji stvarno imaju checkpoint (ako train_kfold.py nije trenirao
    # svih 9, ostali foldovi u oof_probs nemaju validnu predikciju)
    pokrivena_maska = np.isin(foldovi, list(pokriveni_foldovi))
    pragovi = pronadji_optimalne_pragove(y_trainval[pokrivena_maska], oof_probs[pokrivena_maska])

    auc_po_klasi = [roc_auc_score(y_test[:, i], ensemble_test_probs[:, i]) for i in range(y_test.shape[1])]
    macro_auc_ensemble = float(np.mean(auc_po_klasi))
    y_pred_ensemble = (ensemble_test_probs >= pragovi).astype(int)
    macro_f1_ensemble = f1_score(y_test, y_pred_ensemble, average="macro", zero_division=0)
    f1_bez_norm_ensemble = macro_f1_bez_norm(y_test, ensemble_test_probs, prag=pragovi)

    print(f"\n[ANSAMBL] Test macro AUC: {macro_auc_ensemble:.4f}")
    print(f"[ANSAMBL] Test macro F1 (pragovi iz out-of-fold): {macro_f1_ensemble:.4f}")
    print(f"[ANSAMBL] Test macro F1 bez NORM: {f1_bez_norm_ensemble:.4f}")
    for klasa, p in zip(SUPERKLASE, pragovi):
        print(f"    prag[{klasa}] = {p:.4f}")

    df_metrike = detaljne_metrike(y_test, ensemble_test_probs, prag=pragovi)
    print()
    print(df_metrike.to_string(index=False, formatters={
        "Precision": "{:,.4f}".format,
        "Recall": "{:,.4f}".format,
        "F1-Score": "{:,.4f}".format,
        "AUC": "{:,.4f}".format,
    }))

    nacrtaj_matrice_konfuzije(y_test, y_pred_ensemble, sacuvaj_kao="ptbxl_matrice_konfuzije_ansambl.png")

    return df_metrike, macro_f1_ensemble


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    glavna(args.config)