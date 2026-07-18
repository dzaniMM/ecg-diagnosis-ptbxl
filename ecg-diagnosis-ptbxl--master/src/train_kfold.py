"""
train_kfold.py
===============
K-fold unakrsna validacija po zvaničnom PTB-XL protokolu: rotira se koji
od foldova 1-9 služi kao validacija, dok ostatak (od preostalih 1-9) ide
u trening. Fold 10 se NIKAD ne koristi u treningu ovde - njega koristi
isključivo evaluate_kfold.py, kao zajednički test skup za ansambl.

Rezultat: po jedan checkpoint za svaki od 9 modela,
checkpoints/best_{Model}_{loss}_fold{k}.pth (k = 1..9).

Pokretanje:
    python -m src.train_kfold --config config.yaml
"""

import argparse
import os

import numpy as np
import torch

from src.data_loading import pripremi_kfold_dataset
from src.preprocessing import PTBXLDataset, predobradi_dataset
from src.train import naziv_modela_iz_configa, treniraj_model, ucitaj_config


def treniraj_kfold(cfg: dict):
    device = torch.device(cfg["trening"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Koristim uređaj: {device}")

    os.makedirs(cfg["putanje"]["checkpoints"], exist_ok=True)
    os.makedirs(cfg["podaci"]["putanja_processed"], exist_ok=True)

    # sampling_rate je deo imena da promena u config.yaml ne bi tiho učitala
    # keš napravljen za drugi sampling_rate
    sr = cfg["podaci"]["sampling_rate"]
    keš_putanja = os.path.join(cfg["podaci"]["putanja_processed"], f"ptbxl_kfold_cache_{sr}hz.npz")
    d = None
    if os.path.exists(keš_putanja):
        kandidat = np.load(keš_putanja)
        if "y_trainval_meke" in kandidat:
            d = kandidat
            print(f"[INFO] Učitavam keširane k-fold podatke iz {keš_putanja}...")
        else:
            print(f"[INFO] Keš {keš_putanja} je iz starije verzije (bez mekih labela) - pravim ponovo.")

    if d is not None:
        X_trainval = d["X_trainval"]
        y_trainval, y_trainval_meke = d["y_trainval"], d["y_trainval_meke"]
        foldovi = d["foldovi_trainval"]
    else:
        podaci = pripremi_kfold_dataset(cfg["podaci"]["putanja_ptbxl"], cfg["podaci"]["sampling_rate"])
        print("[INFO] Predobrađujem signale (filtriranje + normalizacija)...")
        # podaci.pop() oslobađa sirovu verziju čim je predobrađena verzija
        # gotova - bitno kod k-fold-a jer je X_trainval ~90% celog dataseta
        X_trainval = predobradi_dataset(podaci.pop("X_trainval"), cfg["podaci"]["sampling_rate"])
        X_test = predobradi_dataset(podaci.pop("X_test"), cfg["podaci"]["sampling_rate"])
        y_trainval, y_trainval_meke = podaci["y_trainval"], podaci["y_trainval_meke"]
        y_test = podaci["y_test"]
        foldovi = podaci["foldovi_trainval"]

        print(f"[INFO] Čuvam predobrađene k-fold podatke u keš: {keš_putanja}")
        np.savez_compressed(
            keš_putanja,
            X_trainval=X_trainval, y_trainval=y_trainval, y_trainval_meke=y_trainval_meke,
            foldovi_trainval=foldovi,
            X_test=X_test, y_test=y_test,
        )

    naziv_modela = naziv_modela_iz_configa(cfg)
    rezultati = {}

    foldovi_za_val = cfg.get("kfold", {}).get("foldovi", list(range(1, 10)))
    print(f"[INFO] Foldovi koji rotiraju kao validacija: {foldovi_za_val}")

    for val_fold in foldovi_za_val:
        print(f"\n{'=' * 70}")
        print(f"[FOLD {val_fold}] validacija = fold {val_fold}, trening = ostali foldovi 1-9")
        print(f"{'=' * 70}")

        train_maska = foldovi != val_fold
        val_maska = foldovi == val_fold

        # trening koristi meke (confidence-weighted) labele, validacija tvrde 0/1
        train_ds = PTBXLDataset(
            X_trainval[train_maska], y_trainval_meke[train_maska],
            lead_masking_p=cfg["trening"].get("lead_masking_p", 0.0),
            lead_masking_broj=cfg["trening"].get("lead_masking_broj", 1),
        )
        val_ds = PTBXLDataset(X_trainval[val_maska], y_trainval[val_maska])  # bez augmentacije

        checkpoint_putanja = os.path.join(
            cfg["putanje"]["checkpoints"], f"best_{naziv_modela}_fold{val_fold}.pth"
        )
        rezultati[val_fold] = treniraj_model(cfg, train_ds, val_ds, checkpoint_putanja, device)

    print(f"\n{'=' * 70}")
    print("[GOTOVO] Rezultati po foldu (najbolji val macro AUC):")
    for fold, auc in rezultati.items():
        print(f"  Fold {fold}: {auc:.4f}")
    print(f"  Prosek: {np.mean(list(rezultati.values())):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    cfg = ucitaj_config(args.config)
    treniraj_kfold(cfg)