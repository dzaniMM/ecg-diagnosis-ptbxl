"""
evaluate.py
===========
Pokretanje:
    python -m src.evaluate --config config.yaml --checkpoint checkpoints/best_PTBXL_BiGRU_focal.pth
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from src.data_loading import SUPERKLASE
from src.preprocessing import PTBXLDataset
from src.train import napravi_model, ucitaj_config, evaluiraj, pronadji_optimalne_pragove


def detaljne_metrike(y_true: np.ndarray, y_prob: np.ndarray, prag) -> pd.DataFrame:
    y_pred = (y_prob >= prag).astype(int)
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    auc_po_klasi = []
    for i in range(y_true.shape[1]):
        try:
            auc_po_klasi.append(roc_auc_score(y_true[:, i], y_prob[:, i]))
        except ValueError:
            auc_po_klasi.append(float("nan"))  # ako klasa nema oba tipa uzoraka u test setu

    return pd.DataFrame({
        "Klasa": SUPERKLASE,
        "Precision": p,
        "Recall": r,
        "F1-Score": f1,
        "AUC": auc_po_klasi,
        "Broj pozitivnih": support,
    })


def nacrtaj_matrice_konfuzije(y_true: np.ndarray, y_pred: np.ndarray, sacuvaj_kao: str = None):
    """Crta po jednu binarnu (2x2) matricu konfuzije za svaku od 5 klasa."""
    fig, axes = plt.subplots(1, 5, figsize=(22, 4))

    for i, klasa in enumerate(SUPERKLASE):
        cm = confusion_matrix(y_true[:, i], y_pred[:, i])
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=axes[i], cbar=False,
            xticklabels=["Ne", "Da"], yticklabels=["Ne", "Da"],
            annot_kws={"size": 13, "weight": "bold"},
        )
        axes[i].set_title(f"Klasa: {klasa}", fontsize=13, fontweight="bold")
        axes[i].set_xlabel("Predviđeno")
        axes[i].set_ylabel("Stvarno")

    plt.tight_layout()
    if sacuvaj_kao:
        plt.savefig(sacuvaj_kao, dpi=300)
    plt.show()


def glavna(cfg_putanja: str, checkpoint_putanja: str, prag: float = 0.5):
    cfg = ucitaj_config(cfg_putanja)
    device = torch.device(cfg["trening"]["device"] if torch.cuda.is_available() else "cpu")

    sr = cfg["podaci"]["sampling_rate"]
    keš_putanja = os.path.join(cfg["podaci"]["putanja_processed"], f"ptbxl_cache_{sr}hz.npz")
    if not os.path.exists(keš_putanja):
        raise FileNotFoundError(
            f"Nije pronađen keš {keš_putanja}. Prvo pokreni src/train.py da se podaci "
            f"predobrade i keširaju."
        )

    d = np.load(keš_putanja)
    val_ds = PTBXLDataset(d["X_val"], d["y_val"])
    val_loader = DataLoader(val_ds, batch_size=cfg["trening"]["batch_size"], shuffle=False)
    test_ds = PTBXLDataset(d["X_test"], d["y_test"])
    test_loader = DataLoader(test_ds, batch_size=cfg["trening"]["batch_size"], shuffle=False)

    model = napravi_model(cfg).to(device)
    model.load_state_dict(torch.load(checkpoint_putanja, map_location=device))

    # Pragovi po klasi se biraju na validacionom, ne na test setu
    _, y_prob_val, y_true_val = evaluiraj(model, val_loader, device)
    pragovi = pronadji_optimalne_pragove(y_true_val, y_prob_val)

    macro_auc, y_prob, y_true = evaluiraj(model, test_loader, device)

    y_pred_fiksni = (y_prob >= prag).astype(int)
    y_pred_tuned = (y_prob >= pragovi).astype(int)

    macro_f1_fiksni = f1_score(y_true, y_pred_fiksni, average="macro", zero_division=0)
    macro_f1_tuned = f1_score(y_true, y_pred_tuned, average="macro", zero_division=0)

    print(f"\n[REZULTAT] Test macro AUC (bez praga): {macro_auc:.4f}")
    print(f"[REZULTAT] Test macro F1 (fiksni prag={prag}): {macro_f1_fiksni:.4f}")
    print(f"[REZULTAT] Test macro F1 (pragovi po klasi, birani na val setu): {macro_f1_tuned:.4f}")
    for klasa, p in zip(SUPERKLASE, pragovi):
        print(f"    prag[{klasa}] = {p:.4f}")
    print()

    df_metrike = detaljne_metrike(y_true, y_prob, prag=pragovi)
    print(df_metrike.to_string(index=False, formatters={
        "Precision": "{:,.4f}".format,
        "Recall": "{:,.4f}".format,
        "F1-Score": "{:,.4f}".format,
        "AUC": "{:,.4f}".format,
    }))

    nacrtaj_matrice_konfuzije(y_true, y_pred_tuned, sacuvaj_kao="ptbxl_matrice_konfuzije.png")

    return df_metrike, macro_f1_tuned


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prag", type=float, default=0.5)
    args = parser.parse_args()

    glavna(args.config, args.checkpoint, args.prag)
