"""
train.py
========
Trening petlja za PTB-XL modele. Config-driven (config.yaml), sa
early stopping-om i čuvanjem najboljeg checkpoint-a po validacionom macro F1.

Pokretanje:
    python -m src.train --config config.yaml
"""

import argparse
import csv
import os

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from src.data_loading import pripremi_kompletan_dataset, SUPERKLASE
from src.losses import FocalLossMultiLabel
from src.models import PTBXL_CNN, PTBXL_BiGRU, PTBXL_Transformer
from src.preprocessing import PTBXLDataset, predobradi_dataset


INDEKS_NORM = SUPERKLASE.index("NORM")


def ucitaj_config(putanja: str) -> dict:
    with open(putanja, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def pronadji_optimalne_pragove(y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    """
    Za svaku klasu nezavisno nalazi prag koji maksimizuje F1, na osnovu
    predviđanja modela na VALIDACIONOM setu.
    """
    pragovi = np.full(y_true.shape[1], 0.5)
    for i in range(y_true.shape[1]):
        precision, recall, thresholds = precision_recall_curve(y_true[:, i], y_prob[:, i])
        if len(thresholds) == 0:
            continue  # klasa nema oba tipa uzoraka - ostaje podrazumevani prag 0.5
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        pragovi[i] = thresholds[np.argmax(f1[:-1])]
    return pragovi


def napravi_weighted_sampler(y: np.ndarray) -> WeightedRandomSampler:
    """
    WeightedRandomSampler za multi-label imbalans: svaki uzorak dobija težinu
    jednaku inverznoj frekvenciji svoje NAJREĐE pozitivne klase (npr. uzorak
    sa i NORM i HYP dijagnozom dobija težinu HYP-a, ne NORM-a), da bi retke
    dijagnoze bile češće izvučene u batch-u. Komplementarno sa alpha vektorom
    u focal loss-u, ne zamena.

    Radi i sa tvrdim (0/1) i mekim (0.0-1.0) labelama - "prisutnost" klase
    se proverava preko y > 0, pošto je meka labela 0 tačno kad je i tvrda 0.
    """
    prisutnost = (y > 0).astype(np.float64)
    frekvencije = prisutnost.mean(axis=0)
    inverzne = 1.0 / np.clip(frekvencije, 1e-6, None)

    tezine = (prisutnost * inverzne).max(axis=1)
    tezine = np.where(tezine > 0, tezine, 1.0)  # fallback ako red nema nijednu pozitivnu klasu

    return WeightedRandomSampler(weights=tezine, num_samples=len(tezine), replacement=True)


def macro_f1_bez_norm(mete: np.ndarray, probs: np.ndarray, prag=0.5) -> float:
    """
    Macro F1 preko patoloških klasa (MI, STTC, CD, HYP), bez NORM.
    NORM je dominantna, "laka" klasa - uključivanje u macro F1 bi razblažilo
    signal o tome koliko model stvarno prepoznaje dijagnoze.
    """
    predikcije = (probs >= prag).astype(int)
    maska = np.ones(mete.shape[1], dtype=bool)
    maska[INDEKS_NORM] = False
    return f1_score(mete[:, maska], predikcije[:, maska], average="macro", zero_division=0)


_NAZIVI_MODELA = {
    "cnn": "PTBXL_CNN",
    "bigru": "PTBXL_BiGRU",
    "transformer": "PTBXL_Transformer",
}


def naziv_modela_iz_configa(cfg: dict) -> str:
    return f"{_NAZIVI_MODELA[cfg['model']['tip']]}_{cfg['loss']['tip']}"


def napravi_model(cfg: dict) -> torch.nn.Module:
    m = cfg["model"]
    if m["tip"] == "cnn":
        return PTBXL_CNN(in_channels=m["in_channels"], num_classes=m["num_classes"])
    elif m["tip"] == "bigru":
        return PTBXL_BiGRU(
            in_channels=m["in_channels"],
            hidden_size=m["hidden_size"],
            num_layers=m["num_layers"],
            num_classes=m["num_classes"],
            dropout=m["dropout"],
        )
    elif m["tip"] == "transformer":
        return PTBXL_Transformer(
            in_channels=m["in_channels"],
            d_model=m["hidden_size"],
            nhead=m["nhead"],
            num_layers=m["num_layers"],
            num_classes=m["num_classes"],
            dropout=m["dropout"],
        )
    else:
        raise ValueError(f"Nepoznat tip modela: {m['tip']}")


def napravi_loss(cfg: dict, device: torch.device) -> torch.nn.Module:
    l = cfg["loss"]
    if l["tip"] == "focal":
        loss_fn = FocalLossMultiLabel(alpha=l["alpha"], gamma=l["gamma"])
    elif l["tip"] == "bce":
        loss_fn = torch.nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Nepoznat tip loss funkcije: {l['tip']}")
    return loss_fn.to(device)


def napravi_optimizer(cfg: dict, model: torch.nn.Module) -> torch.optim.Optimizer:
    t = cfg["trening"]
    tip = t.get("optimizer", "adam")
    if tip == "adam":
        return torch.optim.Adam(
            model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"],
        )
    elif tip == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"],
        )
    elif tip == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"],
            momentum=t.get("momentum", 0.9),
        )
    else:
        raise ValueError(f"Nepoznat tip optimizatora: {tip}")


def evaluiraj(model, loader, device):
    model.eval()
    svi_pred, sve_mete = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            logits = model(X)
            probs = torch.sigmoid(logits).cpu().numpy()
            svi_pred.append(probs)
            sve_mete.append(y.numpy())

    probs = np.concatenate(svi_pred, axis=0)
    mete = np.concatenate(sve_mete, axis=0)

    auc_po_klasi = []
    for i in range(mete.shape[1]):
        try:
            auc_po_klasi.append(roc_auc_score(mete[:, i], probs[:, i]))
        except ValueError:
            pass  # klasa nema oba tipa uzoraka (0 i 1) u ovom setu

    macro_auc = float(np.mean(auc_po_klasi)) if auc_po_klasi else 0.0
    return macro_auc, probs, mete


def treniraj_model(cfg: dict, train_ds, val_ds, checkpoint_putanja: str, device: torch.device) -> float:
    """
    Trenira jedan model na datim train/val skupovima, čuva najbolji checkpoint
    (po val macro AUC) na checkpoint_putanja, i vraća najbolji postignuti val AUC.
    Deljena logika za train.py (jedan fiksni split) i train_kfold.py (k-fold).
    """
    if cfg["trening"].get("weighted_sampler", True):
        sampler = napravi_weighted_sampler(train_ds.y.numpy())
        train_loader = DataLoader(train_ds, batch_size=cfg["trening"]["batch_size"], sampler=sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=cfg["trening"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["trening"]["batch_size"], shuffle=False)

    model = napravi_model(cfg).to(device)
    loss_fn = napravi_loss(cfg, device)
    optimizer = napravi_optimizer(cfg, model)
    scheduler = None
    if cfg["trening"].get("scheduler", True):
        warmup_epochs = cfg["trening"].get("warmup_epochs", 0)
        if warmup_epochs > 0:
            # linearno zagrevanje LR-a (0.1x -> 1.0x) kroz warmup_epochs
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs,
            )
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=cfg["trening"]["epochs"] - warmup_epochs,
                eta_min=cfg["trening"]["min_lr"],
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs],
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=cfg["trening"]["epochs"],
                eta_min=cfg["trening"]["min_lr"],
            )

    grad_clip = cfg["trening"].get("grad_clip", 0)


    # -> logs/best_PTBXL_CNN_focal.csv)
    os.makedirs(cfg["putanje"]["logs"], exist_ok=True)
    naziv_loga = os.path.splitext(os.path.basename(checkpoint_putanja))[0] + ".csv"
    log_putanja = os.path.join(cfg["putanje"]["logs"], naziv_loga)
    log_fajl = open(log_putanja, "w", newline="", encoding="utf-8")
    log_writer = csv.writer(log_fajl)
    log_writer.writerow(["epoha", "train_loss", "val_macro_auc", "lr", "sacuvan"])

    najbolji_auc = 0.0
    strpljenje = 0

    for epoha in range(1, cfg["trening"]["epochs"] + 1):
        model.train()
        ukupan_loss = 0.0

        for X, y in tqdm(train_loader, desc=f"Epoha {epoha}", leave=False):
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(X)
            loss = loss_fn(logits, y)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            ukupan_loss += loss.item() * X.size(0)

        prosecan_loss = ukupan_loss / len(train_ds)
        val_auc, _, _ = evaluiraj(model, val_loader, device)
        trenutni_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()

        print(f"[Epoha {epoha}] train_loss={prosecan_loss:.4f}  val_macro_auc={val_auc:.4f}  lr={trenutni_lr:.6f}")

        sacuvan = val_auc > najbolji_auc
        log_writer.writerow([epoha, f"{prosecan_loss:.6f}", f"{val_auc:.6f}", f"{trenutni_lr:.8f}", int(sacuvan)])
        log_fajl.flush()

        if sacuvan:
            najbolji_auc = val_auc
            strpljenje = 0
            torch.save(model.state_dict(), checkpoint_putanja)
            print(f"  [SAVED] Novi najbolji model (val_auc={val_auc:.4f}) -> {checkpoint_putanja}")
        else:
            strpljenje += 1
            if strpljenje >= cfg["trening"]["early_stopping_patience"]:
                print(f"[INFO] Early stopping nakon {epoha} epoha (bez napretka {strpljenje} epoha).")
                break

    log_fajl.close()

    print(f"[GOTOVO] Najbolji validacioni macro AUC: {najbolji_auc:.4f}  -> {checkpoint_putanja}")
    print(f"[GOTOVO] Log po epohama: {log_putanja}")
    return najbolji_auc


def treniraj(cfg: dict):
    device = torch.device(cfg["trening"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Koristim uređaj: {device}")

    os.makedirs(cfg["putanje"]["checkpoints"], exist_ok=True)
    os.makedirs(cfg["putanje"]["logs"], exist_ok=True)
    os.makedirs(cfg["podaci"]["putanja_processed"], exist_ok=True)

    # --- Učitavanje ili keširanje predobrađenih podataka ---
    # sampling_rate je deo imena da promena u config.yaml ne bi tiho učitala
    # keš napravljen za drugi sampling_rate
    sr = cfg["podaci"]["sampling_rate"]
    keš_putanja = os.path.join(cfg["podaci"]["putanja_processed"], f"ptbxl_cache_{sr}hz.npz")
    d = None
    if os.path.exists(keš_putanja):
        kandidat = np.load(keš_putanja)
        if "y_train_meke" in kandidat:
            d = kandidat
            print(f"[INFO] Učitavam keširane podatke iz {keš_putanja}...")
        else:
            print(f"[INFO] Keš {keš_putanja} je iz starije verzije (bez mekih labela) - pravim ponovo.")

    if d is not None:
        X_train, y_train, y_train_meke = d["X_train"], d["y_train"], d["y_train_meke"]
        X_val, y_val = d["X_val"], d["y_val"]
        X_test, y_test = d["X_test"], d["y_test"]
    else:
        podaci = pripremi_kompletan_dataset(
            cfg["podaci"]["putanja_ptbxl"], cfg["podaci"]["sampling_rate"]
        )
        print("[INFO] Predobrađujem signale (filtriranje + normalizacija)...")
        # podaci.pop() oslobađa sirovu verziju čim je predobrađena verzija
        # gotova, umesto da obe koegzistiraju u memoriji do kraja bloka
        X_train = predobradi_dataset(podaci.pop("X_train"), cfg["podaci"]["sampling_rate"])
        X_val = predobradi_dataset(podaci.pop("X_val"), cfg["podaci"]["sampling_rate"])
        X_test = predobradi_dataset(podaci.pop("X_test"), cfg["podaci"]["sampling_rate"])
        y_train, y_train_meke = podaci["y_train"], podaci["y_train_meke"]
        y_val, y_test = podaci["y_val"], podaci["y_test"]

        print(f"[INFO] Čuvam predobrađene podatke u keš: {keš_putanja}")
        np.savez_compressed(
            keš_putanja,
            X_train=X_train, y_train=y_train, y_train_meke=y_train_meke,
            X_val=X_val, y_val=y_val,
            X_test=X_test, y_test=y_test,
        )

    # trening koristi MEKE labele (confidence-weighted, label smoothing na
    # osnovu PTB-XL likelihood anotacije) - validacija ostaje na tvrdim 0/1
    # labelama, jer metrike (AUC/F1) zahtevaju diskretan ground truth
    train_ds = PTBXLDataset(
        X_train, y_train_meke,
        lead_masking_p=cfg["trening"].get("lead_masking_p", 0.0),
        lead_masking_broj=cfg["trening"].get("lead_masking_broj", 1),
    )
    val_ds = PTBXLDataset(X_val, y_val)  # bez augmentacije - validacija mora biti nepromenjena

    naziv_modela = naziv_modela_iz_configa(cfg)
    checkpoint_putanja = os.path.join(cfg["putanje"]["checkpoints"], f"best_{naziv_modela}.pth")

    treniraj_model(cfg, train_ds, val_ds, checkpoint_putanja, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    cfg = ucitaj_config(args.config)
    treniraj(cfg)
