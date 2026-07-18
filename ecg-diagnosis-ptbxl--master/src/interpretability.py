"""
interpretability.py
====================
Grad-CAM vizualizacija za PTBXL_CNN: pokazuje koji delovi EKG signala 
su najviše doprineli predikciji svake dijagnostičke klase.

Pokretanje:
    python -m src.interpretability --config config.yaml --checkpoint checkpoints/best_PTBXL_CNN_focal.pth
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.data_loading import SUPERKLASE
from src.train import napravi_model, naziv_modela_iz_configa, ucitaj_config

ODVODI = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def grad_cam(model: torch.nn.Module, x: torch.Tensor, klasa_idx: int) -> np.ndarray:
    """
    Grad-CAM mapa važnosti kroz vreme za PTBXL_CNN, za jedan uzorak
    ([1, 12, samples]) i ciljanu klasu. Vraća 1D niz iste dužine kao ulazni
    signal, normalizovan na [0, 1].
    """
    aktivacije = {}
    gradijenti = {}

    def forward_hook(module, ulaz, izlaz):
        aktivacije["vrednost"] = izlaz

    def backward_hook(module, grad_ulaz, grad_izlaz):
        gradijenti["vrednost"] = grad_izlaz[0]

    h1 = model.layer3.register_forward_hook(forward_hook)
    h2 = model.layer3.register_full_backward_hook(backward_hook)

    model.eval()
    logits = model(x)
    model.zero_grad()
    logits[:, klasa_idx].sum().backward()

    h1.remove()
    h2.remove()

    akt = aktivacije["vrednost"]    # [1, C, T']
    grad = gradijenti["vrednost"]   # [1, C, T']

    tezine = grad.mean(dim=2, keepdim=True)       # [1, C, 1] - global avg pool gradijenata kroz vreme
    cam = F.relu((tezine * akt).sum(dim=1))        # [1, T']
    cam = F.interpolate(cam.unsqueeze(1), size=x.shape[2], mode="linear", align_corners=False)
    cam = cam.squeeze().detach().cpu().numpy()

    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam


def nacrtaj_gradcam(signal: np.ndarray, cam: np.ndarray, klasa: str, prob: float,
                     naziv_fajla: str, odvod_idx: int = 1):
    """
    Crta jedan odvod EKG signala sa Grad-CAM mapom kao heatmap pozadinom.
    signal: [12, samples] (jedan uzorak). odvod_idx=1 -> odvod II (podrazumevano).
    """
    fig, ax = plt.subplots(figsize=(14, 4))

    duzina = signal.shape[1]
    ax.imshow(
        cam[np.newaxis, :], aspect="auto", cmap="Reds", alpha=0.5,
        extent=[0, duzina, signal[odvod_idx].min(), signal[odvod_idx].max()],
    )
    ax.plot(signal[odvod_idx], color="black", linewidth=1)
    ax.set_title(f"Grad-CAM - klasa: {klasa} (p={prob:.3f}, odvod {ODVODI[odvod_idx]})")
    ax.set_xlabel("Vremenski koraci")
    ax.set_ylabel("Amplituda")

    plt.tight_layout()
    plt.savefig(naziv_fajla, dpi=150)
    plt.close(fig)


def lead_occlusion(model: torch.nn.Module, x: torch.Tensor, klasa_idx: int) -> tuple:
    """
    Za dati uzorak ([1, 12, samples]) i klasu, meri pad predviđene
    verovatnoće kad se svaki od 12 odvoda pojedinačno ugasi (postavi na 0).
    Veći pad => model se više oslanja na taj odvod za tu klasu.

    Vraća (baseline_p, padovi) gde je padovi niz od 12 vrednosti
    (baseline_p - p_bez_tog_odvoda).
    """
    model.eval()
    with torch.no_grad():
        baseline_p = torch.sigmoid(model(x))[0, klasa_idx].item()

        padovi = np.zeros(12)
        for lead_idx in range(12):
            x_ugaseno = x.clone()
            x_ugaseno[0, lead_idx, :] = 0.0
            p = torch.sigmoid(model(x_ugaseno))[0, klasa_idx].item()
            padovi[lead_idx] = baseline_p - p

    return baseline_p, padovi


def nacrtaj_lead_occlusion(padovi_matrica: np.ndarray, klasa: str, naziv_fajla: str):
    """
    padovi_matrica: [broj_primera, 12] - pad verovatnoće po odvodu, po primeru.
    Crta bar chart proseka (+ std) preko primera - veći bar = odvod je bitniji.
    """
    prosek = padovi_matrica.mean(axis=0)
    std = padovi_matrica.std(axis=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    boje = ["tab:red" if v > 0 else "tab:blue" for v in prosek]
    ax.bar(ODVODI, prosek, yerr=std, capsize=4, color=boje)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Prosečan pad P(klasa) kad se odvod ugasi")
    ax.set_xlabel("Odvod")
    ax.set_title(f"Lead occlusion - klasa: {klasa} (prosek preko {padovi_matrica.shape[0]} primera)")

    plt.tight_layout()
    plt.savefig(naziv_fajla, dpi=150)
    plt.close(fig)


def ucitaj_model_i_test_podatke(cfg_putanja: str, checkpoint_putanja: str):
    cfg = ucitaj_config(cfg_putanja)
    if cfg["model"]["tip"] != "cnn":
        raise ValueError(
            "Ova analiza je trenutno implementirana samo za model.tip='cnn' "
            "(kukica je specifično na PTBXL_CNN.layer3)."
        )

    device = torch.device(cfg["trening"]["device"] if torch.cuda.is_available() else "cpu")

    sr = cfg["podaci"]["sampling_rate"]
    keš_putanja = os.path.join(cfg["podaci"]["putanja_processed"], f"ptbxl_cache_{sr}hz.npz")
    if not os.path.exists(keš_putanja):
        raise FileNotFoundError(f"Nije pronađen keš {keš_putanja}. Prvo pokreni src/train.py.")

    d = np.load(keš_putanja)
    X_test, y_test = d["X_test"], d["y_test"]

    model = napravi_model(cfg).to(device)
    model.load_state_dict(torch.load(checkpoint_putanja, map_location=device))
    model.eval()

    return model, device, X_test, y_test


def analiza_lead_occlusion(cfg_putanja: str, checkpoint_putanja: str, klasa: str = "HYP",
                            broj_primera: int = 15, izlazni_direktorijum: str = "gradcam_izlaz"):
    model, device, X_test, y_test = ucitaj_model_i_test_podatke(cfg_putanja, checkpoint_putanja)
    klasa_idx = SUPERKLASE.index(klasa)

    pozitivni_idx = np.where(y_test[:, klasa_idx] == 1)[0]
    if len(pozitivni_idx) == 0:
        print(f"[INFO] Nema pozitivnih primera za klasu {klasa} u test setu.")
        return
    odabrani = pozitivni_idx[:broj_primera]

    padovi_matrica = np.zeros((len(odabrani), 12))
    for j, i in enumerate(odabrani):
        x = torch.tensor(X_test[i], dtype=torch.float32).permute(1, 0).unsqueeze(0).to(device)
        baseline_p, padovi = lead_occlusion(model, x, klasa_idx)
        padovi_matrica[j] = padovi
        print(f"  uzorak {i}: baseline P({klasa})={baseline_p:.3f}")

    os.makedirs(izlazni_direktorijum, exist_ok=True)
    naziv_fajla = os.path.join(izlazni_direktorijum, f"lead_occlusion_{klasa}.png")
    nacrtaj_lead_occlusion(padovi_matrica, klasa, naziv_fajla)
    print(f"[SAVED] {naziv_fajla}")

    prosek = padovi_matrica.mean(axis=0)
    poredak = np.argsort(prosek)[::-1]
    print(f"\n[REZULTAT] Odvodi po važnosti za klasu {klasa} (prosečan pad verovatnoće):")
    for idx in poredak:
        print(f"  {ODVODI[idx]:>4}: {prosek[idx]:+.4f}")


def analiza_lead_occlusion_kfold(cfg_putanja: str, checkpoint_putanje: list, klasa: str = "HYP",
                                  broj_primera: int = 262, izlazni_direktorijum: str = "gradcam_izlaz"):
    """
    Proverava da li je lead occlusion nalaz DOSLEDAN kroz više modela (npr.
    k-fold checkpoint-i, trenirani na različitim podskupovima podataka).
    Ako se isti odvodi dosledno pokazuju kao bitni/zanemareni kroz sve
    modele, to je jak dokaz da je nalaz stvaran, ne slučajnost jednog treninga.
    """
    cfg = ucitaj_config(cfg_putanja)
    device = torch.device(cfg["trening"]["device"] if torch.cuda.is_available() else "cpu")

    sr = cfg["podaci"]["sampling_rate"]
    keš_putanja = os.path.join(cfg["podaci"]["putanja_processed"], f"ptbxl_cache_{sr}hz.npz")
    if not os.path.exists(keš_putanja):
        raise FileNotFoundError(f"Nije pronađen keš {keš_putanja}. Prvo pokreni src/train.py.")

    d = np.load(keš_putanja)
    X_test, y_test = d["X_test"], d["y_test"]

    klasa_idx = SUPERKLASE.index(klasa)
    pozitivni_idx = np.where(y_test[:, klasa_idx] == 1)[0]
    odabrani = pozitivni_idx[:broj_primera]

    rezultati_po_modelu = np.zeros((len(checkpoint_putanje), 12))
    nazivi_modela = []

    for m_idx, checkpoint_putanja in enumerate(checkpoint_putanje):
        model = napravi_model(cfg).to(device)
        model.load_state_dict(torch.load(checkpoint_putanja, map_location=device))
        model.eval()

        padovi_matrica = np.zeros((len(odabrani), 12))
        for j, i in enumerate(odabrani):
            x = torch.tensor(X_test[i], dtype=torch.float32).permute(1, 0).unsqueeze(0).to(device)
            _, padovi = lead_occlusion(model, x, klasa_idx)
            padovi_matrica[j] = padovi

        rezultati_po_modelu[m_idx] = padovi_matrica.mean(axis=0)
        naziv = os.path.basename(checkpoint_putanja)
        nazivi_modela.append(naziv)
        print(f"[{naziv}] gotovo ({len(odabrani)} primera)")

    os.makedirs(izlazni_direktorijum, exist_ok=True)

    granica = np.abs(rezultati_po_modelu).max()
    fig, ax = plt.subplots(figsize=(12, max(4, len(checkpoint_putanje) * 0.7)))
    im = ax.imshow(rezultati_po_modelu, aspect="auto", cmap="RdBu_r", vmin=-granica, vmax=granica)
    ax.set_xticks(range(12))
    ax.set_xticklabels(ODVODI)
    ax.set_yticks(range(len(nazivi_modela)))
    ax.set_yticklabels(nazivi_modela)
    ax.set_title(f"Lead occlusion doslednost kroz modele - klasa: {klasa}")
    plt.colorbar(im, ax=ax, label="Prosečan pad P(klasa)")
    plt.tight_layout()
    naziv_fajla = os.path.join(izlazni_direktorijum, f"lead_occlusion_{klasa}_kfold_konzistentnost.png")
    plt.savefig(naziv_fajla, dpi=150)
    plt.close(fig)
    print(f"[SAVED] {naziv_fajla}")

    prosek_kroz_modele = rezultati_po_modelu.mean(axis=0)
    std_kroz_modele = rezultati_po_modelu.std(axis=0)
    poredak = np.argsort(prosek_kroz_modele)[::-1]

    print(f"\n[REZULTAT] Odvodi po važnosti za {klasa}, usrednjeno kroz {len(checkpoint_putanje)} modela:")
    for idx in poredak:
        znakovi = np.sign(rezultati_po_modelu[:, idx])
        slaganje = np.abs(znakovi.sum()) / len(checkpoint_putanje)  # 1.0 = svi modeli isti znak
        print(f"  {ODVODI[idx]:>4}: prosek={prosek_kroz_modele[idx]:+.4f}  "
              f"std={std_kroz_modele[idx]:.4f}  slaganje_znaka={slaganje:.0%}")

    return rezultati_po_modelu, nazivi_modela


def glavna(cfg_putanja: str, checkpoint_putanja: str, broj_primera: int = 3,
           izlazni_direktorijum: str = "gradcam_izlaz"):
    model, device, X_test, y_test = ucitaj_model_i_test_podatke(cfg_putanja, checkpoint_putanja)

    os.makedirs(izlazni_direktorijum, exist_ok=True)

    for klasa_idx, klasa in enumerate(SUPERKLASE):
        pozitivni_idx = np.where(y_test[:, klasa_idx] == 1)[0]
        if len(pozitivni_idx) == 0:
            print(f"[INFO] Nema pozitivnih primera za klasu {klasa} u test setu, preskačem.")
            continue
        odabrani = pozitivni_idx[:broj_primera]

        for i in odabrani:
            x = torch.tensor(X_test[i], dtype=torch.float32).permute(1, 0).unsqueeze(0).to(device)  # [1, 12, samples]

            cam = grad_cam(model, x, klasa_idx)
            with torch.no_grad():
                prob = torch.sigmoid(model(x))[0, klasa_idx].item()
            signal = X_test[i].T  # [12, samples]

            naziv_fajla = os.path.join(izlazni_direktorijum, f"gradcam_{klasa}_{i}.png")
            nacrtaj_gradcam(signal, cam, klasa, prob, naziv_fajla)
            print(f"[SAVED] {naziv_fajla}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, required=False)
    parser.add_argument("--n-primera", type=int, default=3, dest="broj_primera")
    parser.add_argument("--izlaz", type=str, default="gradcam_izlaz", dest="izlazni_direktorijum")
    parser.add_argument(
        "--analiza", type=str, default="gradcam",
        choices=["gradcam", "lead_occlusion", "lead_occlusion_kfold"],
        help="'gradcam', 'lead_occlusion' (jedan model) ili 'lead_occlusion_kfold' (doslednost kroz k-fold checkpoint-e)",
    )
    parser.add_argument(
        "--klasa", type=str, default="HYP", choices=SUPERKLASE,
        help="klasa za lead_occlusion analizu (podrazumevano HYP)",
    )
    args = parser.parse_args()

    if args.analiza == "lead_occlusion_kfold":
        cfg = ucitaj_config(args.config)
        naziv = naziv_modela_iz_configa(cfg)
        obrazac = os.path.join("checkpoints", f"best_{naziv}_fold*.pth")
        checkpoint_putanje = sorted(glob.glob(obrazac))
        glavni_checkpoint = os.path.join("checkpoints", f"best_{naziv}.pth")
        if os.path.exists(glavni_checkpoint):
            checkpoint_putanje = [glavni_checkpoint] + checkpoint_putanje
        if not checkpoint_putanje:
            raise FileNotFoundError(f"Nije pronađen nijedan checkpoint po obrascu {obrazac}.")
        print(f"[INFO] Pronađeno {len(checkpoint_putanje)} checkpoint-a: {checkpoint_putanje}")
        analiza_lead_occlusion_kfold(
            args.config, checkpoint_putanje, args.klasa, args.broj_primera, args.izlazni_direktorijum,
        )
    elif args.analiza == "lead_occlusion":
        analiza_lead_occlusion(
            args.config, args.checkpoint, args.klasa, args.broj_primera, args.izlazni_direktorijum,
        )
    else:
        glavna(args.config, args.checkpoint, args.broj_primera, args.izlazni_direktorijum)
