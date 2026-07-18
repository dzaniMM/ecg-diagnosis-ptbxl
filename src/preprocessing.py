"""
preprocessing.py
=================
Filtriranje, normalizacija signala i PyTorch Dataset klasa za PTB-XL.

Za razliku od MIT-BIH gde su otkucaji već bili isečeni i centrirani oko
R-pika, PTB-XL sirovi signali sadrže šum, pomeranje bazne linije (baseline
wander) i nisu prethodno segmentirani na otkucaje - radimo sa celim
10-sekundnim prozorom odjednom.
"""

import numpy as np
import torch
from scipy import signal as scipy_signal
from torch.utils.data import Dataset


def ukloni_baseline_wander(ekg_signal: np.ndarray, sampling_rate: int = 100) -> np.ndarray:
    """
    Uklanja nisko-frekventno pomeranje bazne linije pomoću high-pass
    ekg_signal: oblik [samples, channels]
    """
    nyquist = sampling_rate / 2.0
    granica = 0.5 / nyquist
    b, a = scipy_signal.butter(N=3, Wn=granica, btype="highpass")
    filtriran = scipy_signal.filtfilt(b, a, ekg_signal, axis=0)
    return filtriran.astype(np.float32)


def ukloni_visokofrekventni_sum(ekg_signal: np.ndarray, sampling_rate: int = 100) -> np.ndarray:
    """Low-pass filter na 40Hz da ukloni mišićni šum i mrežnu smetnju (50/60Hz)."""
    nyquist = sampling_rate / 2.0
    granica = 40.0 / nyquist
    granica = min(granica, 0.99)  # sigurnosna provera za niže sampling rate-ove
    b, a = scipy_signal.butter(N=4, Wn=granica, btype="lowpass")
    filtriran = scipy_signal.filtfilt(b, a, ekg_signal, axis=0)
    return filtriran.astype(np.float32)


def normalizuj_po_kanalu(ekg_signal: np.ndarray) -> np.ndarray:
    """Z-score normalizacija svakog od 12 kanala nezavisno (mean=0, std=1)."""
    mean = ekg_signal.mean(axis=0, keepdims=True)
    std = ekg_signal.std(axis=0, keepdims=True) + 1e-8
    return (ekg_signal - mean) / std


def predobradi_signal(ekg_signal: np.ndarray, sampling_rate: int = 100) -> np.ndarray:
    """Kompletan pipeline predobrade za jedan snimak: filtriranje + normalizacija."""
    s = ukloni_baseline_wander(ekg_signal, sampling_rate)
    s = ukloni_visokofrekventni_sum(s, sampling_rate)
    s = normalizuj_po_kanalu(s)
    return s


def predobradi_dataset(X: np.ndarray, sampling_rate: int = 100) -> np.ndarray:
    """
    Primenjuje predobradu na ceo niz snimaka [N, samples, 12].

    Puni se in-place u unapred alociran niz (ne gradi se Python lista pa
    konvertuje na kraju) - lista bi privremeno držala dodatnu kopiju celog
    dataseta u memoriji, pored ulaznog X i izlaznog niza.
    """
    izlaz = np.empty_like(X, dtype=np.float32)
    for i in range(len(X)):
        izlaz[i] = predobradi_signal(X[i], sampling_rate)
    return izlaz


class PTBXLDataset(Dataset):
    """
    PyTorch Dataset za PTB-XL.

    X: numpy niz oblika [N, samples, 12] (nakon predobrade)
    y: numpy niz oblika [N, 5] (multi-hot labele: NORM, MI, STTC, CD, HYP)

    Vraća tenzore u obliku [12, samples] (Channels, Length) - format koji
    PyTorch Conv1d očekuje - i [5] float labelu za multi-label BCE loss.

    lead_masking_p: verovatnoća (po uzorku, po epohi) da se primeni random
    lead masking - gasi se (postavlja na 0) lead_masking_broj nasumično
    izabranih odvoda. Motivacija: lead occlusion analiza je pokazala da
    model prekomerno zavisi od pojedinih odvoda (npr. V1 za HYP) dok druge
    (V4/V5/V6) koristi kao suzbijajući, ne pozitivan dokaz - lead masking
    tera model da nauči redundantnije, ravnomernije korišćenje svih 12
    odvoda. Podrazumevano isključeno (p=0) - treba ga eksplicitno uključiti
    SAMO za trening skup, nikad za validaciju/test.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray,
                 lead_masking_p: float = 0.0, lead_masking_broj: int = 1):
        assert len(X) == len(y), "X i y moraju imati isti broj uzoraka"
        # Transponujemo iz [N, samples, 12] u [N, 12, samples]
        self.X = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.lead_masking_p = lead_masking_p
        self.lead_masking_broj = lead_masking_broj

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.lead_masking_p > 0 and torch.rand(1).item() < self.lead_masking_p:
            x = x.clone()
            broj = min(self.lead_masking_broj, x.shape[0])
            odvodi = torch.randperm(x.shape[0])[:broj]
            x[odvodi, :] = 0.0
        return x, self.y[idx]
