"""
losses.py
=========
Focal Loss - nastavak koncepta iz MIT-BIH faze, prilagođen za MULTI-LABEL
klasifikaciju (PTB-XL). Kod MIT-BIH je Focal Loss radio nad Softmax
verovatnoćama i CrossEntropy-jem (jedan otkucaj = tačno jedna klasa).

Kod PTB-XL, klase nisu uzajamno isključive (pacijent može imati i MI i CD
istovremeno), pa se koristi Binary Cross-Entropy (BCE) po svakoj od 5 klasa
nezavisno, sa istim "focal" mehanizmom prigušivanja lakih uzoraka:

    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

gde je p_t verovatnoća predviđena za TAČNU vrednost te binarne labele
(bilo 0 ili 1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLossMultiLabel(nn.Module):
    """
    Multi-label Focal Loss zasnovan na BCEWithLogits.

    Args:
        alpha: težina za balansiranje pozitivnih/negativnih uzoraka. Može biti
               jedan broj (ista vrednost za svih 5 klasa) ili lista/tenzor
               dužine num_classes, sa po-klasa vrednošću u redosledu
               SUPERKLASE = [NORM, MI, STTC, CD, HYP] (npr. veći alpha za
               ređe klase poput HYP, da im pozitivni primeri ne bi bili
               zagušeni gradijentom od dominantne NORM klase).
        gamma: stepen prigušivanja lakih (dobro klasifikovanih) uzoraka.
               gamma=0 svodi Focal Loss na običan (težinski) BCE.
        pos_weight: opciono, per-klasa dodatna težina za pozitivnu klasu -
                    korisno ako je npr. MI mnogo ređi od NORM.
    """

    def __init__(self, alpha=0.25, gamma: float = 2.0, pos_weight=None):
        super().__init__()
        self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32))
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )

        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)  # verovatnoća tačne klase
        focal_faktor = (1 - p_t) ** self.gamma

        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        loss = alpha_t * focal_faktor * bce
        return loss.mean()


if __name__ == "__main__":
    logits = torch.randn(8, 5)
    targets = torch.randint(0, 2, (8, 5)).float()

    fl_skalar = FocalLossMultiLabel(alpha=0.25, gamma=2.0)
    print(f"Focal Loss (skalarni alpha=0.25): {fl_skalar(logits, targets).item():.4f}")

    # po-klasa alpha, redosled [NORM, MI, STTC, CD, HYP]
    fl_vektor = FocalLossMultiLabel(alpha=[0.25, 0.45, 0.45, 0.50, 0.65], gamma=2.0)
    print(f"Focal Loss (vektorski alpha po klasi): {fl_vektor(logits, targets).item():.4f}")

    bce_obican = nn.BCEWithLogitsLoss()
    print(f"Obican BCE (za poredjenje): {bce_obican(logits, targets).item():.4f}")
