"""
models.py
=========

  1. in_channels: 1 -> 12   (single-lead -> 12-kanalni EKG)
  2. izlazni sloj: Softmax nad 5 uzajamno isključivih klasa
                   -> Sigmoid nad 5 NEZAVISNIH binarnih labela (multi-label),
                      jer pacijent u PTB-XL može imati više dijagnoza istovremeno
                      (npr. i MI i CD u istom snimku).

Dužina signala: 10s @ 100/500Hz 

`ResidualBlock1D` (deljen od sva tri modela) ima ugrađenu
Squeeze-and-Excitation (SE) kanalsku pažnju - mreža uči koji feature kanal
(posredno povezan sa EKG odvodima) da pojača/priguši u zavisnosti od
konteksta, umesto da sve kanale tretira ravnopravno.
"""

import math

import torch
import torch.nn as nn


class SEBlock1D(nn.Module):
    """
    Squeeze-and-Excitation blok za 1D podatke. Uči koji kanal (odvod/feature
    mapa) treba pojačati a koji prigušiti, u zavisnosti od globalnog konteksta
    celog signala (Hu et al., 2018).
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.LeakyReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _ = x.size()
        weights = self.fc(x).view(b, c, 1)
        return x * weights


class ResidualBlock1D(nn.Module):
    """
    Rezidualni 1D konvolucioni blok sa shortcut vezom sa Squeeze-and-Excitation kanalskom pažnjom 
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, se_reduction: int = 4):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.se = SEBlock1D(out_channels, reduction=se_reduction)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += residual
        out = self.relu(out)
        return out


class PTBXL_CNN(nn.Module):
    """
    Multi-lead CNN sa rezidualnim blokovima
    """

    def __init__(self, in_channels: int = 12, num_classes: int = 5, dropout: float = 0.2):
        super().__init__()

        self.init_conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
        )

        self.layer1 = ResidualBlock1D(32, 32, stride=1)
        self.pool1 = nn.MaxPool1d(kernel_size=5, stride=2, padding=2)  # 1000 -> 500

        self.layer2 = ResidualBlock1D(32, 64, stride=1)
        self.pool2 = nn.MaxPool1d(kernel_size=5, stride=2, padding=2)  # 500 -> 250

        self.layer3 = ResidualBlock1D(64, 128, stride=1)
        self.pool3 = nn.AdaptiveAvgPool1d(1)  # 250 -> 1

        self.fc = nn.Sequential(
            nn.Linear(128, 32),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        # x: [Batch, 12, samples]
        x = self.init_conv(x)
        x = self.pool1(self.layer1(x))
        x = self.pool2(self.layer2(x))
        x = self.pool3(self.layer3(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x  # logiti - sigmoid se primenjuje u loss funkciji (BCEWithLogits)


class PositionalEncoding(nn.Module):
    """Standardno sinusoidno pozicijsko kodiranje (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        # x: [Batch, Time, d_model]
        return x + self.pe[:, :x.size(1)]


class AttentionPooling1D(nn.Module):
    """
    Naučeno pooling kroz vreme - umesto uniformnog proseka, model uči skor
    (relevantnost) po vremenskom koraku, pa se finalni vektor računa kao
    težinski zbir (meko biranje GDE u signalu je dokaz najjači), umesto da
    se sve vremenske pozicije tretiraju podjednako.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: [Batch, Time, d_model]
        scores = self.score(x)                  # [Batch, Time, 1]
        weights = torch.softmax(scores, dim=1)   # [Batch, Time, 1]
        return (x * weights).sum(dim=1)          # [Batch, d_model]


class PTBXL_Transformer(nn.Module):
    """
    Hibridna CNN + Transformer arhitektura. Isti rezidualni CNN uvod kao
    kod PTBXL_CNN/PTBXL_BiGRU izvlači morfologiju i skraćuje sekvencu, a
    Transformer encoder (self-attention) uči globalne zavisnosti duž cele
    sekvence direktno (svaki korak "gleda" sve ostale odjednom), za razliku
    od BiGRU-a koji ih prenosi sekvencijalno kroz skrivena stanja.

    Ima četiri rezidualna bloka (jedan više nego PTBXL_BiGRU), jer je
    self-attention kvadratne složenosti po dužini sekvence - bitno je
    ući u transformer sa što kraćom, ali još uvek informativnom sekvencom.
    """

    def __init__(self, in_channels: int = 12, d_model: int = 128, nhead: int = 4,
                 num_layers: int = 2, num_classes: int = 5, dropout: float = 0.3):
        super().__init__()

        self.init_conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )

        self.layer1 = ResidualBlock1D(32, 32, stride=2)        # 5000 -> 2500
        self.layer2 = ResidualBlock1D(32, 64, stride=2)        # 2500 -> 1250
        self.layer3 = ResidualBlock1D(64, 128, stride=2)       # 1250 -> 625
        self.layer4 = ResidualBlock1D(128, d_model, stride=2)  # 625 -> ~313

        self.pos_encoding = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.attn_pool = AttentionPooling1D(d_model)

        self.fc = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        # x: [Batch, 12, samples]
        x = self.init_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)              # [Batch, d_model, ~313]
        x = x.permute(0, 2, 1)          # [Batch, ~313, d_model]
        x = self.pos_encoding(x)
        x = self.transformer(x)         # [Batch, ~313, d_model]
        pooled = self.attn_pool(x)      # naučeni attention-pooling nad vremenom
        logits = self.fc(pooled)
        return logits


class PTBXL_BiGRU(nn.Module):
    """
    Hibridna CNN + BiGRU arhitektura sa rezidualnim blokovima - nastavak
    `MITBIH_BiGRU_Max`. Ima treći rezidualni blok (stride=2) više nego
    MIT-BIH verzija, jer je PTB-XL signal (1000 tačaka) mnogo duži od
    pojedinačnog otkucaja kod MIT-BIH (187 tačaka).

    Koristi MAX pooling nad vremenskom osom GRU izlaza (isto kao
    `MITBIH_BiGRU_Max`). Napomena: eksperimentisali smo i sa average
    pooling-om (manje osetljiv na overfitting u kombinaciji sa
    WeightedRandomSampler-om), ali je vraćeno na max.
    """

    def __init__(self, in_channels: int = 12, hidden_size: int = 128,
                 num_layers: int = 1, num_classes: int = 5, dropout: float = 0.3):
        super().__init__()

        self.init_conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )

        # svaka faza: rezidualni blok (stride=2) + MaxPool1d (stride=2) => 4x smanjenje po fazi
        self.layer1 = ResidualBlock1D(32, 32, stride=2)
        self.pool1 = nn.MaxPool1d(kernel_size=5, stride=2, padding=2)   # 5000 -> 1250
        self.layer2 = ResidualBlock1D(32, 64, stride=2)
        self.pool2 = nn.MaxPool1d(kernel_size=5, stride=2, padding=2)   # 1250 -> 313
        self.layer3 = ResidualBlock1D(64, 128, stride=2)
        self.pool3 = nn.MaxPool1d(kernel_size=5, stride=2, padding=2)   # 313 -> 79

        self.gru = nn.GRU(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        # x: [Batch, 12, samples]
        x = self.init_conv(x)
        x = self.pool1(self.layer1(x))
        x = self.pool2(self.layer2(x))
        x = self.pool3(self.layer3(x))  # [Batch, 128, ~79]
        x = x.permute(0, 2, 1)          # [Batch, ~79, 128]
        out, _ = self.gru(x)     # [Batch, ~79, hidden*2]
        out, _ = out.max(dim=1)    # globalni max pooling nad vremenom -> [Batch, hidden*2]
        logits = self.fc(out)
        return logits


if __name__ == "__main__":
    # Brza provera da dimenzije prolaze kroz mrežu bez grešaka
    dummy_input = torch.randn(4, 12, 5000)  # batch=4, 12 odvoda, 5000 tačaka (10s@500Hz)

    cnn = PTBXL_CNN()
    out_cnn = cnn(dummy_input)
    print(f"PTBXL_CNN izlaz: {out_cnn.shape}  (očekivano: [4, 5])")

    bigru = PTBXL_BiGRU()
    out_bigru = bigru(dummy_input)
    print(f"PTBXL_BiGRU izlaz: {out_bigru.shape}  (očekivano: [4, 5])")

    transformer = PTBXL_Transformer()
    out_transformer = transformer(dummy_input)
    print(f"PTBXL_Transformer izlaz: {out_transformer.shape}  (očekivano: [4, 5])")
