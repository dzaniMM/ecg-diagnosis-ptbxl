"""
data_loading.py
================
Učitavanje PTB-XL skupa podataka i mapiranje SCP-ECG dijagnostičkih kodova
u 5 standardnih superklasa: NORM, MI, STTC, CD, HYP.

Ovo je nastavak MIT-BIH projekta: umesto pojedinačnih otkucaja (187 tačaka,
1 kanal), ovde radimo sa celim 10-sekundnim, 12-kanalnim EKG snimcima, a
zadatak postaje MULTI-LABEL klasifikacija (jedan pacijent može imati više
istovremenih dijagnoza), za razliku od MIT-BIH gde je svaki otkucaj imao
tačno jednu klasu.
"""

import ast
import os

import numpy as np
import pandas as pd
import wfdb

SUPERKLASE = ["NORM", "MI", "STTC", "CD", "HYP"]


def ucitaj_metapodatke(putanja_do_ptbxl: str) -> pd.DataFrame:
    """
    Učitava ptbxl_database.csv i parsira scp_codes kolonu (koja je snimljena
    kao string reprezentacija Python rečnika) u pravi dict.
    """
    csv_path = os.path.join(putanja_do_ptbxl, "ptbxl_database.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Nije pronađen {csv_path}. Proveri da li si preuzeo PTB-XL "
            f"(pogledaj scripts/download_ptbxl.sh) i da je putanja tačna."
        )

    df = pd.read_csv(csv_path, index_col="ecg_id")
    df.scp_codes = df.scp_codes.apply(lambda x: ast.literal_eval(x))
    return df


def ucitaj_scp_mapiranje(putanja_do_ptbxl: str) -> pd.DataFrame:
    """Učitava scp_statements.csv koji sadrži mapiranje SCP kod -> superklasa."""
    csv_path = os.path.join(putanja_do_ptbxl, "scp_statements.csv")
    agg_df = pd.read_csv(csv_path, index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]
    return agg_df


def mapiraj_u_superklase(scp_dict: dict, agg_df: pd.DataFrame) -> list:
    """
    Za dati rečnik SCP kodova jednog pacijenta (npr. {'NDT': 100.0, 'PVC': 0.0})
    vraća listu diagnostic superklasa kojima ti kodovi pripadaju.
    Pacijent može imati 0, 1 ili više superklasa (multi-label).
    """
    superklase = set()
    for scp_kod in scp_dict.keys():
        if scp_kod in agg_df.index:
            superklase.add(agg_df.loc[scp_kod].diagnostic_class)
    return list(superklase)


def izracunaj_pouzdanost_klase(scp_dict: dict, agg_df: pd.DataFrame, klasa: str) -> float:
    """
    Max pouzdanost (0.0-1.0) da uzorak pripada zadatoj superklasi, na osnovu
    PTB-XL 'likelihood' vrednosti u scp_codes (0-100, procenat pouzdanosti
    anotacije). Ako više kodova pripada istoj superklasi, uzima se najveća.

    PTB-XL konvencija: likelihood=0 znači "pouzdanost nije eksplicitno
    navedena" (kod je ipak zvanično anotiran kao prisutan), NE "dijagnoza
    odsutna" - zato se tretira kao puna pouzdanost (1.0), a ne kao 0.0.
    (Potvrđeno na podacima: ~47% svih anotacija ima likelihood=0, što bi
    bilo besmisleno da zaista znači "odsutno".)
    """
    najbolja = 0.0
    for scp_kod, likelihood in scp_dict.items():
        if scp_kod not in agg_df.index:
            continue
        if agg_df.loc[scp_kod].diagnostic_class != klasa:
            continue
        pouzdanost = 1.0 if likelihood == 0 else likelihood / 100.0
        najbolja = max(najbolja, pouzdanost)
    return najbolja


def napravi_multihot_labele(df: pd.DataFrame, agg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Dodaje binarne kolone NORM, MI, STTC, CD, HYP (multi-hot 0/1, na osnovu
    df['diagnostic_superclass']) i njihove "meke" (_meke) verzije (0.0-1.0,
    na osnovu PTB-XL confidence/likelihood anotacije) - tvrde labele se
    koriste za validaciju/test, a meke se koriste kao trening signal (label smoothing na osnovu stvarne
    pouzdanosti anotacije umesto tretiranja svake dijagnoze kao 100% sigurne).

    Redovi bez ijedne prepoznate superklase se izbacuju.
    """
    df = df[df.diagnostic_superclass.apply(len) > 0].copy()

    for klasa in SUPERKLASE:
        df[klasa] = df.diagnostic_superclass.apply(lambda lst: 1 if klasa in lst else 0)
        df[f"{klasa}_meka"] = df.scp_codes.apply(lambda d: izracunaj_pouzdanost_klase(d, agg_df, klasa))

    return df


def ucitaj_signale(df: pd.DataFrame, sampling_rate: int, putanja_do_ptbxl: str) -> np.ndarray:
    """
    Učitava sirove EKG signale za redove iz df, koristeći wfdb.
    sampling_rate: 100 ili 500 (Hz), mora odgovarati filename_lr/filename_hr koloni.

    Vraća numpy niz oblika [N, samples, 12] (12 standardnih odvoda).
    """
    if sampling_rate == 100:
        putanje = df.filename_lr
    elif sampling_rate == 500:
        putanje = df.filename_hr
    else:
        raise ValueError("sampling_rate mora biti 100 ili 500")

    # Unapred alociramo float32 niz i punimo ga in-place, umesto da gradimo
    # Python listu pa je na kraju konvertujemo - wfdb.rdsamp vraća float64,
    # pa bi lista + finalni np.array() privremeno držali DVE kopije celog
    # dataseta u memoriji (float64 + float32) - na 500Hz to lako puca RAM.
    dužina_signala = sampling_rate * 10  # 10-sekundni snimci
    signali = np.empty((len(putanje), dužina_signala, 12), dtype=np.float32)
    for i, putanja in enumerate(putanje):
        record = wfdb.rdsamp(os.path.join(putanja_do_ptbxl, putanja))
        signali[i] = record[0]  # record[0] su sirovi podaci, record[1] su metapodaci

    return signali


def napravi_train_val_test_split(df: pd.DataFrame):
    """
    Koristi ZVANIČNU preporučenu podelu PTB-XL skupa (kolona strat_fold, 1-10),
    stratifikovanu tako da je distribucija dijagnoza slična u svim foldovima.

    Preporuka autora skupa:
      - foldovi 1-8: trening
      - fold 9:      validacija
      - fold 10:     test (najčistiji, najpouzdaniji anotacije)
    """
    train_df = df[df.strat_fold <= 8]
    val_df = df[df.strat_fold == 9]
    test_df = df[df.strat_fold == 10]
    return train_df, val_df, test_df


def pripremi_kompletan_dataset(putanja_do_ptbxl: str, sampling_rate: int = 100):
    """
    Glavna funkcija koja spaja sve gornje korake:
    1. učitava metapodatke
    2. mapira SCP kodove u superklase
    3. pravi multi-hot labele
    4. deli na train/val/test po zvaničnim foldovima
    5. učitava sirove signale za svaki split

    Vraća: dict sa X_train, y_train, X_val, y_val, X_test, y_test (numpy nizovi)
    """
    print("[INFO] Učitavam metapodatke...")
    df = ucitaj_metapodatke(putanja_do_ptbxl)
    agg_df = ucitaj_scp_mapiranje(putanja_do_ptbxl)

    print("[INFO] Mapiram SCP kodove u dijagnostičke superklase (NORM/MI/STTC/CD/HYP)...")
    df["diagnostic_superclass"] = df.scp_codes.apply(lambda x: mapiraj_u_superklase(x, agg_df))
    df = napravi_multihot_labele(df, agg_df)

    print(f"[INFO] Preostalo {len(df)} snimaka nakon filtriranja onih bez dijagnostičke labele.")
    print("[INFO] Distribucija klasa (multi-label, procenti mogu sumirati > 100%):")
    for klasa in SUPERKLASE:
        procenat = 100 * df[klasa].mean()
        print(f"  {klasa}: {df[klasa].sum():.0f} uzoraka ({procenat:.2f}%)")

    train_df, val_df, test_df = napravi_train_val_test_split(df)

    print(f"[INFO] Učitavam sirove signale (sampling_rate={sampling_rate}Hz)... ovo može potrajati.")
    X_train = ucitaj_signale(train_df, sampling_rate, putanja_do_ptbxl)
    X_val = ucitaj_signale(val_df, sampling_rate, putanja_do_ptbxl)
    X_test = ucitaj_signale(test_df, sampling_rate, putanja_do_ptbxl)

    kolone_meke = [f"{k}_meka" for k in SUPERKLASE]
    y_train = train_df[SUPERKLASE].values.astype(np.float32)
    y_train_meke = train_df[kolone_meke].values.astype(np.float32)
    y_val = val_df[SUPERKLASE].values.astype(np.float32)
    y_test = test_df[SUPERKLASE].values.astype(np.float32)

    return {
        "X_train": X_train, "y_train": y_train, "y_train_meke": y_train_meke,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "klase": SUPERKLASE,
    }


def pripremi_kfold_dataset(putanja_do_ptbxl: str, sampling_rate: int = 100):
    """
    Priprema podatke za k-fold unakrsnu validaciju po zvaničnom PTB-XL protokolu:
    foldovi 1-9 ostaju zajedno (sa fold brojem po uzorku) da bi train_kfold.py
    mogao da rotira koji od njih služi kao validacija, dok fold 10 ostaje
    netaknut kao konačni test skup - isto kao u pripremi_kompletan_dataset,
    samo bez fiksnog train/val razdvajanja unutar foldova 1-9.

    Vraća: dict sa X_trainval, y_trainval, foldovi_trainval (broj folda 1-9
    po uzorku), X_test, y_test.
    """
    print("[INFO] Učitavam metapodatke...")
    df = ucitaj_metapodatke(putanja_do_ptbxl)
    agg_df = ucitaj_scp_mapiranje(putanja_do_ptbxl)

    print("[INFO] Mapiram SCP kodove u dijagnostičke superklase (NORM/MI/STTC/CD/HYP)...")
    df["diagnostic_superclass"] = df.scp_codes.apply(lambda x: mapiraj_u_superklase(x, agg_df))
    df = napravi_multihot_labele(df, agg_df)

    trainval_df = df[df.strat_fold <= 9]
    test_df = df[df.strat_fold == 10]

    print(f"[INFO] Učitavam sirove signale (sampling_rate={sampling_rate}Hz)... ovo može potrajati.")
    X_trainval = ucitaj_signale(trainval_df, sampling_rate, putanja_do_ptbxl)
    X_test = ucitaj_signale(test_df, sampling_rate, putanja_do_ptbxl)

    kolone_meke = [f"{k}_meka" for k in SUPERKLASE]
    y_trainval = trainval_df[SUPERKLASE].values.astype(np.float32)
    y_trainval_meke = trainval_df[kolone_meke].values.astype(np.float32)
    y_test = test_df[SUPERKLASE].values.astype(np.float32)
    foldovi_trainval = trainval_df.strat_fold.values.astype(np.int64)

    return {
        "X_trainval": X_trainval, "y_trainval": y_trainval, "y_trainval_meke": y_trainval_meke,
        "foldovi_trainval": foldovi_trainval,
        "X_test": X_test, "y_test": y_test,
        "klase": SUPERKLASE,
    }


if __name__ == "__main__":
    # Brzi test učitavanja 
    podaci = pripremi_kompletan_dataset("data/raw/ptb-xl", sampling_rate=100)
    print("\n[INFO] Oblici nizova:")
    for kljuc in ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]:
        print(f"  {kljuc}: {podaci[kljuc].shape}")
