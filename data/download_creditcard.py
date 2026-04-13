"""
Téléchargement optionnel de creditcard.csv.

Par défaut : vérifie la présence du fichier et affiche les instructions.
Pour télécharger depuis Kaggle : installez `kaggle`, configurez ~/.kaggle/kaggle.json,
puis : kaggle datasets download -c ieee-fraud-detection  (ou le dataset ULB creditcardfraud).
Le jeu classique est souvent nommé creditcard.csv sur le dépôt mlg-ulb/creditcardfraud.
"""
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
TARGET = DATA_DIR / "creditcard.csv"

if TARGET.exists():
    print(f"OK : {TARGET} est déjà présent ({TARGET.stat().st_size // 1024 // 1024} Mo environ).")
else:
    print("Fichier absent. Étapes possibles :")
    print("  1. Télécharger creditcard.csv depuis https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
    print("  2. Placer le fichier dans :", DATA_DIR)
    print("  3. Relancer les notebooks depuis le dossier notebooks/")
