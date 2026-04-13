"""Génère Presentation.pdf à partir du contenu de soutenance (français)."""
from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Presentation.pdf"


class PDF(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 9)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


def main():
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(0, 10, "Detection de fraude par carte bancaire", align="C")
    pdf.ln(4)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(
        0,
        7,
        "Projet Machine Learning - soutenance\nPolytech, Semestre 4 - Avril 2026",
    )
    pdf.ln(8)

    # Texte limite au Latin-1 pour les polices core Helvetica
    slides = [
        (
            "Contexte",
            "Fraude par carte : enjeux financiers, conformite, confiance client.\n"
            "Dataset creditcard.csv : ~285k transactions, variables Time, Amount, V1-V28 (PCA), cible Class.",
        ),
        (
            "Probleme",
            "Classification binaire supervisee.\n"
            "Desequilibre fort : F1, ROC-AUC et precision/rappel sur la fraude priment sur l'accuracy.",
        ),
        (
            "Pipeline",
            "EDA puis preprocessing : standardisation, split stratifie 80/20.\n"
            "SMOTE dans le pipeline (imblearn), uniquement sur le train.",
        ),
        (
            "Modeles",
            "Regression logistique, arbre, Random Forest (GridSearch F1),\n"
            "Gradient boosting (XGBoost ou HistGradientBoosting).",
        ),
        (
            "Resultats",
            "Voir outputs/model_results.csv.\n"
            "Random Forest optimise : bon compromis F1 / ROC-AUC (typique).",
        ),
        (
            "Limites",
            "Donnees 2013 ; variables anonymisees (PCA).\n"
            "Validation temporelle et couts metier a traiter en production.",
        ),
        (
            "Conclusion",
            "Projet reproductible : requirements.txt, notebooks 01_EDA et 02_Modeling.\n"
            "Outil d'aide a la decision, pas un substitut a la gouvernance du risque.",
        ),
    ]

    for title, body in slides:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, title, ln=True)
        pdf.ln(2)
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 7, body)

    pdf.output(OUT)
    print(f"Ecrit : {OUT}")


if __name__ == "__main__":
    main()
