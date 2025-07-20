import streamlit as st
from PIL import Image
import pandas as pd
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt
from io import BytesIO
from fpdf import FPDF
import base64

# ========== CONFIGURATION ========== #
st.set_page_config(
    page_title="Crédit Pour Tous",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== CHARGEMENT DU MODÈLE ========== #
model = joblib.load("model/xgb_pipeline.pkl")
features_list = model.feature_names_in_

# ========== LOGO & LANGUES ========== #
logo_path = "static/logo.png"
if logo_path:
    st.sidebar.image(Image.open(logo_path), width=150)

language = st.sidebar.radio("🌐 Choisissez votre langue / Choose your language", ("Français", "English"))

# ========== TRADUCTIONS ========== #
text = {
    "Français": {
        "title": "💳 Prédiction de défaut de paiement",
        "predict": "Faire une prédiction",
        "form": "📝 Formulaire individuel",
        "batch": "📁 Mode batch (CSV)",
        "shap": "📊 Interprétation SHAP",
        "jury": "🧾 Mode Jury",
        "download": "📥 Télécharger le rapport PDF",
        "submit": "Soumettre",
    },
    "English": {
        "title": "💳 Payment Default Prediction",
        "predict": "Make a Prediction",
        "form": "📝 Individual Form",
        "batch": "📁 Batch Mode (CSV)",
        "shap": "📊 SHAP Interpretation",
        "jury": "🧾 Jury Mode",
        "download": "📥 Download PDF Report",
        "submit": "Submit",
    }
}[language]

st.title(text["title"])

# ========== FORMULAIRE DE PRÉDICTION ========== #
st.header(text["form"])

with st.form("prediction_form"):
    inputs = {}
    for feat in features_list:
        if feat in ["int_rate", "dti", "revol_util"]:
            inputs[feat] = st.slider(f"{feat}", 0.0, 60.0, 15.0)
        elif feat in ["term"]:
            inputs[feat] = st.selectbox(f"{feat}", [36, 60])
        else:
            inputs[feat] = st.number_input(f"{feat}", value=1000)
    submitted = st.form_submit_button(text["submit"])

if submitted:
    X_input = pd.DataFrame([inputs])
    prediction = model.predict(X_input)[0]
    proba = model.predict_proba(X_input)[0][1]

    st.success(f"✅ {text['predict']} : {'Défaut probable' if prediction==1 else 'Pas de défaut'} ({proba:.2%})")

# ========== INTERPRÉTATION SHAP ========== #
st.header(text["shap"])
explainer = shap.Explainer(model)
shap_values = explainer(X_input)
st.set_option('deprecation.showPyplotGlobalUse', False)
shap.plots.bar(shap_values[0], max_display=10)
st.pyplot(bbox_inches="tight")

# ========== MODE JURY PDF ========== #
st.header(text["jury"])
st.markdown("Résumé du projet, courbes ROC, score modèle, etc. à venir...")

def generate_pdf():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Crédit Pour Tous - Résumé Jury", ln=1, align="C")
    pdf.cell(200, 10, txt=f"Résultat : {'Défaut' if prediction else 'Aucun défaut'}", ln=2, align="L")
    pdf.cell(200, 10, txt=f"Probabilité : {proba:.2%}", ln=3, align="L")
    return pdf

if st.button(text["download"]):
    pdf = generate_pdf()
    b = BytesIO()
    pdf.output(b)
    st.download_button(label=text["download"],
                       data=b.getvalue(),
                       file_name="rapport_credit_jury.pdf",
                       mime="application/pdf")

# ========== MODE BATCH ========== #
st.header(text["batch"])
uploaded_file = st.file_uploader("Uploader un fichier CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    preds = model.predict(df)
    df["prediction"] = preds
    st.dataframe(df.head())
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Télécharger résultats", data=csv, file_name="predictions.csv", mime="text/csv")
