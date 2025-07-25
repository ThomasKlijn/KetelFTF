from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, FileResponse
import pandas as pd
import numpy as np
import joblib
from io import BytesIO
import openpyxl
from openpyxl.styles import PatternFill
import uvicorn

app = FastAPI()

@app.get("/")
async def read_root():
    return FileResponse("index.html")

# Modellen en vectorizers laden
ketel_model = joblib.load("ketel_model_Vfinal.joblib")
ketel_vec = joblib.load("ketel_vectorizer_Vfinal.joblib")
ftf_model = joblib.load("ftf_model_Vfinal.joblib")
ftf_vec = joblib.load("ftf_vectorizer_Vfinal.joblib")

# Uitgebreide keywordlijst Ketel gerelateerd met context (j of n)
keyword_kg = {
    "Flowsensor vervangen": "j",
    "aansturing ivm 9U": "j",
    "aansturing Nefit": "j",
    "aansturing ivm 9P": "j",
    "lek hydroblok": "j",
    "defecte KIM": "j",
    "platenwisselaar vervangen": "j",
    "Het hydroblok was lek": "j",
    "RGA beugelen": "n",
    "RGA gebeugeld": "n",
    "Druksensor vervangen": "j",
    "Condensafvoer": "n",
    "Hoofdprint vervangen": "j",
    "ontstek pen vv": "j",
    "thermostaat van de klant is niet goed": "n",
    "wartel bij de pomp": "j",
    "Ontsteekpen vervangen": "j",
    "Condensafvoer herstellen.": "n",
    "nieuwe afvoer maken": "n",
    "Flowsensor vv": "j",
    "Sensorflow vervangen": "j",
    "Sensorflow vv": "j",
    "Expansievat vervangen": "n",
    "RGA aanpassen": "n",
    "ltv(Luchttoevoer) vervangen": "n",
    "verroest van binnen": "j",
    "Batterijen vervangen": "n",
    "Wisselaar is lek": "j",
    "Ketel bij gevuld": "n",
    "Pomp zat vast": "j",
    "geen gas toevoer": "n",
    "Batterij van de koolmonoxide melder": "n",
    "Vaillant pomp aansluitkabel": "j",
    "Vaillant druksensor": "j",
    "thermostaat": "n",
    "Vulkraan vervangen": "n",
    "Stekkertje ontsteekkabel": "n",
    "RGA herstellen": "n",
    "RGA vervangen en gebeugeld": "n",
    "LTV (luchttoevoer) gebeugeld": "n",
    "Warmtewisselaar vervangen": "j",
    "Sensor flow vervangen": "j",
    "Sam trechter vervangen": "n",
    "RGA en LTV + dakdoorvoer aanpassen.": "n",
    "rookgas vervangen": "n",
    "Bijgevuld": "n",
    "druksensor defect": "j",
    "Alle radiatorkranen dicht": "n",
    "Gasblok vervangen": "j",
    "Wasmachinekraan vervangen": "n",
    "lijkt op print te zijn.": "j",
    "overstort druppelde": "n",
    "Overstort en wasmachine kraan": "n",
    "Overstort vervangen": "n",
    "Uitgevoerde werkzaamheden: ketel afgekeurd": "j",
    "Spoed - Ketel afgekeurd - nog vervangen": "j",
    "Ketel afgekeurd - nog vervangen": "j",
    "thermostaat vervangen": "n",
    "Afuizing badkamer": "n",
    "Kamerthermostaat vervangen": "n"
}

def keyword_based_ketel(text: str):
    text_lower = text.lower()
    for key, val in keyword_kg.items():
        # Exact phrase match, case-insensitive
        if key.lower() in text_lower:
            return val
    return None

# Keywordlijsten voor rule-based FTF filtering
positive_keywords = [
    "vervangen", "gerepareerd", "hersteld", "reset", "opgelost",
    "functioneert", "controle uitgevoerd", "storingscode verwijderd"
]
negative_keywords = [
    "afspraak", "moet worden nagekeken", "onderdeel besteld", 
    "kan niet oplossen", "opvolging", "storingscode blijft", "niet gelukt"
]

def keyword_based_ftf(text: str):
    text_lower = text.lower()
    if any(k in text_lower for k in positive_keywords):
        return "1"
    if any(k in text_lower for k in negative_keywords):
        return "0"
    return None

@app.post("/process_excel/")
async def process_excel(file: UploadFile = File(...)):
    contents = await file.read()
    df_prod = pd.read_excel(BytesIO(contents), sheet_name="Leeg")

    alle_kolommen = df_prod.columns.tolist()
    potentiele_oplossingskolommen = ["Oplossingen"] + [col for col in alle_kolommen if col.startswith("Unnamed:")]
    oplossingskolommen = [col for col in potentiele_oplossingskolommen if col in alle_kolommen]
    
    basis_tekstkolommen = [
        "Werkbeschrijving", "Werkbon is vervolg van",
        "Werkbon nummer", "Uitvoerdatum", "Object referentie", "Installatie apparaat omschrijving"
    ]
    tekstkolommen = [col for col in basis_tekstkolommen if col in alle_kolommen]

    if oplossingskolommen:
        df_prod[oplossingskolommen] = df_prod[oplossingskolommen].fillna("")
        df_prod["Oplossingen_samengevoegd"] = df_prod[oplossingskolommen].astype(str).agg(" ".join, axis=1)
    else:
        df_prod["Oplossingen_samengevoegd"] = ""

    tekstkolommen.append("Oplossingen_samengevoegd")
    df_prod[tekstkolommen] = df_prod[tekstkolommen].fillna("")
    df_prod["combined_text"] = df_prod[tekstkolommen].apply(lambda r: " ".join([str(x) for x in r]), axis=1)
    df_prod["heeft_vervolg"] = df_prod["Werkbon is vervolg van"].apply(lambda x: int(bool(str(x).strip())))
    df_prod["tekstlengte"] = df_prod["combined_text"].apply(len)
    df_prod["woordenaantal"] = df_prod["combined_text"].apply(lambda x: len(x.split()))
    df_prod["contains_onderdeel"] = df_prod["combined_text"].str.contains("onderdeel|vervangen|vervang", case=False).astype(int)
    df_prod["contains_reset"] = df_prod["combined_text"].str.contains("reset|herstart", case=False).astype(int)
    df_prod["contains_advies"] = df_prod["combined_text"].str.contains("advies|aanbeveling", case=False).astype(int)

    # Ketel gerelateerd: eerst keywords checken, anders ML model
    kg_keywords = df_prod["combined_text"].apply(keyword_based_ketel)
    mask_kg_keywords = kg_keywords.notnull()
    df_prod.loc[mask_kg_keywords, "Ketel gerelateerd"] = kg_keywords[mask_kg_keywords]

    # Voor de rest ML voorspellen
    mask_ml = ~mask_kg_keywords
    if mask_ml.any():
        X_text_k = df_prod.loc[mask_ml, "combined_text"]
        X_vec_k = ketel_vec.transform(X_text_k).toarray()
        extra_k = df_prod.loc[mask_ml, ["heeft_vervolg", "tekstlengte", "woordenaantal", "contains_onderdeel", "contains_reset", "contains_advies"]].values
        X_comb_k = np.hstack((X_vec_k, extra_k))

        y_proba_k = ketel_model.predict_proba(X_comb_k)
        y_pred_k = ketel_model.predict(X_comb_k)

        df_prod.loc[mask_ml, "Ketel gerelateerd"] = np.where(y_proba_k.max(axis=1) > 0.6, np.where(y_pred_k == 1, "j", "n"), "")

        # Ketel zekerheid toevoegen
        df_prod.loc[mask_ml, "Ketel zekerheid"] = y_proba_k.max(axis=1)

    # Vul zekerheid voor keyword rows (we kunnen hier 1 zetten, omdat keyword match zeker is)
    df_prod.loc[mask_kg_keywords, "Ketel zekerheid"] = 1.0

    # FTF voorspellen voor rijen waar Ketel gerelateerd = "j" (zonder zekerheid cutoff)
    mask_ftf = (df_prod["Ketel gerelateerd"] == "j")
    df_ftf = df_prod[mask_ftf].copy()

    # Eerst rule-based FTF toewijzen
    df_ftf["FTF_keyword"] = df_ftf["combined_text"].apply(keyword_based_ftf)

    # Rijen zonder rule-based FTF invullen met ML-voorspelling
    mask_ml_ftf = df_ftf["FTF_keyword"].isnull()
    X_text_f = df_ftf.loc[mask_ml_ftf, "combined_text"]
    if not X_text_f.empty:
        X_vec_f = ftf_vec.transform(X_text_f).toarray()
        extra_f = df_ftf.loc[mask_ml_ftf, ["heeft_vervolg", "tekstlengte", "woordenaantal", "contains_onderdeel", "contains_reset", "contains_advies"]].values
        X_comb_f = np.hstack((X_vec_f, extra_f))

        y_proba_f = ftf_model.predict_proba(X_comb_f)
        y_pred_f = ftf_model.predict(X_comb_f)

        # Gebruik threshold 0.6 voor FTF=1
        ftf_pred = np.where(y_proba_f.max(axis=1) > 0.6, y_pred_f.astype(str), "0")

        df_ftf.loc[mask_ml_ftf, "FTF_keyword"] = ftf_pred

    # Vul FTF kolom met rule-based + ML resultaten
    df_prod.loc[df_ftf.index, "FTF"] = df_ftf["FTF_keyword"].replace({"1": "1", "0": "", "": ""})

    # Voeg zekerheid toe voor ML voorspellingen alleen
    df_ftf["FTF zekerheid"] = 0
    if not X_text_f.empty:
        df_ftf.loc[mask_ml_ftf, "FTF zekerheid"] = y_proba_f.max(axis=1)
    df_prod.loc[df_ftf.index, "FTF zekerheid"] = df_ftf["FTF zekerheid"]

    # Kleurcodering in output Excel
    wb = openpyxl.Workbook()
    ws = wb.active

    for col_idx, col_name in enumerate(df_prod.columns, start=1):
        ws.cell(row=1, column=col_idx, value=col_name)

    geel = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")
    oranje = PatternFill(start_color="FFA500", end_color="FFA500", fill_type="solid")

    for row_idx, row in df_prod.iterrows():
        excel_row = row_idx + 2
        for col_idx, col_name in enumerate(df_prod.columns, start=1):
            cell = ws.cell(row=excel_row, column=col_idx, value=row[col_name])

            # Kleur Ketel gerelateerd geel bij zekerheid > 0.6
            if col_name == "Ketel gerelateerd":
                if row.get("Ketel zekerheid", 0) > 0.6:
                    cell.fill = geel

            # Kleur FTF geel bij FTF=1 (ongeacht zekerheid)
            if col_name == "FTF":
                if str(row["FTF"]) == "1":
                    cell.fill = geel
                elif 0.4 <= row.get("FTF zekerheid", 0) <= 0.6:
                    cell.fill = oranje

    stream = BytesIO()
    wb.save(stream)
    stream.seek(0)

    return StreamingResponse(
        stream,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename=voorspeld_{file.filename}"}
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
