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

# Keywordlijsten Ketel gerelateerd (lowercase keys voor case-insensitive matching)
keywords_kg = {
    "flowsensor vervangen": "j",
    "aansturing ivm 9u": "j",
    "aansturing nefit": "j",
    "aansturing ivm 9p": "j",
    "lek hydroblok": "j",
    "hydroblok": "j",
    "3wk": "j",
    "3WK": "j",
    "branderautomaat": "j",
    "defecte kim": "j",
    "platenwisselaar vervangen": "j",
    "wisselaar vervangen": "j",
    "het hydroblok was lek": "j",
    "rga beugelen": "n",
    "rga gebeugeld": "n",
    "druksensor vervangen": "j",
    "condensafvoer": "n",
    "hoofdprint vervangen": "j",
    "driewegklep": "j",
    "drie weg klep": "j",
    "3wegklep": "j",
    "3weg klep": "j",
    "wisselaar": "j",
    "stromingssensor": "j",
    "hoofdprint": "j",
    "drieweg klep": "j",
    "ontstek pen vv": "j",
    "thermostaat van de klant is niet goed": "n",
    "wartel bij de pomp": "j",
    "ontsteekpen vervangen": "j",
    "pomp vervangen": "j",
    "ketel vervangen": "j",
    "pomp": "j",
    "condensafvoer herstellen.": "n",
    "nieuwe afvoer maken": "n",
    "flowsensor vv": "j",
    "sensorflow vervangen": "j",
    "sensorflow vv": "j",
    "expansievat": "n",
    "vloerverwarming": "n",
    "slang op verdeler": "n",
    "expansievat vervangen": "n",
    "rga aanpassen": "n",
    "warmtewisselaar": "j",
    "ltv(luchttoevoer) vervangen": "n",
    "verroest van binnen": "j",
    "batterijen vervangen": "n",
    "wisselaar is lek": "j",
    "ketel bij gevuld": "n",
    "pomp zat vast": "j",
    "geen gas toevoer": "n",
    "batterij van de koolmonoxide melder": "n",
    "vaillant pomp aansluitkabel": "j",
    "vaillant druksensor": "j",
    "thermostaat": "n",
    "vulkraan vervangen": "n",
    "stekkertje ontsteekkabel": "n",
    "rga herstellen": "n",
    "rga vervangen en gebeugeld": "n",
    "ltv (luchttoevoer) gebeugeld": "n",
    "warmtewisselaar vervangen": "j",
    "sensor flow vervangen": "j",
    "pomp vv": "j",
    "pomp verv": "j",
    "sensor flow": "j",
    "ketel vv": "j",
    "flow sensor": "j",
    "flowsensor": "j",
    "ketel verv": "j",
    "ventilator": "j",
    "fentilator": "j",
    "pakking": "j",
    "gloeiplug": "j",
    "3 weg klep": "j",
    "nieuwe ketel": "j",
    "sam trechter vervangen": "n",
    "rga en ltv + dakdoorvoer aanpassen.": "n",
    "rookgas vervangen": "n",
    "inspuiter": "j",
    "bijgevuld": "n",
    "flowswitch": "j",
    "druksensor defect": "j",
    "alle radiatorkranen dicht": "n",
    "gasblok vervangen": "j",
    "wasmachinekraan vervangen": "n",
    "lijkt op print te zijn.": "j",
    "overstort druppelde": "n",
    "overstort en wasmachine kraan": "n",
    "overstort vervangen": "n",
    "uitgevoerde werkzaamheden: ketel afgekeurd": "j",
    "spoed - ketel afgekeurd - nog vervangen": "j",
    "ketel afgekeurd - nog vervangen": "j",
    "thermostaat vervangen": "n",
    "afuizing badkamer": "n",
    "kamerthermostaat vervangen": "n"
}

# Rule-based FTF keywords
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

def apply_keywords_per_column(df, keywords_dict, columns):
    df["Ketel gerelateerd_keyword"] = np.nan
    for col in columns:
        text_lower = df[col].fillna("").str.lower()
        for kw, val in keywords_dict.items():
            # Zet regex=False om waarschuwing te vermijden
            mask = text_lower.str.contains(kw, na=False, regex=False)
            if val == "j":
                # Zet 'j' altijd
                df.loc[mask, "Ketel gerelateerd_keyword"] = "j"
            else:
                # Alleen zet 'n' als nog geen 'j' aanwezig is
                df.loc[mask & df["Ketel gerelateerd_keyword"].isna(), "Ketel gerelateerd_keyword"] = "n"
    return df

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

    # ---- Keyword matching Ketel gerelateerd, per kolom met prioriteit ----
    keyword_check_columns = ["Werkbeschrijving", "Oplossingen_samengevoegd"]
    df_prod = apply_keywords_per_column(df_prod, keywords_kg, keyword_check_columns)

    # Vul kolom Ketel gerelateerd vanuit keywords (prioriteit)
    mask_keyword = df_prod["Ketel gerelateerd_keyword"].notnull()
    df_prod.loc[mask_keyword, "Ketel gerelateerd"] = df_prod.loc[mask_keyword, "Ketel gerelateerd_keyword"]

    # Model voorspelling Ketel gerelateerd alleen toepassen op rijen zonder keyword invulling
    mask_ml = df_prod["Ketel gerelateerd"].isnull() | (df_prod["Ketel gerelateerd"] == "")
    if mask_ml.any():
        X_text_k = df_prod.loc[mask_ml, "combined_text"]
        X_vec_k = ketel_vec.transform(X_text_k).toarray()
        extra_k = df_prod.loc[mask_ml, ["heeft_vervolg", "tekstlengte", "woordenaantal", "contains_onderdeel", "contains_reset", "contains_advies"]].values
        X_comb_k = np.hstack((X_vec_k, extra_k))

        y_proba_k = ketel_model.predict_proba(X_comb_k)
        y_pred_k = ketel_model.predict(X_comb_k)

        df_prod.loc[mask_ml, "Ketel gerelateerd"] = np.where(y_proba_k.max(axis=1) > 0.7, np.where(y_pred_k == 1, "j", "n"), "")

    df_prod["Ketel zekerheid"] = 0
    if mask_ml.any():
        df_prod.loc[mask_ml, "Ketel zekerheid"] = y_proba_k.max(axis=1)
    df_prod.loc[mask_keyword, "Ketel zekerheid"] = 1  # Keywords zijn zeker

    # FTF voorspellen voor rijen waar Ketel gerelateerd = "j"
    mask_ftf = (df_prod["Ketel gerelateerd"] == "j")
    df_ftf = df_prod[mask_ftf].copy()

    # Rule-based FTF toewijzen
    df_ftf["FTF_keyword"] = df_ftf["combined_text"].apply(keyword_based_ftf)

    # ML FTF voorspelling voor rijen zonder rule-based FTF
    mask_ml_ftf = df_ftf["FTF_keyword"].isnull()
    if mask_ml_ftf.any():
        X_text_f = df_ftf.loc[mask_ml_ftf, "combined_text"]
        X_vec_f = ftf_vec.transform(X_text_f).toarray()
        extra_f = df_ftf.loc[mask_ml_ftf, ["heeft_vervolg", "tekstlengte", "woordenaantal", "contains_onderdeel", "contains_reset", "contains_advies"]].values
        X_comb_f = np.hstack((X_vec_f, extra_f))

        y_proba_f = ftf_model.predict_proba(X_comb_f)
        y_pred_f = ftf_model.predict(X_comb_f)

        ftf_pred = np.where(y_proba_f.max(axis=1) > 0.6, y_pred_f.astype(str), "0")
        df_ftf.loc[mask_ml_ftf, "FTF_keyword"] = ftf_pred

    # Vul FTF kolom met gecombineerde resultaten, vervang "0" met lege string
    df_prod.loc[df_ftf.index, "FTF"] = df_ftf["FTF_keyword"].replace({"1": "1", "0": "", "": ""})

    # FTF zekerheid alleen voor ML voorspellingen
    df_ftf["FTF zekerheid"] = 0
    if mask_ml_ftf.any():
        df_ftf.loc[mask_ml_ftf, "FTF zekerheid"] = y_proba_f.max(axis=1)
    df_prod.loc[df_ftf.index, "FTF zekerheid"] = df_ftf["FTF zekerheid"]

    # Kolommen die NIET in output mogen
    exclude_cols = [
        "Unnamed: 18", "Unnamed: 19", "Unnamed: 20", "Unnamed: 21", "Unnamed: 22", "Unnamed: 23",
        "Oplossingen_samengevoegd", "combined_text", "heeft_vervolg", "tekstlengte", "woordenaantal",
        "contains_onderdeel", "contains_reset", "contains_advies", "Ketel zekerheid", "FTF zekerheid",
        "Ketel gerelateerd_keyword"
    ]

    output_cols = [col for col in df_prod.columns if col not in exclude_cols]

    # Kolommen die wél in output blijven maar zonder kolomnaam in Excel
    cols_with_empty_name = [
        "Werkbon is vervolg van", "Werkbon nummer", "Uitvoerdatum", "Object referentie",
        "Installatie apparaat omschrijving"
    ]

    output_col_names = []
    for col in output_cols:
        if col in cols_with_empty_name:
            output_col_names.append("")
        else:
            output_col_names.append(col)

    # Output Excel met kleurcodering
    wb = openpyxl.Workbook()
    ws = wb.active

    # Schrijf header (kolomnamen)
    for col_idx, col_name in enumerate(output_col_names, start=1):
        ws.cell(row=1, column=col_idx, value=col_name)

    geel = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")
    oranje = PatternFill(start_color="FFA500", end_color="FFA500", fill_type="solid")

    for row_idx, row in df_prod.iterrows():
        excel_row = row_idx + 2
        for col_idx, col_name in enumerate(output_cols, start=1):
            cell = ws.cell(row=excel_row, column=col_idx, value=row[col_name])

            # Kleur Ketel gerelateerd geel bij zekerheid >= 0.6, oranje bij 0 < zekerheid < 0.6
            if col_name == "Ketel gerelateerd":
                if row["Ketel zekerheid"] >= 0.7:
                    cell.fill = geel
                elif 0 < row["Ketel zekerheid"] < 0.7:
                    cell.fill = oranje

            # Kleur FTF geel bij FTF=1 en zekerheid >= 0.6, oranje bij zekerheid tussen 0 en 0.6
            if col_name == "FTF":
                ftf_certainty = row.get("FTF zekerheid", 0)
                if str(row["FTF"]) == "1" and ftf_certainty >= 0.6:
                    cell.fill = geel
                elif 0 < ftf_certainty < 0.6:
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
