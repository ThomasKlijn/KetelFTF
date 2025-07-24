from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, FileResponse
import pandas as pd
import numpy as np
import joblib
from io import BytesIO
import openpyxl
from openpyxl.styles import PatternFill
import uvicorn
import re

app = FastAPI()

@app.get("/")
async def read_root():
    return FileResponse("index.html")

# Modellen en vectorizers laden (zorg dat ze in je project staan)
ketel_model = joblib.load("ketel_model_Vfinal.joblib")
ketel_vec = joblib.load("ketel_vectorizer_Vfinal.joblib")
ftf_model = joblib.load("ftf_model_Vfinal.joblib")
ftf_vec = joblib.load("ftf_vectorizer_Vfinal.joblib")

# Keywords lijst met labels (j of n), case-insensitive, incl. losse woorden en woordfouten gefaciliteerd door regex
keywords_list = [
    ("flowsensor vervangen", "j"),
    ("aansturing ivm 9u", "j"),
    ("aansturing nefit", "j"),
    ("aansturing ivm 9p", "j"),
    ("lek hydroblok", "j"),
    ("defecte kim", "j"),
    ("platenwisselaar vervangen", "j"),
    ("het hydroblok was lek", "j"),
    ("rga beugelen", "n"),
    ("rga gebeugeld", "n"),
    ("druksensor vervangen", "j"),
    ("condensafvoer", "n"),
    ("hoofdprint vervangen", "j"),
    ("ontstek pen vv", "j"),
    ("thermostaat van de klant is niet goed", "n"),
    ("wartel bij de pomp", "j"),
    ("ontsteekpen vervangen", "j"),
    ("condensafvoer herstellen", "n"),
    ("nieuwe afvoer maken", "n"),
    ("expansievat vervangen", "n"),
    ("rga aanpassen", "n"),
    ("ltv.*luchttoevoer.*", "n"),  # regex voor ltv(Luchttoevoer)
    ("verroest van binnen", "j"),
    ("batterijen vervangen", "n"),
    ("wisselaar is lek", "j"),
    ("ketel bij gevuld", "n"),
    ("pomp zat vast", "j"),
    ("geen gas toevoer", "n"),
    ("batterij van de koolmonoxide melder", "n"),
    ("vaillant pomp aansluitkabel", "j"),
    ("vaillant druksensor", "j"),
    ("thermostaat", "n"),
    ("vulkraan vervangen", "n"),
    ("stekkertje ontsteekkabel", "n"),
    ("rga herstellen", "n"),
    ("rga vervangen en gebeugeld", "n"),
    ("ltv.*luchttoevoer.*gebeugeld", "n"),
    ("warmtewisselaar vervangen", "j"),
    ("sensor flow vervangen", "j"),
    ("sam trechter vervangen", "n"),
    ("rga en ltv \+ dakdoorvoer aanpassen", "n"),
    ("rookgas vervangen", "n"),
    ("bijgevuld", "n"),
    ("druksensor defect", "j"),
    ("alle radiatorkranen dicht", "n"),
    ("gasblok vervangen", "j"),
    ("wasmachinekraan vervangen", "n"),
    ("lijkt op print te zijn", "j"),
    ("overstort druppelde", "n"),
    ("overstort en wasmachine kraan", "n"),
    ("overstort vervangen", "n"),
    ("uitgevoerde werkzaamheden: ketel afgekeurd", "j"),
    ("spoed - ketel afgekeurd - nog vervangen", "j"),
    ("ketel afgekeurd - nog vervangen", "j"),
    ("thermostaat vervangen", "n"),
    ("afuizing badkamer", "n"),
    ("kamerthermostaat vervangen", "n")
]

# Maak een dictionary met regex patterns
keywords_dict = [(re.compile(k, re.IGNORECASE), v) for k, v in keywords_list]

def find_keyword_label(text: str, keywords) -> str | None:
    for pattern, label in keywords:
        if pattern.search(text):
            return label
    return None

@app.post("/process_excel/")
async def process_excel(file: UploadFile = File(...)):
    # Lees geüpload bestand in memory
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

    # Vul ketel gerelateerd direct op basis van keywords
    df_prod["Ketel gerelateerd keyword"] = df_prod["combined_text"].apply(lambda x: find_keyword_label(x, keywords_dict))

    # Zet de keyword voorspellingen in 'Ketel gerelateerd' en zekerheid 1.0
    df_prod.loc[df_prod["Ketel gerelateerd keyword"].notnull(), "Ketel gerelateerd"] = df_prod["Ketel gerelateerd keyword"]
    df_prod.loc[df_prod["Ketel gerelateerd keyword"].notnull(), "Ketel zekerheid"] = 1.0

    # Model voorspellingen alleen voor rijen waar ketel nog leeg is (nog niet door keyword gevuld)
    mask_model = df_prod["Ketel gerelateerd"].isnull() | (df_prod["Ketel gerelateerd"] == "")

    if mask_model.any():
        X_text_k = df_prod.loc[mask_model, "combined_text"]
        X_vec_k = ketel_vec.transform(X_text_k).toarray()
        extra_k = df_prod.loc[mask_model, ["heeft_vervolg", "tekstlengte", "woordenaantal", "contains_onderdeel", "contains_reset", "contains_advies"]].values
        X_comb_k = np.hstack((X_vec_k, extra_k))

        y_proba_k = ketel_model.predict_proba(X_comb_k)
        y_pred_k = ketel_model.predict(X_comb_k)

        df_prod.loc[mask_model, "Ketel gerelateerd"] = np.where(y_proba_k.max(axis=1) > 0.6, np.where(y_pred_k == 1, "j", "n"), "")
        df_prod.loc[mask_model, "Ketel zekerheid"] = y_proba_k.max(axis=1)

    # FTF voorspellen zoals voorheen, alleen voor ketel = 'j' en zekerheid > 0.4
    mask_ftf = (df_prod["Ketel gerelateerd"] == "j") & (df_prod["Ketel zekerheid"] > 0.4)
    df_ftf = df_prod[mask_ftf].copy()

    X_text_f = df_ftf["combined_text"]
    X_vec_f = ftf_vec.transform(X_text_f).toarray()
    extra_f = df_ftf[["heeft_vervolg", "tekstlengte", "woordenaantal", "contains_onderdeel", "contains_reset", "contains_advies"]].values
    X_comb_f = np.hstack((X_vec_f, extra_f))

    y_proba_f = ftf_model.predict_proba(X_comb_f)
    y_pred_f = ftf_model.predict(X_comb_f)

    df_ftf["FTF"] = np.where(y_proba_f.max(axis=1) > 0.7, y_pred_f.astype(str), "")
    df_ftf["FTF zekerheid"] = y_proba_f.max(axis=1)

    df_prod.loc[df_ftf.index, "FTF"] = df_ftf["FTF"]
    df_prod.loc[df_ftf.index, "FTF zekerheid"] = df_ftf["FTF zekerheid"]

    # Kleurcodering in output Excel
    wb = openpyxl.Workbook()
    ws = wb.active

    # Schrijf kolomnamen
    for col_idx, col_name in enumerate(df_prod.columns, start=1):
        ws.cell(row=1, column=col_idx, value=col_name)

    geel = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")
    oranje = PatternFill(start_color="FFA500", end_color="FFA500", fill_type="solid")

    for row_idx, row in df_prod.iterrows():
        excel_row = row_idx + 2
        for col_idx, col_name in enumerate(df_prod.columns, start=1):
            cell = ws.cell(row=excel_row, column=col_idx, value=row[col_name])

            # Kleur Ketel gerelateerd kolom, geel ook als zekerheid 1 (keyword)
            if col_name == "Ketel gerelateerd":
                if row.get("Ketel zekerheid", 0) >= 1.0:
                    cell.fill = geel
                elif row.get("Ketel zekerheid", 0) > 0.6:
                    cell.fill = geel
                elif 0.4 <= row.get("Ketel zekerheid", 0) <= 0.6:
                    cell.fill = oranje

            # Kleur FTF kolom
            if col_name == "FTF":
                if row.get("FTF zekerheid", 0) > 0.7:
                    cell.fill = geel
                elif 0.4 <= row.get("FTF zekerheid", 0) <= 0.7:
                    cell.fill = oranje

    # Output naar bytes buffer
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
