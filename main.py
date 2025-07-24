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

# Modellen en vectorizers laden (zorg dat ze in je project staan)
ketel_model = joblib.load("ketel_model_Vfinal.joblib")
ketel_vec = joblib.load("ketel_vectorizer_Vfinal.joblib")
ftf_model = joblib.load("ftf_model_Vfinal.joblib")
ftf_vec = joblib.load("ftf_vectorizer_Vfinal.joblib")

@app.post("/process_excel/")
async def process_excel(file: UploadFile = File(...)):
    # Lees geüpload bestand in memory
    contents = await file.read()
    df_prod = pd.read_excel(BytesIO(contents), sheet_name="Leeg")

    # Dynamisch bepalen welke kolommen er zijn
    alle_kolommen = df_prod.columns.tolist()
    
    # Zoek naar oplossingskolommen die daadwerkelijk bestaan
    potentiele_oplossingskolommen = ["Oplossingen"] + [col for col in alle_kolommen if col.startswith("Unnamed:")]
    oplossingskolommen = [col for col in potentiele_oplossingskolommen if col in alle_kolommen]
    
    # Basis tekstkolommen die meestal bestaan
    basis_tekstkolommen = [
        "Werkbeschrijving", "Werkbon is vervolg van",
        "Werkbon nummer", "Uitvoerdatum", "Object referentie", "Installatie apparaat omschrijving"
    ]
    tekstkolommen = [col for col in basis_tekstkolommen if col in alle_kolommen]

    # Vul oplossingen kolommen leeg met lege strings
    if oplossingskolommen:
        df_prod[oplossingskolommen] = df_prod[oplossingskolommen].fillna("")
        df_prod["Oplossingen_samengevoegd"] = df_prod[oplossingskolommen].astype(str).agg(" ".join, axis=1)
    else:
        df_prod["Oplossingen_samengevoegd"] = ""
    
    # Voeg Oplossingen_samengevoegd toe aan tekstkolommen
    tekstkolommen.append("Oplossingen_samengevoegd")

    df_prod[tekstkolommen] = df_prod[tekstkolommen].fillna("")
    df_prod["combined_text"] = df_prod[tekstkolommen].apply(lambda r: " ".join([str(x) for x in r]), axis=1)
    df_prod["heeft_vervolg"] = df_prod["Werkbon is vervolg van"].apply(lambda x: int(bool(str(x).strip())))
    df_prod["tekstlengte"] = df_prod["combined_text"].apply(len)
    df_prod["woordenaantal"] = df_prod["combined_text"].apply(lambda x: len(x.split()))
    df_prod["contains_onderdeel"] = df_prod["combined_text"].str.contains("onderdeel|vervangen|vervang", case=False).astype(int)
    df_prod["contains_reset"] = df_prod["combined_text"].str.contains("reset|herstart", case=False).astype(int)
    df_prod["contains_advies"] = df_prod["combined_text"].str.contains("advies|aanbeveling", case=False).astype(int)

    # Ketel voorspellen
    X_text_k = df_prod["combined_text"]
    X_vec_k = ketel_vec.transform(X_text_k).toarray()
    extra_k = df_prod[["heeft_vervolg", "tekstlengte", "woordenaantal", "contains_onderdeel", "contains_reset", "contains_advies"]].values
    X_comb_k = np.hstack((X_vec_k, extra_k))

    y_proba_k = ketel_model.predict_proba(X_comb_k)
    y_pred_k = ketel_model.predict(X_comb_k)

    # Vul bestaande kolom 'Ketel gerelateerd'
    df_prod["Ketel gerelateerd"] = np.where(y_proba_k.max(axis=1) > 0.6, np.where(y_pred_k == 1, "j", "n"), "")

    # Voeg zekerheid toe
    df_prod["Ketel zekerheid"] = y_proba_k.max(axis=1)

    # FTF voorspellen waar ketel 'j' is (zonder extra zekerheid filter)
    mask_ftf = (df_prod["Ketel gerelateerd"] == "j")
    df_ftf = df_prod[mask_ftf].copy()

    X_text_f = df_ftf["combined_text"]
    X_vec_f = ftf_vec.transform(X_text_f).toarray()
    extra_f = df_ftf[["heeft_vervolg", "tekstlengte", "woordenaantal", "contains_onderdeel", "contains_reset", "contains_advies"]].values
    X_comb_f = np.hstack((X_vec_f, extra_f))

    y_proba_f = ftf_model.predict_proba(X_comb_f)
    y_pred_f = ftf_model.predict(X_comb_f)

    # Eerst 'j'/'n' als output, met predictiedrempel 0.6
    df_ftf["FTF"] = np.where(y_proba_f.max(axis=1) > 0.6, np.where(y_pred_f == 1, "j", "n"), "")
    df_ftf["FTF zekerheid"] = y_proba_f.max(axis=1)

    df_prod["FTF"] = df_prod["FTF"].astype(object)
    df_prod.loc[df_ftf.index, "FTF"] = df_ftf["FTF"]
    df_prod.loc[df_ftf.index, "FTF zekerheid"] = df_ftf["FTF zekerheid"]

    # Zet 'j' om naar '1', 'n' naar lege string voor output
    df_prod["FTF"] = df_prod["FTF"].replace({"j": "1", "n": ""})

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

            # Kleur Ketel gerelateerd kolom
            if col_name == "Ketel gerelateerd":
                if row["Ketel zekerheid"] > 0.6:
                    cell.fill = geel
                elif 0.4 <= row["Ketel zekerheid"] <= 0.6:
                    cell.fill = oranje
            # Kleur FTF kolom
            if col_name == "FTF":
                if row["FTF zekerheid"] > 0.6:
                    cell.fill = geel
                elif 0.4 <= row["FTF zekerheid"] <= 0.6:
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
