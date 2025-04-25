from fastapi import FastAPI, UploadFile, File
import pandas as pd
from predictor import predict_single, predict_batch
from pydantic import BaseModel
from sklearn.metrics import classification_report

app = FastAPI()


class PairInput(BaseModel):
    history: str
    response: str


@app.post("/predict/single")
def single(pair: PairInput):
    return predict_single(pair.history, pair.response)


@app.post("/predict/batch")
def batch(file: UploadFile = File(...)):
    """
    CSV must have: conversation_history, response
    Optional: label (ground‑truth). If present, a classification_report is returned.
    """
    df = pd.read_csv(file.file)

    if not {"conversation_history", "response"}.issubset(df.columns):
        return {"error": "CSV must contain columns: conversation_history, response"}

    preds = predict_batch(df["conversation_history"].tolist(),
                          df["response"].tolist())

    df_preds = pd.DataFrame(preds)
    df_out   = pd.concat([df, df_preds], axis=1)

    if "label" in df.columns:
        y_true = df["label"]
        y_pred = df_out["prediction"].map({"No":0, "Yes":1, "To some extent":2})
        report = classification_report(
            y_true, y_pred, target_names=["No", "Yes", "To some extent"], output_dict=True
        )
        return {"predictions": df_out.to_dict(orient="records"),
                "classification_report": report}

    return {"predictions": df_out.to_dict(orient="records")}
