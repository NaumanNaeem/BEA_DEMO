from typing import List
import torch
import torch.nn.functional as F
from embeddings import mpnet
from model_loader import MODEL, DEVICE

LABEL_INT2STR = {0: "No", 1: "Yes", 2: "To some extent"}


def _infer_batch(histories: List[str], responses: List[str]) -> list[dict]:
    h_emb = mpnet.compute_cls(histories).to(DEVICE)
    r_emb = mpnet.compute_cls(responses).to(DEVICE)

    with torch.inference_mode():
        logits = MODEL(h_emb, r_emb)
        probs  = F.softmax(logits, dim=-1).cpu()

    out = []
    for p in probs:
        pred = int(p.argmax())
        out.append({
            "prediction": LABEL_INT2STR[pred],
            "confidence": round(float(p.max()), 4),
            "probs": {LABEL_INT2STR[i]: round(float(p[i]), 4) for i in range(3)}
        })
    return out


def predict_single(history: str, response: str) -> dict:
    return _infer_batch([history], [response])[0]


def predict_batch(histories: List[str], responses: List[str]) -> list[dict]:
    return _infer_batch(histories, responses)
