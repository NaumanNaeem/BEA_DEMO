"""
Sentence‑Transformer encoder (all‑mpnet‑base‑v2) → [CLS] vector.
Cache stays in ~/.cache/huggingface/…
"""
import torch
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
_encoder   = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()


@torch.inference_mode()
def compute_cls(texts: list[str]) -> torch.Tensor:
    """
    texts → Tensor shape (B, 768) on *CPU*
    """
    batch = _tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(DEVICE)

    out = _encoder(**batch).last_hidden_state[:, 0, :]   # CLS
    return out.cpu()
