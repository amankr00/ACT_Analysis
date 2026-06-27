from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Cross-encoder that scores (query, passage) pairs directly for relevance.
_RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_tokenizer = None
_model = None


def get_reranker():
    global _tokenizer, _model

    if _model is None:
        print(f"🔵 Loading cross-encoder reranker ({_RERANK_MODEL_NAME})...")
        _tokenizer = AutoTokenizer.from_pretrained(_RERANK_MODEL_NAME)
        _model = AutoModelForSequenceClassification.from_pretrained(_RERANK_MODEL_NAME)
        _model.eval()

    return _tokenizer, _model


def rerank(query, docs, top_k=None, batch_size=16, score_key="rerank_score"):
    """Re-orders ``docs`` by cross-encoder relevance to ``query``.

    Each doc must be a dict with a ``text`` field. A normalized relevance
    probability (sigmoid of the raw logit, in [0, 1]) is written to each doc
    under ``score_key``. Returns a new list sorted by that score, descending.

    If ``top_k`` is given, only the best ``top_k`` docs are returned.
    """
    if not docs:
        return docs

    tokenizer, model = get_reranker()

    scores = []
    for start in range(0, len(docs), batch_size):
        batch = docs[start:start + batch_size]
        queries = [query] * len(batch)
        passages = [d.get("text", "") for d in batch]

        encoded = tokenizer(
            queries,
            passages,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        )

        with torch.no_grad():
            logits = model(**encoded).logits.squeeze(-1)
            probs = torch.sigmoid(logits)

        scores.extend(probs.tolist())

    for doc, score in zip(docs, scores):
        doc[score_key] = round(float(score), 4)

    ranked = sorted(docs, key=lambda d: d.get(score_key, 0.0), reverse=True)
    if top_k is not None:
        ranked = ranked[:top_k]
    return ranked
