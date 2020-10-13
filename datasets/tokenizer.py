__all__ = ["tokenize", "spacy_tokenize", "corenlp_tokenize"]

from typing import Sequence

import spacy
from stanza.nlp.corenlp import CoreNLPClient


def tokenize(sent: str, method="space") -> Sequence[str]:
    if method == "space":
        return sent.split()
    elif method == "spacy":
        return list(spacy_tokenize(sent))
    elif method == "corenlp":
        return list(corenlp_tokenize(sent))
    else:
        raise ValueError(f"unknown tokenization method: {method}")


def spacy_tokenize(s: str, model_name="en_core_web_sm") -> Sequence[str]:
    global nlp
    if "nlp" not in globals():
        nlp = spacy.load(model_name)
    return [t.text for t in nlp.tokenizer(s)]


def corenlp_tokenize(s: str) -> Sequence[str]:
    global corenlp_client
    if "corenlp_client" not in globals():
        corenlp_client = CoreNLPClient(
            default_annotators=["ssplit", "tokenize"])
    return [t.word for s in corenlp_client.annotate(s).sentences for t in s]
