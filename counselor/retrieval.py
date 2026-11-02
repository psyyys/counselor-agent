"""BM25 检索。中文按相邻两字切分，英文按单词切分。

不用向量模型：不花 API 费用、不用装大模型，中文短句匹配效果也够用。
手册是英文、对话是中文，所以手册检索用每次会谈开头生成的英文关键词（见 session.py）。
"""
import math
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path

_EN = re.compile(r"[a-z][a-z\-']+")
_ZH = re.compile(r"[一-鿿]+")
_EN_STOP = set("the a an and or of to in on for is are was were be been it this that with as at by from "
               "you your we our they their he she his her i me my not but if so do does did can will".split())


def tokenize(text: str) -> list[str]:
    text = text.lower()
    tokens = [w for w in _EN.findall(text) if w not in _EN_STOP]
    for run in _ZH.findall(text):
        if len(run) == 1:
            tokens.append(run)
        tokens.extend(run[i:i + 2] for i in range(len(run) - 1))
    return tokens


class BM25Index:
    def __init__(self, docs: list[dict], texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1, self.b = k1, b
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.lengths = []
        for i, text in enumerate(texts):
            tf = Counter(tokenize(text))
            self.lengths.append(sum(tf.values()))
            for tok, n in tf.items():
                self.postings[tok].append((i, n))
        self.avg_len = sum(self.lengths) / max(len(self.lengths), 1)
        n_docs = len(docs)
        self.idf = {t: math.log(1 + (n_docs - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()}
        self.postings = dict(self.postings)

    def search(self, query: str, k: int = 5, where=None) -> list[tuple[float, dict]]:
        scores: dict[int, float] = defaultdict(float)
        for tok in set(tokenize(query)):
            idf = self.idf.get(tok)
            if idf is None:
                continue
            for i, tf in self.postings[tok]:
                norm = tf + self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avg_len)
                scores[i] += idf * tf * (self.k1 + 1) / norm
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        out = []
        for i, s in ranked:
            if where is None or where(self.docs[i]):
                out.append((s, self.docs[i]))
                if len(out) >= k:
                    break
        return out

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "BM25Index":
        with open(path, "rb") as f:
            return pickle.load(f)
