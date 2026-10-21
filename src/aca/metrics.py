"""TriviaQA exact match: lowercase, ASCII punctuation/articles, whitespace."""
import re
import string
from collections import Counter


def normalize_answer(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', text).split())


def exact_match(prediction, aliases):
    prediction = normalize_answer(prediction)
    return int(any(prediction == normalize_answer(a) for a in aliases))


def token_f1(prediction, aliases):
    p = normalize_answer(prediction).split()
    def score(answer):
        a = normalize_answer(answer).split()
        if not p or not a:
            return float(p == a)
        overlap = sum((Counter(p) & Counter(a)).values())
        return 2 * overlap / (len(p) + len(a))
    return max(map(score, aliases), default=0.0)
