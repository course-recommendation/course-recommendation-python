from dataclasses import dataclass
from typing import List, Tuple
from collections import Counter, defaultdict
from itertools import chain
from math import exp

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob

from algorithms.feature_sentiments.types import FSItemReview

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

@dataclass
class FeatureOpinionSentiment:
    feature: str
    opinion: str
    sentiment: int

@dataclass
class SentiresResult:
    user_id: str
    item_id: str
    aspects: List[FeatureOpinionSentiment]


def parse_line(line, n_grams=[1]):
    tokens = word_tokenize(line)
    results = []
    for i in n_grams:
        if i == 1:
            results += tokens
        elif len(tokens) >= i:
            results += [' '.join(tokens[j:j+i]) for j in range(len(tokens) - i + 1)]
    return results


def get_sentiment(polarity):
    if polarity > 0:
        return 1
    elif polarity < 0:
        return -1
    return 0


def compute_feature_quality_score(sentiment, N=5):
    return 1. + (N - 1) / (1 + exp(-sentiment))


def most_frequent(elements, counter, exclude=None):
    if exclude is None:
        exclude = []

    element_freq = [counter[e] for e in elements if e not in exclude]
    if not element_freq:
        return None

    valid = [e for e in elements if e not in exclude]
    return valid[element_freq.index(max(element_freq))]


def read_concepts(path):
    concepts = set()
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                concepts.add(parts[0])
                concepts.add(parts[1])
    return list(concepts)


def read_opinion_words(path):
    words = []
    with open(path, "r") as f:
        for line in f:
            w = line.strip()
            if w and not w.startswith(";"):  # skip comments
                words.append(w)
    return words


def sentires_extract_sentiments(
        reviews: list[FSItemReview],
        concept_file: str = BASE_DIR / "resources/data-concept-instance-relations.txt",
        positive_opinion_file: str = BASE_DIR / "resources/positive-words.txt",
        negative_opinion_file: str = BASE_DIR / "resources/negative-words.txt",
        num_top_freq_aspect=2000,
        num_top_corr_aspect=500
) -> list[SentiresResult]:

    # ---- load concepts ----
    concepts = read_concepts(concept_file)

    stop_words = set(stopwords.words('english'))
    concepts = [c for c in concepts if c not in stop_words]

    # ---- load opinions ----
    positive_words = read_opinion_words(positive_opinion_file)
    negative_words = read_opinion_words(negative_opinion_file)
    opinions = set(positive_words + negative_words)

    # ---- build corpus ----
    all_text = " ".join([r.review_text.lower() for r in reviews])
    tokens = parse_line(all_text, [1])
    token_counter = Counter(tokens)

    concept_freq = [(c, token_counter[c]) for c in concepts]
    concept_freq.sort(key=lambda x: x[1], reverse=True)
    top_freq_aspects = [c for c, _ in concept_freq[:num_top_freq_aspect]]

    # ---- sentence sentiment ----
    sentence_rows = []

    for r in reviews:
        blob = TextBlob(r.review_text.lower())

        for sentence in blob.sentences: # type: ignore
            s = sentence.tokens
            if s and s[-1] == '.':
                s.remove('.')

            if s:
                sentence_rows.append({
                    "user_id": r.user_id,
                    "item_id": r.item_id,
                    "sentence": " ".join(s),
                    "sentiment": compute_feature_quality_score(get_sentiment(sentence.sentiment.polarity))
                })

    # ---- aspect sentiment extraction ----
    aspect_rows = []

    for row in sentence_rows:
        for tok in parse_line(row["sentence"], [1]):
            if tok in top_freq_aspects:
                aspect_rows.append(
                    (row["user_id"], row["item_id"], tok, row["sentiment"])
                )

    aspect_sum = defaultdict(int)

    for u, i, a, s in aspect_rows:
        aspect_sum[(u, i, a)] += s

    aspect_scores = []

    for (u, i, a), s in aspect_sum.items():
        score = compute_feature_quality_score(s)
        aspect_scores.append((u, i, a, score))

    # ---- variance ranking ----
    aspect_map = defaultdict(list)

    for _, _, a, score in aspect_scores:
        aspect_map[a].append(score)

    aspect_variance = []

    for a, scores in aspect_map.items():
        if len(scores) > 1:
            mean = sum(scores) / len(scores)
            var = sum((x - mean) ** 2 for x in scores) / len(scores)
        else:
            var = 0
        aspect_variance.append((a, var))

    aspect_variance.sort(key=lambda x: x[1], reverse=True)
    top_corr_aspects = {a for a, _ in aspect_variance[:num_top_corr_aspect]}

    # ---- aspect-opinion extraction ----
    ao_rows = []

    for row in sentence_rows:

        sentence_aspects = set()
        sentence_opinions = set()

        for tok in parse_line(row["sentence"], [1]):
            if tok in opinions:
                sentence_opinions.add(tok)
            elif tok in top_corr_aspects:
                sentence_aspects.add(tok)

        if sentence_aspects and sentence_opinions:
            ao_rows.append({
                "user_id": row["user_id"],
                "item_id": row["item_id"],
                "aspect": list(sentence_aspects),
                "opinion": list(sentence_opinions),
                "sentiment": row["sentiment"]
            })

    aspect_counter = Counter(chain.from_iterable(r["aspect"] for r in ao_rows))
    opinion_counter = Counter(chain.from_iterable(r["opinion"] for r in ao_rows))

    result_map = defaultdict(list)

    for row in ao_rows:

        aspect = most_frequent(row["aspect"], aspect_counter)
        if not aspect:
            continue

        opinion = most_frequent(
            row["opinion"],
            opinion_counter,
            aspect.split() + [aspect]
        )

        if not opinion:
            continue

        result_map[(row["user_id"], row["item_id"])].append(
            FeatureOpinionSentiment(
                feature=aspect,
                opinion=opinion,
                sentiment=row["sentiment"]
            )
        )

    result = []

    for (u, i), triples in result_map.items():
        result.append(
            SentiresResult(
                user_id=u,
                item_id=i,
                aspects=triples
            )
        )

    return result

if __name__ == "__main__":
    # Example usage
    reviews = [
        FSItemReview(item_id="course1", user_id="user1", review_text="The content was great but the instructor was boring."),
        FSItemReview(item_id="course1", user_id="user2", review_text="I loved the instructor, but the content was too basic."),
        FSItemReview(item_id="course2", user_id="user3", review_text="The course was fantastic!"),
    ]

    sentires_sentiments = sentires_extract_sentiments(reviews)
    print(sentires_sentiments)