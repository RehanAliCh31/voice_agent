"""Classical baseline: TF-IDF word and character n-grams into a linear SVM.

Its job is to populate the comparison row that shows whether the transformer
earns its cost. A weak baseline would make the comparison meaningless, so the
character n-grams are included deliberately: they handle morphology and typos
that word features miss.
"""

from __future__ import annotations

from collections.abc import Sequence

from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import-untyped]
from sklearn.metrics import accuracy_score, f1_score  # type: ignore[import-untyped]
from sklearn.pipeline import FeatureUnion, Pipeline  # type: ignore[import-untyped]
from sklearn.svm import LinearSVC  # type: ignore[import-untyped]

from voicebot.nlu.clinc import TextExample
from voicebot.utils.logging import get_logger

log = get_logger("nlu.baseline")


def build_baseline(random_state: int = 42) -> Pipeline:
    """TF-IDF (word 1-2 grams + char 3-5 grams) into LinearSVC."""
    return Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "word",
                            TfidfVectorizer(
                                analyzer="word", ngram_range=(1, 2), sublinear_tf=True, min_df=1
                            ),
                        ),
                        (
                            "char",
                            TfidfVectorizer(
                                analyzer="char_wb",
                                ngram_range=(3, 5),
                                sublinear_tf=True,
                                min_df=1,
                            ),
                        ),
                    ]
                ),
            ),
            ("clf", LinearSVC(C=1.0, random_state=random_state)),
        ]
    )


def evaluate_baseline(pipeline: Pipeline, examples: Sequence[TextExample]) -> dict[str, float]:
    """Accuracy and macro-F1 on the given examples."""
    texts = [ex.text for ex in examples]
    gold = [ex.intent for ex in examples]
    predicted = pipeline.predict(texts)
    return {
        "accuracy": float(accuracy_score(gold, predicted)),
        "macro_f1": float(f1_score(gold, predicted, average="macro", zero_division=0)),
    }


def train_baseline(
    train: Sequence[TextExample],
    evaluation: Sequence[TextExample],
    random_state: int = 42,
) -> tuple[Pipeline, dict[str, float]]:
    """Fit the baseline and score it."""
    pipeline = build_baseline(random_state=random_state)
    pipeline.fit([ex.text for ex in train], [ex.intent for ex in train])
    metrics = evaluate_baseline(pipeline, evaluation)
    log.info("baseline accuracy=%.4f macro_f1=%.4f", metrics["accuracy"], metrics["macro_f1"])
    return pipeline, metrics
