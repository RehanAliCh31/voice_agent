from voicebot.nlu.baseline_svm import build_baseline, evaluate_baseline, train_baseline
from voicebot.nlu.clinc import TextExample

TRAIN = [
    TextExample("what is my balance", "balance"),
    TextExample("show my balance please", "balance"),
    TextExample("how much money do i have", "balance"),
    TextExample("check the balance on my account", "balance"),
    TextExample("transfer fifty dollars to sarah", "transfer"),
    TextExample("send money to john", "transfer"),
    TextExample("move funds to my savings", "transfer"),
    TextExample("transfer cash to ahmed", "transfer"),
]

TEST = [
    TextExample("tell me my balance", "balance"),
    TextExample("send fifty dollars to sarah", "transfer"),
]


def test_pipeline_has_vectorizer_and_classifier():
    pipeline = build_baseline()
    assert "features" in pipeline.named_steps
    assert "clf" in pipeline.named_steps


def test_baseline_learns_a_separable_two_class_problem():
    pipeline, metrics = train_baseline(TRAIN, TEST)
    assert metrics["accuracy"] == 1.0
    assert pipeline.predict(["what is my balance"])[0] == "balance"


def test_evaluate_returns_accuracy_and_macro_f1():
    pipeline, _ = train_baseline(TRAIN, TEST)
    metrics = evaluate_baseline(pipeline, TEST)
    assert set(metrics) == {"accuracy", "macro_f1"}
    assert 0.0 <= metrics["macro_f1"] <= 1.0


def test_char_features_handle_an_unseen_word_form():
    pipeline, _ = train_baseline(TRAIN, TEST)
    # "balanced" never appears in training; char n-grams should still carry it.
    assert pipeline.predict(["what is my balanced amount"])[0] == "balance"
