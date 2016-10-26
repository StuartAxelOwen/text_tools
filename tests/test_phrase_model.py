
from text_tools import train_phrase_model, apply_phrase_models
from gensim.models import Phrases


def test_phrase_model():
    texts = [
        "This is how we do it. You know it's friday night, and I feel all right, and party's here on the west side.",
        "Homies said yo I don't",
        "Cuz it's Friday, Friday, gotta get down on friday.  Everyone's looking forward to the weekend, weekend."
    ]

    model = train_phrase_model(texts)
    assert model is not None
    assert model.vocab[b'this'] > 0


def test_multi_phrase_model():
    texts = [
        "This is how we do it. You know it's friday night, and I feel all right, and party's here on the west side.",
        "Homies said yo I don't",
        "Cuz it's Friday, Friday, gotta get down on friday.  Everyone's looking forward to the weekend, weekend."
    ]

    models = train_phrase_model(texts, 3)

    assert len(models) == 3
    for model in models:
        assert isinstance(model, Phrases)
        assert model.vocab[b'this'] > 0

    phrased = list(apply_phrase_models(models, texts))
    # Make sure applying phrase models works, that it returns a stream of token lists
    assert isinstance(phrased[0], list)

