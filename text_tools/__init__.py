
from nltk.tokenize import RegexpTokenizer
import toolz
import re


@toolz.memoize()
def get_tokenizer(pattern='\w+'):
    return RegexpTokenizer(pattern)


def tokenize(text, lowercase=True):
    if lowercase:
        text = text.lower()
    return get_tokenizer().tokenize(text)


def train_phrase_model(texts, num_models=1, delimiter=b'$DELIM$'):
    """ texts can be iter if num_models is 1, but must be list otherwise! """
    from gensim.models import Phrases

    def make_token_stream():
        sentence_splitter = re.compile('[.!?\n]')
        sentences = toolz.concat(map(sentence_splitter.split, texts))
        return map(tokenize, sentences)

    threshold_scale = 5
    models = []

    for idx in range(num_models):
        model = Phrases(
            _make_models_iter(make_token_stream(), models),
            threshold=(idx+1)*threshold_scale,
            delimiter=delimiter
        )
        models.append(model)

    if num_models == 1:
        return models[0]
    else:
        return models


def _make_models_iter(stream, models):
    if len(models) == 0:
        return stream
    else:
        return _make_models_iter(models[0][stream], models[1:])


def apply_phrase_models(models, text):
    text_iter = iter(text)
    head = next(text_iter)
    if isinstance(head, str):
        # Iter of texts, need to tokenize
        token_iter = toolz.concat([[tokenize(head)], map(tokenize, text_iter)])
    elif isinstance(head, list):
        token_iter = text_iter
    else:
        raise ValueError("Received unexpected type {}, expected text to contain strings or "
                         "lists of strings (tokens)".format(type(head)))

    return _make_models_iter(token_iter, models)
