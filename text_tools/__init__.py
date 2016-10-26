
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

    def make_models_iter(stream, models):
        if len(models) == 0:
            return stream
        else:
            return make_models_iter(models[0][stream], models[1:])

    for idx in range(num_models):
        model = Phrases(
            make_models_iter(make_token_stream(), models),
            threshold=(idx+1)*threshold_scale,
            delimiter=delimiter
        )
        models.append(model)

    if num_models == 1:
        return models[0]
    else:
        return models

