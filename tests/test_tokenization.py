
from text_tools import tokenize


def test_basic_tokenize():
    text = 'This is how we do it.'
    expected = ['this', 'is', 'how', 'we', 'do', 'it']

    tokens = tokenize(text, True)
    assert tokens == expected
