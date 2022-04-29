# %%
import jieba


def _remove_stop_words(words, stopwords_path):
    '''Remove stopwords
    Parameters
        words: list<str>, List of word segmentation
        stopwords_path: str, Path of stopwords text file

    Returns
        list<str>, A list that after remove stopwords
    '''
    result = []
    stopwords = set(line.strip() for line in open(stopwords_path))
    for word in words:
        if word not in stopwords:
            result.append(word)
    return result


def word_segment(sentence, stopwords_path):
    '''Word segment and remove stopwords
    Parameters
        sentence: str, Raw text
        stopwords_path: str, Path of stopwords text file

    Returns
        list<str>, A list that after segment and remove stopwords
    '''
    words = list(jieba.cut(sentence))
    words = _remove_stop_words(words, stopwords_path)
    return words
