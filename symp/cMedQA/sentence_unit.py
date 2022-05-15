from word_unit import WordUnit


class SentenceUnit:
    '''composed of WordUnit
    param
        seg_lst: list<str>
        dep_lst: list<tuple<int, int, str>>
        ne_pos_lst: list<int>
        ne_dct: list<dict<'word': str, 'type': str, 'pos': tuple<int, int>>>

    '''

    def __init__(self, seg_lst, dep_lst, ne_pos_lst, ne_dct):
        self.words = []
        ne_idx = 0
        for i in range(len(seg_lst)):
            self.words.append(WordUnit(seg_lst[0]))
