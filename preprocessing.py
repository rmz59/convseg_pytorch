from typing import List


# -----------------------------------------------------------------------------
#               Process vocabulary
# -----------------------------------------------------------------------------
def build_vocab(items, add_unk=True, add_pad=True):
    """ build vocabularies from a item list
    Args:
        items: list of items
        add_unk: bool. Whether to add unk 
    
    >>> build_vocab(["a","b","b"], add_unk=False, add_pad=True)
    {'a': 1, 'b': 2, '<PAD>': 1e+20}
    """
    assert type(items) in (list, tuple)
    dic = {}
    for item in items:
        dic[item] = dic.get(item, 0) + 1
    if add_pad:
        dic['<PAD>'] = 1e20
    if add_unk:
        dic['<UNK>'] = 1e10
    return dic

def add_id(items):
    """ add ids to tokents

    Args:
        items: dict or list
    Returns:
        token2id, id2token: dict, dict
    
    >>> add_id({'a':10,'b':2,'<UNK>':1e20})
    ({'<UNK>': 0, 'a': 1, 'b': 2}, {0: '<UNK>', 1: 'a', 2: 'b'})
    >>> add_id(["S","D","E","S"])
    ({'S': 3, 'D': 1, 'E': 2}, {0: 'S', 1: 'D', 2: 'E', 3: 'S'})
    """
    if type(items) is dict:
        sorted_items = sorted(items.items(), key=lambda x: (-x[1], x[0]))
        id2token = {i: v[0] for i, v in enumerate(sorted_items)}
        token2id = {v: k for k, v in id2token.items()}
        return token2id, id2token
    elif type(items) is list:
        id2token = {i: v for i, v in enumerate(items)}
        token2id = {v: k for k, v in id2token.items()}
        return token2id, id2token


# -----------------------------------------------------------------------------
#               Process Sentence
# -----------------------------------------------------------------------------
def process_sentence(sentence: str, stage='train'):
    """ process a single sentence

    get characters and tags 
    Args:
        sentence: string. 
        stage: 'train' or 'test'. 

    Example:
    >>> process_sentence("中国 中央 电视台", stage='train')
    (['中', '国', '中', '央', '电', '视', '台'], ['B', 'E', 'B', 'E', 'B', 'M', 'E'])
    """
    chars = []
    tags = [] if stage == 'train' else None
    for word in sentence.strip().split():
        chars.extend(list(word))
        if stage == 'train':
            if len(word) == 1:
                tags.append('S')
            else:
                tags.extend(['B'] + ['M'] * (len(word) - 2) + ['E'])
    return chars, tags


def get_feature(sentences: List[str], stage='train'):
    """ generate features

    get text features from sentences
    Args:
        sentences: list of strings. 
        stage: 'train' or 'test'. In stage 'train', we will 
    Return:
        chars_array: List[List[str]]  List of characters in the sent
        tags: List[List[str]]. List of tags for each characters in each string
              empty list in test stage.

    Example:
    >>> sent = ["中国 中央 电视台", "南京 市 长江 大桥"]
    >>> get_feature(sent, stage="train")
    ([['中', '国', '中', '央', '电', '视', '台'], ['南', '京', '市', '长', '江', '大', '桥']], [['B', 'E', 'B', 'E', 'B', 'M', 'E'], ['B', 'E', 'S', 'B', 'E', 'B', 'E']])
    """
    chars_array = []
    tags_array = []
    for sent in sentences:
        chars, tags = process_sentence(sent, stage)
        chars_array.append(chars)
        if stage == 'train':
            tags_array.append(tags)
    return chars_array, tags_array
