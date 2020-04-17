from typing import List

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

