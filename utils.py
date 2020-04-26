import codecs
from pathlib import Path
from typing import List
from collections.abc import Iterable
from torch.utils.data import Dataset

# -----------------------------------------------------------------------------
#               Datasets
# -----------------------------------------------------------------------------
class SIGHAN(Dataset):
    def __init__(self, split, root_path, debug=False):
        """ Create SIGHAN datasets

        Arg:
            root_path: the root path of datasets, including 3 txt files, example:
                sighan2005-msr
                ├── dev.txt
                ├── test.txt
                └── train.txt            
            split: name of the split. ['dev', 'test', 'train']
        """
        assert split in ['dev', 'test', 'train'], "unknown splits: must be in ['dev', 'test', 'train']"
        self.root_path = root_path
        self.file_name = f"{split}.txt"
        self.file_path = Path(self.root_path) / self.file_name

        with codecs.open(self.file_path, 'r', 'utf8') as f:
            self.data = list(map(lambda sent: sent.strip(), f.readlines()))
        
        if debug:
            self.data = self.data[:10]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]




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
    if stage == 'train':
        chars_array = []
        tags_array = []
        for sent in sentences:
            chars, tags = process_sentence(sent, stage)
            if chars:
                chars_array.append(chars)
            if tags and stage == 'train':
                tags_array.append(tags)
        return chars_array, tags_array
    elif stage == 'predict':
        chars_array = []
        for sent in sentences:
            chars, tags = process_sentence(sent, stage)
            if chars:
                chars_array.append(chars)
            else:
                chars_array.append(['EMPTY'])
        return chars_array

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
    assert isinstance(items, Iterable), "input 'items' is not iterable"
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

def token2id(token_lists, map_token_to_id, unk_id=1):
    """ transform tokens to ids

    Args:
        token_lists: List[List[Char]] -> input list of lists
        map_token_to_id: dict
    Returns:
        ids: List[List[int]]
    
    >>> token_lists = [["中","国","结"]]
    >>> map_token_to_id = {"中":2,"国":3,"<UNK>":1}
    >>> token2id(token_lists, map_token_to_id)
    [[2, 3, 1]]
    """
    ids = []
    for tokens in token_lists:
        ids.append([map_token_to_id.get(t, unk_id) for t in tokens])
    return ids


def id2token(id_lists, map_id_to_token):
    """ transform tokens to ids

    Args:
        token_lists: List[List[int]] -> input list of lists
        map_token_to_id: dict
    Returns:
        ids: List[List[char]]
    """
    tokens = []
    for ids in id_lists:
        tokens.append([map_id_to_token[i] for i in ids])
    return tokens


def pad_sents(sents, pad_token, max_len=200):
    """ Pad list of sentences according to the longest sentence in the batch.
        The paddings should be at the end of each sentence.
    
    Args:   
        sents: list[list[str]]list of sentences, where each sentence
                                    is represented as a list of words
        pad_token (str): padding token
    Returns: 
        sents_padded: list[list[str]] list of sentences where sentences shorter
            than the max length sentence are padded out with the pad_token, such that
            each sentences in the batch now has equal length.
        padding_mask: list[list[str]] list of masks. 
    
    Example:
    >>> pad_sents([['a','b','c'],['d'],['e','f','g','h']], '<PAD>', 3)
    ([['a', 'b', 'c'], ['d', '<PAD>', '<PAD>'], ['e', 'f', 'g']], [[1, 1, 1], [1, 0, 0], [1, 1, 1]])
    """
    sents_padded = [s[:max_len] + [pad_token]*(max_len - len(s)) for s in sents]
    padding_mask = [[1]*len(s[:max_len]) + [0]*(max_len - len(s)) for s in sents]
    return sents_padded, padding_mask
