import codecs
from pathlib import Path
from torch.utils.data import Dataset


class SIGHAN(Dataset):
    def __init__(self, split, root_path):
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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

