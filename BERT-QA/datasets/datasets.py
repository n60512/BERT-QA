from torch.utils.data import Dataset
from .make_drcd import make_drcd
import torch, pandas, os

class DRCDDataset(Dataset):
    def __init__(self, mode, tokenizer, _file = 'BERT-QA/data/DRCD_train.csv', _ml=10000):
        assert mode in ["train", "test"]

        self.mode = mode

        if(os.path.isfile(_file)):
            print('{} exists.'.format(_file))
        else:
            print('{} does not exists.\nCreat csv file  ...'.format(_file))
            make_drcd(
                'BERT-QA/data/DRCD_{}.json'.format(mode),
                _file,
                tokenizer,
                MAX_SEN_LEN = _ml
                )

        self.df = pandas.read_csv(_file, sep=",")
        self.len = len(self.df)
        self.tokenizer = tokenizer
    

    def __getitem__(self, idx):
        sen = self.df.iloc[idx, 0]
        que = self.df.iloc[idx, 1]
        ans = self.df.iloc[idx, 2]
        sp = self.df.iloc[idx, 3]
        ep = self.df.iloc[idx, 4]

        # 建立第一個句子的 BERT tokens 並加入分隔符號 [SEP]
        word_pieces = ["[CLS]"]
        tokens_que = self.tokenizer.tokenize(que)
        word_pieces += tokens_que + ["[SEP]"]
        len_que = len(word_pieces)
        
        # BERT tokens of `answer`
        tokens_sen = self.tokenizer.tokenize(sen)
        word_pieces += tokens_sen + ["[SEP]"]
        len_sen = len(word_pieces) - len_que
        
        # 將整個 token 序列轉換成索引序列
        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        tokens_tensor = torch.tensor(ids)
        
        # 將第一句包含 [SEP] 的 token 位置設為 0，其他為 1 表示第二句
        segments_tensor = torch.tensor([0] * len_que + [1] * len_sen, 
                                        dtype=torch.long)

        start_position = torch.tensor(len_que + sp)
        end_position = torch.tensor(len_que + ep)

        return (tokens_tensor, segments_tensor, start_position, end_position)

    def __len__(self):
        return self.len