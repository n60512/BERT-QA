import torch
from torch.utils.data import DataLoader, random_split
from preprocessing import create_mini_batch
from datasets.datasets import DRCDDataset
from utils.displaydata import displaySample

def dataset_factory(args, tokenizer):

    if(args.do_train):
        mode = 'train'
        # 初始化一個專門讀取訓練樣本的 Dataset，使用中文 BERT 斷詞
        full_dataset = DRCDDataset(
            mode, 
            tokenizer=tokenizer, 
            _file='BERT-QA/data/DRCD_{}_ml{}.csv'.format(mode, args.max_seq_length), 
            _ml=args.max_seq_length
            )        

        train_size = int(0.8 * len(full_dataset))
        validation_size = len(full_dataset) - train_size
        trainset, validationset = random_split(
            full_dataset, 
            [train_size, validation_size]
            )

        # displaySample(trainset, tokenizer, sample_idx =77)
        # 初始化一個每次回傳 64 個訓練樣本的 DataLoader
        # 利用 `collate_fn` 將 list of samples 合併成一個 mini-batch 是關鍵
        train_loader = DataLoader(
            trainset, 
            batch_size=args.batch_size, 
            collate_fn=create_mini_batch
            )

        val_loader = DataLoader(
            validationset,
            batch_size=args.batch_size, 
            collate_fn=create_mini_batch
        )

        return train_loader, val_loader

    elif (args.do_eval):
        mode = 'test'
        # 初始化一個專門讀取訓練樣本的 Dataset，使用中文 BERT 斷詞
        test_set = DRCDDataset(
            mode, 
            tokenizer=tokenizer, 
            _file='BERT-QA/data/DRCD_{}_ml{}.csv'.format(mode, args.max_seq_length), 
            _ml=args.max_seq_length
            )
        return test_set

def build_tokens(tokenizer, que, sen):

    # 建立第一個句子的 BERT tokens 並加入分隔符號 [SEP]
    word_pieces = ["[CLS]"]
    tokens_que = tokenizer.tokenize(que)
    word_pieces += tokens_que + ["[SEP]"]
    len_que = len(word_pieces)
    
    # BERT tokens of `answer`
    tokens_sen = tokenizer.tokenize(sen)
    word_pieces += tokens_sen + ["[SEP]"]
    len_sen = len(word_pieces) - len_que
    
    # 將整個 token 序列轉換成索引序列
    ids = tokenizer.convert_tokens_to_ids(word_pieces)
    tokens_tensor = torch.tensor(ids)
    
    # 將第一句包含 [SEP] 的 token 位置設為 0，其他為 1 表示第二句
    segments_tensor = torch.tensor([0] * len_que + [1] * len_sen, 
                                    dtype=torch.long)

    return tokens_tensor, segments_tensor