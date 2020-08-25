import json, os
import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizer

def _get_tokenize_position(c_list, a_list, tokenizer):
    _flag = False
    tmp_sp = -1
    x = 0
    tmp_list = []

    for _pos, c_char in enumerate(c_list[:]):
        for a_char in a_list[x:]:
            if(a_char == c_char):
                tmp_list.append(a_char)
                x += 1
                
                if(not _flag):
                    tmp_sp = _pos
                    _flag = True
            else:
                _flag = False
            break

        if(len(tmp_list)==len(a_list)):
            return tmp_list, tmp_sp, (tmp_sp + len(a_list))

        if(not _flag and len(tmp_list)>0): 
            x = 0
            tmp_list = []
            pass    
    
    return None, None, None


def make_drcd(loadfile, savefile, tokenizer, MAX_SEN_LEN=10000):
    group = ['sentence', 'question', 'answser', 'start_position', 'end_position']
    df = pd.DataFrame(columns = group)

    with open(loadfile, 'r') as _file:
        content = _file.read()

    drcd = json.loads(content)
    drcd_data =  drcd['data']
    nan=0

    for idx, row in enumerate(tqdm(drcd_data)):
        for paragraphs in row['paragraphs']:
            
            if( len(paragraphs['context']) <= MAX_SEN_LEN):
                qas = paragraphs['qas']

                c_list = tokenizer.tokenize(paragraphs['context'])

                for qas in paragraphs['qas']:
                    ans = qas['answers'][0]['text']
                    ans_len = len(ans)

                    answer_start = qas['answers'][0]['answer_start']
                    end_position = answer_start + ans_len - 1
                    
                    a_list = tokenizer.tokenize(ans)
                    
                    ans_in_clist, start_position, end_position = _get_tokenize_position(c_list, a_list, tokenizer)

                    if(ans_in_clist!=None):
                        # display(qas['question'], c_list, a_list ,ans_in_clist ,start_position ,end_position)
                    
                        # add one row of df
                        df_row = pd.DataFrame([
                            (
                            ''.join(c_list), 
                            qas['question'], 
                            ''.join(a_list), 
                            start_position, 
                            end_position)
                            ] , 
                            columns = group)
                        df = df.append(df_row, ignore_index=True)
                    else:
                        nan +=1
        pass
        
    print(nan)
    df.to_csv(savefile, index = False, header=True)
    pass

def display(que, c_list, a_list, ans_in_clist, start_position, end_position):
    print(f"""
--------------------------
que : {que}
sen : {''.join(c_list[start_position-20:end_position+20])}
ans : {''.join(a_list)}
ans_in_clist : {ans_in_clist}
start_position : {start_position}
end_position : {end_position}
    """)    
    pass

if __name__ == "__main__":
    # Get model tokenizer
    PRETRAINED_MODEL_NAME = "bert-base-chinese"
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

    make_drcd(
        'BERT-QA/data/DRCD_training.json',
        'BERT-QA/data/DRCD_training_ml300.csv',
        tokenizer,
        MAX_SEN_LEN = 300        
    )

    pass