from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from tensorboardX import SummaryWriter
import os
import torch
from datetime import datetime
from utils.metric import compute_f1
from datasets import build_tokens

class BertQATrainer():
    def __init__(self, args, model, train_loader, val_loader, test_set):
        
        self.args = args
        self.device = args.device
        self.model = model.to(self.device)

        if(args.do_train):
            self.train_loader = train_loader
            self.val_loader = val_loader

            self.optimizer = self._create_optimizer()

            # 使用 Adam Optim 更新整個分類模型的參數
            t_total = len(self.train_loader) // args.gradient_accumulation_steps * args.num_train_epochs
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer, 
                num_warmup_steps=args.warmup_steps, 
                num_training_steps = t_total
                )

            self.output_dir= self._create_output_folder()
            self.tb_writer = SummaryWriter('{}/runs'.format(self.output_dir))
            pass

        if(args.do_eval):
            self.test_set = test_set
            os.makedirs('{}/predict'.format(args.load_model_path), exist_ok=True)
            pass

        pass

    def train(self):
        
        self.model.train()
        self.optimizer.zero_grad()
        self.model.zero_grad()

        best_loss = 100000.0
        running_loss, logging_loss = 0.0, 0.0
        
        global_step = 0
        logging_steps = 50

        for epoch in range(self.args.num_train_epochs):
            
            ep_loss = 0

            for step , batch in enumerate(tqdm(self.train_loader, desc="Epoch")):
                tokens_tensors, segments_tensors, masks_tensors, start_position, end_position = batch
                tokens_tensors, segments_tensors, masks_tensors, start_position, end_position = \
                    tokens_tensors.to(self.device), segments_tensors.to(self.device), masks_tensors.to(self.device), start_position.to(self.device), end_position.to(self.device)

                # 计算loss
                loss, _, _ = self.model(
                    tokens_tensors, 
                    token_type_ids=segments_tensors, 
                    attention_mask=masks_tensors, 
                    start_positions=start_position, 
                    end_positions=end_position
                    )

                loss.backward()

                # 紀錄當前 batch loss
                running_loss += loss.item()
                ep_loss += loss.item()
                
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.model.zero_grad()
                    self.scheduler.step()  # Update learning rate schedule
                    
                    global_step += 1

                    self.tb_writer.add_scalar('lr', self.scheduler.get_lr()[0], global_step)
                    self.tb_writer.add_scalar(
                        'train loss', 
                        (running_loss - logging_loss)/logging_steps, 
                        global_step
                    )
            
                    logging_loss = running_loss
                    pass

                if ((step + 1) % self.args.evaluate_accumulation_steps == 0 and self.args.evaluate_during_training):
                    eval_loss = self._validate()
                    self.tb_writer.add_scalar(
                        'validation loss', 
                        eval_loss, 
                        global_step
                        )
                    
                    if(eval_loss < best_loss):
                        best_loss = eval_loss
                        torch.save(
                            self.model.state_dict(), 
                            '{}/model/best_model.bin'.format(self.output_dir)
                            )
                    
                    self.model.train()
                    pass
                pass
            pass
        pass

    def evaluate(self, tokenizer, _write=False):
        f1_scores = []
        with torch.no_grad():
            self.model.eval()
            for index in tqdm(range(len(self.test_set))):

                # 將原始文本拿出做比較
                sen, que, ans, sp, ep = self.test_set.df.iloc[index].values
                # 利用剛剛建立的 Dataset 取出轉換後的 id tensors
                tokens_tensor, segments_tensor, _sp, _ep = self.test_set[index]
                # 將 tokens_tensor 還原成文本
                tokens = tokenizer.convert_ids_to_tokens(tokens_tensor.tolist())
                input_mask = torch.tensor([1] * len(tokens_tensor))

                tokens_tensor, segments_tensor, input_mask = \
                    tokens_tensor.to(self.device), segments_tensor.to(self.device), input_mask.to(self.device)

                # Evaluation
                start_prob, end_prob = self.model(
                    tokens_tensor.unsqueeze(0), 
                    token_type_ids=segments_tensor.unsqueeze(0),
                    attention_mask = input_mask.unsqueeze(0)
                    )
                
                prob_start, best_start = torch.max(start_prob, 1)
                prob_end, best_end = torch.max(end_prob, 1)

                gold_toks = tokenizer.tokenize(ans)
                pred_toks = tokens[best_start:best_end]

                # Compute F1 score
                f1 = compute_f1(gold_toks, pred_toks, tokenizer)
                f1_scores.append(f1)

                # Parm. for simplily display
                displayLen = 30
                displayStart = best_start-displayLen if best_start-displayLen>0 else 0
                displayEnd = best_end+displayLen if best_end+displayLen>0 else 0

                w_text = (f"""
    --------------------            
    [Oringinal text]
    Quesion：{que}
    Sentence：{''.join(tokens[displayStart:displayEnd])}
    Answer：{ans}

    Predict: 
    {''.join(pred_toks)}

    f1:
    {f1}
                """)

                if(_write):
                    with open('{}/predict/result.txt'.format(self.args.load_model_path), 'a') as _file:
                        _file.writelines(w_text)
                        pass
                    pass
                pass

            print('Average f1 : {}'.format(sum(f1_scores)/len(f1_scores)))
        pass

    def interaction(self, tokenizer, qa_text):

        que = qa_text['question']
        sen = qa_text['sentence']

        tokens_tensor, segments_tensor = build_tokens(tokenizer, que, sen)

        with torch.no_grad():
            self.model.eval()

            # 將 tokens_tensor 還原成文本
            tokens = tokenizer.convert_ids_to_tokens(tokens_tensor.tolist())
            input_mask = torch.tensor([1] * len(tokens_tensor))

            tokens_tensor, segments_tensor, input_mask = \
                tokens_tensor.to(self.device), segments_tensor.to(self.device), input_mask.to(self.device)

            # Evaluation
            start_prob, end_prob = self.model(
                tokens_tensor.unsqueeze(0), 
                token_type_ids=segments_tensor.unsqueeze(0),
                attention_mask = input_mask.unsqueeze(0)
                )
            
            prob_start, best_start = torch.max(start_prob, 1)
            prob_end, best_end = torch.max(end_prob, 1)

            pred_toks = tokens[best_start:best_end]

            print(f"""
                
Predict: 
{''.join(pred_toks)}
            """)
        pass

    def _validate(self):
        total_loss = 0.0
        for batch in self.val_loader:
            self.model.eval()
            
            tokens_tensors, segments_tensors, masks_tensors, start_position, end_position = batch
            tokens_tensors, segments_tensors, masks_tensors, start_position, end_position = \
                tokens_tensors.to(self.device), segments_tensors.to(self.device), masks_tensors.to(self.device), start_position.to(self.device), end_position.to(self.device)

            with torch.no_grad():
                loss, _, _ = self.model(
                    tokens_tensors, 
                    token_type_ids=segments_tensors, 
                    attention_mask=masks_tensors,
                    start_positions=start_position, 
                    end_positions=end_position
                    )            
                total_loss += loss.item()
                pass
        return total_loss/len(self.val_loader)

    def _create_optimizer(self):
        args = self.args
        optimizer = AdamW(
            self.model.parameters(), 
            lr=args.learning_rate, 
            eps=args.adam_epsilon
            )
        return optimizer
    
    def _create_output_folder(self):
        current_time = datetime.now()
        output_dir= '{}/{:%Y%m%d_%H_%M}'.format(self.args.output_dir, current_time)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs('{}/model'.format(output_dir), exist_ok=True)
        return output_dir