import argparse



parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("--train_file", default=None, type=str, required=True,
                   help="SQuAD json for training. E.g., train-v1.1.json")
parser.add_argument("--predict_file", default=None, type=str, required=True,
                   help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")
parser.add_argument('--device', type=str, required=True, default='cpu', choices=['cpu', 'cuda'])
parser.add_argument("--output_dir", default=None, type=str, required=False,
                    help="The output directory where the model checkpoints and predictions will be written.")
parser.add_argument("--do_train", action='store_true',
                    help="Whether to run training.")
parser.add_argument("--do_eval", action='store_true',
                    help="Whether to run eval on the dev set.")
parser.add_argument("--do_interaction", action='store_true',
                    help="Whether to run eval on the dev set.")
parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                    help="Path to pre-trained model or shortcut name selected in the list: ")
parser.add_argument("--load_model_path", default=None, type=str, required=False,
                    help="Path to evaluation model")


parser.add_argument("--evaluate_during_training", action='store_true',
                    help="Rul evaluation during training at each logging step.")
parser.add_argument('--evaluate_accumulation_steps', type=int, default=50,
                    help="Number of evaluate steps .")

parser.add_argument("--max_seq_length", default=300, type=int,
                    help="The maximum total input sequence length after WordPiece tokenization. Sequences ")
parser.add_argument("--batch_size", default=4, type=int,
                    help="Batch size per GPU/CPU for training.")
parser.add_argument("--learning_rate", default=3e-5, type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument('--gradient_accumulation_steps', type=int, default=8,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                    help="Epsilon for Adam optimizer.")
parser.add_argument("--num_train_epochs", default=3, type=int,
                    help="Total number of training epochs to perform.")
parser.add_argument("--warmup_steps", default=0, type=int,
                    help="Linear warmup over warmup_steps.")
args = parser.parse_args()
