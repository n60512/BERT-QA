python BERT-QA \
    --train_file BERT-QA/data/DRCD_train_ml300.csv \
    --predict_file BERT-QA/data/DRCD_test.json \
    --device cuda \
    --model_name_or_path bert-base-chinese \
    --load_model_path BERT-QA/log/output/20200825_00_23 \
    --output_dir BERT-QA/log/output \
    --do_eval