
def displaySample(dataset, tokenizer, sample_idx = 0):

    # 將原始文本拿出做比較
    text_a, text_b, text_c, sp, ep = dataset.df.iloc[sample_idx].values

    # 利用剛剛建立的 Dataset 取出轉換後的 id tensors
    tokens_tensor, segments_tensor, start_position, end_position = dataset[sample_idx]

    # 將 tokens_tensor 還原成文本
    tokens = tokenizer.convert_ids_to_tokens(tokens_tensor.tolist())
    combined_text = "".join(tokens)

    # 將 tokens_tensor 還原成文本
    ans_tokens = tokenizer.convert_ids_to_tokens(
        tokens_tensor[start_position:end_position].tolist()
        )
    ans_combined_text = "".join(ans_tokens)

    # 渲染前後差異，毫無反應就是個 print。可以直接看輸出結果
    print(f"""[Oringinal text]
    Quesion：{text_b}
    Sentence：{text_a}
    Answer：{text_c}
    sp：{start_position}
    ep：{end_position}

    --------------------

    [tensors of Dataset]
    tokens_tensor  ：{tokens_tensor}

    segments_tensor：{segments_tensor}

    start_position：{start_position}

    end_position：{end_position}

    --------------------

    [還原 tokens_tensors]
    {combined_text}

    [還原 answer]
    {ans_combined_text}
    """)   
    stop = 1 
    pass

def displayBertModules(model):
    # high-level 顯示此模型裡的 modules
    print("""
    name            module
    ----------------------""")
    for name, module in model.named_children():
        if name == "bert":
            for n, _ in module.named_children():
                print(f"{name}:{n}")
        else:
            print("{:15} {}".format(name, module))