import os


def load_datas(args, flag):

    if flag == 'train':
        input_path = args.data.text.train
        label_path = args.data.text.train_label
    elif flag == 'valid':
        input_path = args.data.text.valid
        label_path = args.data.text.valid_label
    elif flag == 'test':
        input_path = args.data.text.test
        label_path = args.data.text.test_label

    with open(input_path, 'r', encoding='utf-8') as f:
        datas = f.readlines()
    with open(label_path, 'r', encoding='utf-8') as f:
        labels = f.readlines()

    datas = [sent.rstrip() for sent in datas]
    labels = [l.rstrip() for l in labels]
    label_dict = {l:k for k,l in enumerate(set(labels))}
    assert len(datas) == len(labels)

    label_ids = [label_dict.get(l, None) for l in labels]
    assert len(labels) == len(label_ids)


    return datas, label_ids, labels, label_dict


def bert_tokenize(tokenizer, inputs):

    all_tokenized, all_token_ids, all_mask_ids, all_segment_ids = [], [], [], []

    for sent in inputs:
        tokenized = tokenizer.tokenize(sent)

        tokenized = [tokenizer.cls_token] + tokenized + [tokenizer.sep_token]
        tokenize_ids = tokenizer.convert_tokens_to_ids(tokenized)
        all_tokenized.append(tokenized)
        all_token_ids.append(tokenize_ids)
        all_mask_ids.append([1] * len(tokenize_ids))
        all_segment_ids.append([0] * len(tokenize_ids))

    return all_tokenized, all_token_ids, all_mask_ids, all_segment_ids




