import torch.nn as nn

from transformers.models.bert.modeling_bert import BertModel


class ACDModel(nn.Module):
    def __init__(self, args):
        super(ACDModel, self).__init__()
        self.args = args
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path=args.model.model_name_or_path,
                                              output_attentions=args.model.output_attentions,
                                              output_hidden_states=args.model.output_hidden_states)
        self.fc = nn.Linear(args.model.bert_hidden_size, args.model.num_class)
        self.softmax = nn.Softmax(dim=-1)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, input_mask, segment_ids, label=None):
        bert_output = self.bert(input_ids=input_ids,
                                attention_mask=input_mask,
                                token_type_ids=segment_ids)
        output = self.fc(bert_output[1])
        output = self.softmax(output)
        if label is not None:
            loss = self.loss_fn(output, label)
            return loss, output

        return output


