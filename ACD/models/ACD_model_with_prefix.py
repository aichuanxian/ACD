import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import CrossEntropyLoss

from transformers import BertModel, BertPreTrainedModel, BertConfig
from transformers.modeling_outputs import TokenClassifierOutput

from ACD.modules.prefix_encoder import PrefixEncoder

class ACDModelWithPrefix(nn.Module):
    def __init__(self, args):
        super(ACDModelWithPrefix, self).__init__()
        self.args = args
        self.num_labels = args.model.num_class
        # self.bert = BertModel(args, add_pooling_layer=False)
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path=args.model.model_name_or_path,
                                        output_attentions=args.model.output_attentions,
                                        output_hidden_states=args.model.output_hidden_states)
        self.dropout = nn.Dropout(args.model.hidden_dropout_prob)
        self.classifier = nn.Linear(args.model.hidden_size, args.model.num_class)
        self.softmax = nn.Softmax(dim=-1)

        if args.experiment.with_parameter_freeze:
            print(f'冻结参数')
            for param in self.bert.parameters():
                param.requires_grad = False
        else:
            print(f'未冻结参数')
            for param in self.bert.parameters():
                param.requires_grad = True
        
        self.pre_seq_len = args.model.pre_seq_len
        self.n_layer = args.model.num_hidden_layers
        self.n_head = args.model.num_attention_heads
        self.n_embd = args.model.hidden_size // args.model.num_attention_heads

        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = PrefixEncoder(args)
 

        bert_param = 0
        for name, param in self.bert.named_parameters():
            bert_param += param.numel()
        all_param = 0
        for name, param in self.named_parameters():
            all_param += param.numel()
        total_param = all_param - bert_param
        print('total param is {}'.format(total_param)) # 9860105
    
    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.bert.device)
        past_key_values = self.prefix_encoder(prefix_tokens)
        # bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2, 
            self.n_head,
            self.n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.args.model.use_return_dict

        batch_size = input_ids.shape[0]
        past_key_values = self.get_prompt(batch_size=batch_size)
        # print(f'past_key_values:{past_key_values.size}')
        # input_ids = torch.cat((past_key_values, input_ids))
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.bert.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        # print(f'input_ids:{input_ids.shape},\nprefix_attention_mask:{prefix_attention_mask.shape}\n,attention_mask:{attention_mask.shape}')

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            past_key_values=past_key_values,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        # print(f'sequence_output:{sequence_output}')
        first_token_tensor = sequence_output[:, 0]
        logits = self.classifier(first_token_tensor)
        # print(f'logits:{logits}')
        attention_mask = attention_mask[:,self.pre_seq_len:].contiguous()

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            loss = loss_fct(logits, labels)
            # print(f'loss:{loss}')

        if not return_dict:
            # output = (logits,) + outputs[2:]
            output = self.softmax(logits)
            # print(f'output:{output}')
            return (loss, output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

