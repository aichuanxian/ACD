import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import CrossEntropyLoss

from transformers import BertModel, BertPreTrainedModel, BertConfig
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.bert.modeling_bert import BertSelfAttention
from transformers import BertConfig
from transformers import AutoTokenizer


from ACD.modules.prefix_encoder import PrefixEncoder

class ACDModelWithPrefix(nn.Module):
    def __init__(self, args):
        super(ACDModelWithPrefix, self).__init__()
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.model.model_name_or_path) 
        self.num_labels = args.model.num_class
        # self.bert = BertModel(args, add_pooling_layer=False)
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path=args.model.model_name_or_path,
                                        output_attentions=args.model.output_attentions,
                                        output_hidden_states=args.model.output_hidden_states)
        self.dropout = nn.Dropout(args.model.hidden_dropout_prob)
        self.classifier = nn.Linear(args.model.hidden_size, args.model.num_class)
        self.softmax = nn.Softmax(dim=-1)
        # self.cross_att = nn.MultiheadAttention(args.model.hidden_size, args.model.num_attention_heads)

        # 创建一个新的 BertConfig 对象
        config = BertConfig(hidden_size=args.model.hidden_size, num_attention_heads=args.model.num_attention_heads)

        # 使用这个 config 对象来初始化 BertSelfAttention
        self.cross_att = BertSelfAttention(config=config)


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
        # Hard Prompt
        # hard_prompt_text = ["地区", "医疗", "生活保障", "政府","感染者","经济","政策","疫情"]
        # hard_prompt_text = " ".join(hard_prompt_text)
        # hard_prompt_tokens = self.tokenizer.encode(hard_prompt_text, add_special_tokens=False)
        # hard_prompt_tokens = torch.tensor(hard_prompt_tokens).unsqueeze(0).expand(batch_size, -1).to(self.bert.device)
        # hard_prompt = self.bert.embeddings(hard_prompt_tokens)  


        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.bert.device)
        # prompt = torch.cat((hard_prompt, prefix_tokens))
        past_key_values = self.prefix_encoder(prefix_tokens)
        # bsz, seqlen, _ = past_key_values.shape

        # Combine Hard Prompt and Soft Prompt
        
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
        

        # prompt_features = [hidden_state[:, :self.pre_seq_len] for hidden_state in outputs[2][1:]]
        prompt_features = [hidden_state[:, :self.pre_seq_len] for hidden_state in outputs[2]]
        prompt_features = torch.stack(prompt_features) # shape: (num_layers, batch_size, seq_len, hidden_size)
        # print(f'Number of layers in prompt_features: {len(prompt_features)}')

        # 获取CLS标记的特征
        cls_features = outputs[0][:, 0] # shape: (batch_size, hidden_size)
        # print(f'Number of layers in cls_features: {len(cls_features)}')
        cross_att_features_list, attention_weights_list= [], []
        for layer_prompt_features in prompt_features:
            # 转换维度以匹配MultiheadAttention的输入要求
            layer_prompt_features = layer_prompt_features.permute(1, 0, 2) # shape: (batch_size, seq_len, hidden_size)
            cls_features_unsqueeze = cls_features.unsqueeze(1) # shape: (batch_size, 1, hidden_size)
            # 对每一层的prompt特征和CLS标记的特征进行交叉注意力操作
            cross_att_outputs = self.cross_att(hidden_states=cls_features_unsqueeze.transpose(0, 1),
                                            attention_mask=None,
                                            head_mask=None,
                                            encoder_hidden_states=layer_prompt_features.transpose(0, 1),
                                            encoder_attention_mask=None,
                                            past_key_value=None,
                                            output_attentions=True)
            cross_att_features, attention_weights = cross_att_outputs[0], cross_att_outputs[1]
            cross_att_features_list.append(cross_att_features)
            attention_weights_list.append(attention_weights)


        cross_att_features = torch.cat(cross_att_features_list, dim=0) # shape: (num_layers*batch_size, 1, hidden_size)
        cross_att_features = cross_att_features.transpose(0, 1)  # shape: (1, num_layers*batch_size, hidden_size)
        
        # 将 cross_att_features 调整为 (num_layers, batch_size, hidden_size)
        cross_att_features = cross_att_features.permute(1, 0, 2)        

        first_cross_att_features = torch.mean(cross_att_features, dim=0) # 沿 num_layers 维度取平均

        # # 将first_cross_att_features和cls_features进行残差连接
        residual_features = first_cross_att_features + cls_features
        dropout_output = self.dropout(residual_features)
        dropout_output = dropout_output.unsqueeze(1)

        logits = self.classifier(dropout_output[:, 0])
        attention_mask = attention_mask[:,self.pre_seq_len:].contiguous()

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            # print(f'labels shape:{labels.shape}')
            loss = loss_fct(logits, labels)
            # print(f'loss:{loss}')

        if not return_dict:
            # output = (logits,) + outputs[2:]
            output = self.softmax(logits)
            # print(f'output:{output}')
            return (loss, output, attention_weights_list) if loss is not None else output
            # return (loss, output, attention_weights) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )