import torch
import torch.nn as nn

from transformers.models.bert.modeling_bert import BertModel
from transformers import BertTokenizer




class ACDModelWithCombinedPrompt(nn.Module):
    def __init__(self, args):
        super(ACDModelWithCombinedPrompt, self).__init__()
        self.args = args
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path=args.model.model_name_or_path,
                                              output_attentions=args.model.output_attentions,
                                              output_hidden_states=args.model.output_hidden_states)
        self.fc = nn.Linear(args.model.bert_hidden_size, args.model.num_class)
        self.embeddings = self.bert.embeddings
        self.dropout = nn.Dropout(args.model.hidden_dropout_prob)
        self.softmax = nn.Softmax(dim=-1)
        self.loss_fct = nn.CrossEntropyLoss()

        for param in self.bert.parameters():
            param.requires_grad = False

        # 固定的提示语句
        self.fixed_prompt = "1.地区，2.医护，3.生活保障，4.政府机构，5.感染者，6.经济，7.政策宣传，8.疫情"
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=args.model.model_name_or_path,
                                                       do_lower_case=args.model.do_lower_case)
        self.fixed_prefix_tokens = torch.tensor(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(self.fixed_prompt)), device=self.bert.device)
        self.fixed_prefix_encoder = torch.nn.Embedding(len(self.fixed_prefix_tokens), args.model.bert_hidden_size)

       # 从config中获取预设序列长度等信息
        self.pre_seq_len = args.model.pre_seq_len
        # 创建一个表示提示令牌的张量
        self.prefix_tokens = torch.arange(self.args.model.pre_seq_len).long()
        # 创建一个嵌入层，用于将提示令牌嵌入到指定的隐藏维度中
        self.prefix_encoder = torch.nn.Embedding(self.args.model.pre_seq_len, args.model.bert_hidden_size)

    def get_random_prompt(self, batch_size):
        # 生成随机的提示语句（示例中随机生成一个数字）
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.bert.device)
        prompts = self.prefix_encoder(prefix_tokens)
        return prompts

    def forward(
        self, 
        input_ids, 
        input_mask, 
        segment_ids, 
        label=None, 
        inputs_embeds=None, 
        position_ids=None, 
        token_type_ids=None, 
        output_attentions=None,
        output_hidden_states=None,
    ):
        batch_size = input_ids.shape[0]
        
        # 获取固定提示的嵌入表示
        fixed_prompts = self.fixed_prefix_encoder(self.fixed_prefix_tokens.to(self.bert.device))
        
        # 获取随机生成的提示语句嵌入
        random_prompts = self.get_random_prompt(batch_size)

        # 将固定提示和随机提示连接在一起
        combined_prompts = torch.cat((fixed_prompts, random_prompts), dim=1)

        # 创建一个与提示令牌对应的注意力掩码
        prefix_attention_mask = torch.ones(batch_size, len(fixed_prompts[0])).to(self.bert.device)
        input_mask = torch.cat((prefix_attention_mask, input_mask), dim=1)

        # 使用提示嵌入进行前向传播
        outputs = self.bert(
            attention_mask=input_mask,
            inputs_embeds=combined_prompts,  # 使用固定提示和随机提示嵌入
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,           
        )
        
        # 获取Bert模型的序列输出
        sequence_output = outputs[0]
        # 截取序列输出，去掉提示部分
        sequence_output = sequence_output[:, self.pre_seq_len:, :].contiguous()
        # 获取序列输出的第一个令牌
        first_token_tensor = sequence_output[:, 0]

        # 通过一个线性层将第一个令牌的表示映射为池化输出
        pooled_output = self.bert.pooler.dense(first_token_tensor)
        # 应用激活函数到池化输出
        pooled_output = self.bert.pooler.activation(pooled_output)
        # 应用Dropout以防止过拟合
        pooled_output = self.dropout(pooled_output)
        # 通过线性分类器生成分类的logits
        output = self.fc(pooled_output)

        #output = self.fc(first_token_tensor)
        output = self.softmax(output)
        # 初始化损失值为None
        loss = None

        if label is not None:
            loss = self.loss_fct(output, label)
            return loss, output

        return output
