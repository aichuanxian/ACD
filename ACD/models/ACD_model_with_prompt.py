import torch
import torch.nn as nn

from transformers.models.bert.modeling_bert import BertModel



class ACDModelWithPrompt(nn.Module):
    def __init__(self, args):
        super(ACDModelWithPrompt, self).__init__()
        self.args = args
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path=args.model.model_name_or_path,
                                              output_attentions=args.model.output_attentions,
                                              output_hidden_states=args.model.output_hidden_states)
        self.fc = nn.Linear(args.model.bert_hidden_size, args.model.num_class)
        # 获取Bert模型的嵌入层，用于处理输入文本的嵌入表示
        self.embeddings = self.bert.embeddings
        # 创建一个Dropout层，用于在模型训练时进行随机失活
        self.dropout = nn.Dropout(args.model.hidden_dropout_prob)
        self.softmax = nn.Softmax(dim=-1)
        self.loss_fct = nn.BCEWithLogitsLoss()

        for param in self.bert.parameters():
            param.requires_grad = False

        # 从config中获取预设序列长度等信息
        self.pre_seq_len = args.model.pre_seq_len
        # 创建一个表示提示令牌的张量
        self.prefix_tokens = torch.arange(self.args.model.pre_seq_len).long()
        # 创建一个嵌入层，用于将提示令牌嵌入到指定的隐藏维度中
        self.prefix_encoder = torch.nn.Embedding(self.args.model.pre_seq_len, args.model.bert_hidden_size)

    def get_prompt(self, batch_size):
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
        output_hidden_states=None,):
        # 获取输入的batch_size
        batch_size = input_ids.shape[0]
        # 使用嵌入层处理原始输入文本，生成嵌入表示
        raw_embedding = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )
        # 调用get_prompt方法获取提示的嵌入表示
        prompts = self.get_prompt(batch_size=batch_size)
        # 将提示的嵌入表示与原始嵌入表示连接在一起，形成最终的输入嵌入表示
        inputs_embeds = torch.cat((prompts, raw_embedding), dim=1)
        # 创建一个与提示令牌对应的注意力掩码
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.bert.device)
        # 将提示的注意力掩码与输入文本的注意力掩码连接在一起
        input_mask = torch.cat((prefix_attention_mask, input_mask), dim=1)

        outputs = self.bert(
            attention_mask=input_mask,
            inputs_embeds=inputs_embeds,
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
        output = self.softmax(output)
        # 初始化损失值为None
        loss = None

        if label is not None:
            loss = self.loss_fct(output, label)
            return loss, output

        return output
