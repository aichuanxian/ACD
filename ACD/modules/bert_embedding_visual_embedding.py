import os
import torch
import torch.nn as nn
import resnet.resnet as resnet
from resnet.resnet_utils import myResnet


class BertEmbeddingsWithVisualEmbedding(nn.Module):
    """Construct the embeddings from word, position, token_type embeddings and visual embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddingsWithVisualEmbedding, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        #### Below are specific for encoding visual features
        # image infomation
        net = getattr(resnet, 'resnet152')()
        net.load_state_dict(torch.load(os.path.join('./resnet', 'resnet152.pth')))
        # net = resnet.resnet152(pretrained=True)
        self.resnet152 = myResnet(net, True)  # True


        # # Segment and position embedding for image features
        # self.token_type_embeddings_visual = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.position_embeddings_visual = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        if hasattr(config, 'visual_embedding_dim'):
            self.projection = nn.Linear(config.visual_embedding_dim, config.hidden_size)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                visual_raw_datas=None,
                visual_embeddings_type=None,
                visual_position_ids=None):
        '''
        input_ids = [batch_size, sequence_length]
        token_type_ids = [batch_size, sequence_length]
        visual_embedding = [batch_size, image_feature_length, image_feature_dim]
        '''

        input_shape = input_ids.size()
        seq_length = input_ids.size(1)
        device = input_ids.device

        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)


        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings


        # if visual_position_ids is None:
        #     visual_position_ids = torch.arange()


        if visual_raw_datas is not None:
            _, _, enc_img_input_att = self.resnet152(visual_raw_datas)
            enc_img = enc_img_input_att.view(-1, 2048, 49).permute(0, 2, 1)

            visual_embeddings = self.projection(enc_img)

        if visual_position_ids is None:
            visual_position_ids = torch.arange(visual_embeddings.size(1), dtype=torch.long, device=device)
            visual_position_ids = visual_position_ids.unsqueeze(0).repeat(visual_embeddings.size(0), 1)

        # visual_position_ids = torch.zeros(*visual_embeddings.size()[:-1], dtype = torch.long)#.cuda()
        position_embeddings_visual = self.position_embeddings_visual(visual_position_ids)

        if visual_embeddings_type is None:
            visual_embeddings_type = torch.ones_like(visual_position_ids)

            token_type_embeddings_visual = self.token_type_embeddings(visual_embeddings_type)



            v_embeddings = visual_embeddings + position_embeddings_visual + token_type_embeddings_visual

            # Concate the two:
            embeddings = torch.cat((embeddings, v_embeddings), dim = 1) # concat the visual embeddings after the attentions

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings