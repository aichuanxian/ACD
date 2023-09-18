from torch import nn

from model.layers.crf import CRF


class CrfTagger(nn.Module):
    def __init__(self, input_dim, tag_num):
        super().__init__()
        self.input_dim = input_dim
        self.tag_num = tag_num

        self.crf = CRF(tag_num)

        self.linear = nn.Linear(input_dim, tag_num)

    def forward(self, inputs, mask):
        """
        :param inputs: bsz, max_seq_len, feat_size
        :param mask: bsz, max_seq_len
        :return:
        """
        return self.crf(self.linear(inputs))

    def loss(self, logits, tags):
        """
        :param logits: bsz, max_seq_len, 2
        :param tags: bsz, max_seq_len
        :return:
        """
        return self.crf.neg_log_likelihood_loss(logits, (tags >= 0), tags)

    def decode(self, logits, mask):
        tag_paths, scores = self.crf.viterbi_decode(logits, mask)
        return tag_paths
