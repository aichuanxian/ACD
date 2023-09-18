#! -*- coding: utf-8 -*-

import torch
from torch import nn


class CRF(torch.nn.Module):
    def __init__(self, tag_num, include_start_end=False):
        """
        Initialize CRF by number of tags.
        :param num_tags: number of tags
        :param include_start_end: Set to True if input logits include CRF <start> and <end> tags,
                                  otherwise set to False
        """
        super().__init__()

        self.include_start_end = include_start_end
        self.num_tags_ext = tag_num if include_start_end else tag_num + 2
        self.start_idx = self.num_tags_ext - 2
        self.end_idx = self.num_tags_ext - 1

        # transition score form j-th tag (dim 1) to i-th tag (dim 0)
        self.transitions = nn.Parameter(torch.randn(self.num_tags_ext, self.num_tags_ext))

    def reset_parameters(self):
        torch.nn.init.normal(self.transitions, 0, 1)
        self.transitions.data[self.start_idx, :] = -10000.
        self.transitions.data[:, self.end_idx] = -10000.

    def viterbi_decode(self, logits, mask):
        """
        Infer tag sequences by viterbi algorithm
        :param logits: [bsz, seq_len, n_tags]
        :param mask:  [bsz, seq_len]
        :return: [bsz, seq_len]
        """
        if not self.include_start_end:
            logits = self._pad_start_end(logits)

        bsz, seq_len, n_tags = logits.size()
        prevs = logits.new_full((bsz, self.num_tags_ext), fill_value=-10000)
        prevs[:, self.start_idx] = 0

        ptrs = []
        lens = torch.sum(mask, dim=-1)
        for i in range(seq_len):
            obs = logits[:, i, :].squeeze(dim=1)
            prevs_ext = prevs.unsqueeze(1).expand(bsz, n_tags, n_tags)
            trans_ext = self.transitions.unsqueeze(0).expand_as(prevs_ext)
            prevs_trans_sum = prevs_ext + trans_ext
            prevs_max, prevs_argmax = prevs_trans_sum.max(2)

            prevs_max = prevs_max.squeeze(-1)
            prevs_nxt = prevs_max + obs
            ptrs.append(prevs_argmax.squeeze(-1).unsqueeze(0))

            obs_msk = mask[:, i].unsqueeze(dim=-1)
            prevs = torch.where(obs_msk, prevs_nxt, prevs)

            len_msk = torch.eq(lens, i + 1).unsqueeze(dim=-1).expand_as(prevs_nxt)
            prevs += len_msk * self.transitions[self.end_idx].unsqueeze(0).expand_as(prevs_nxt)

        scores, idx = prevs.max(dim=1)
        scores = scores.squeeze(-1)
        idx_ext = idx.unsqueeze(dim=-1)

        tag_paths = [idx_ext]
        ptrs = torch.cat(ptrs, dim=0)
        for argmax in reversed(ptrs):
            idx_ext = torch.gather(argmax, -1, idx_ext)
            tag_paths.insert(0, idx_ext)

        tag_paths = torch.cat(tag_paths[1:], -1)
        tag_paths = torch.where(tag_paths.lt(self.num_tags_ext - 2), tag_paths, 0)
        tag_paths = torch.where(mask, tag_paths, -1)

        return tag_paths, scores

    def _all_paths_score(self, logits, mask):
        """
        compute scores of all paths
        :param logits: [bsz, seq_len, n_tags]
        :param mask:  [bsz, seq_len]
        :return: [bsz]
        """
        bsz, seq_len, n_tags = logits.size()
        assert n_tags == self.num_tags_ext

        # [bsz, n_tags]
        prevs = logits.new_full((bsz, n_tags), fill_value=-10000)
        prevs[:, self.start_idx] = 0

        for i in range(seq_len):
            # [bsz, n_tags]
            obs = logits[:, i, :].squeeze(dim=1)
            # [bsz, n_tags, n_tags(new)]
            obs_ext = obs.unsqueeze(dim=-1).expand(bsz, n_tags, n_tags)
            # [bsz, n_tags(new), n_tags]
            prevs_ext = prevs.unsqueeze(dim=1).expand(bsz, n_tags, n_tags)
            # [bsz(new), n_tags, n_tags]
            trans_ext = self.transitions.unsqueeze(dim=0).expand_as(prevs_ext)
            # [bsz, n_tags, n_tags]
            scores = prevs_ext + obs_ext + trans_ext
            # [bsz, n_tags]
            prevs_nxt = torch.logsumexp(scores, dim=2).squeeze(-1)

            # [bsz, 1]
            obs_msk = mask[:, i].unsqueeze(dim=-1)
            # [bsz, n_tags]
            prevs = torch.where(obs_msk.bool(), prevs_nxt, prevs)

        # [bsz, n_tags]
        prevs = prevs + self.transitions[self.end_idx].unsqueeze(dim=0).expand_as(prevs)
        # [bsz]
        norm = torch.logsumexp(prevs, dim=1).squeeze(-1)

        return norm

    def _real_path_score(self, logits, mask, tags):
        """
        compute the real path score according to tags
        :param logits: [bsz, seq_len, num_tags]
        :param mask: [bsz, seq_len]
        :param tags: [bsz, seq_len]
        :return: [bsz]
        """
        # convert padding tags to end index
        tags = torch.where(mask.bool(), tags, self.end_idx)

        # [bsz, seq_len, 1]
        tags_ext = tags.unsqueeze(-1)
        # [bsz, seq_len]
        emission_scores = torch.gather(logits, dim=2, index=tags_ext).squeeze(dim=-1)
        # [bsz]
        emission_scores = torch.sum(emission_scores * mask, dim=1)

        bsz, seq_len = tags.size()
        # pad labels with <start> and <end> indices
        # [bsz, seq_len + 2]
        tags_ext = tags.new_full((bsz, seq_len + 2), fill_value=self.end_idx)
        tags_ext[:, 0] = self.start_idx
        tags_ext[:, 1:-1] = tags

        trn = self.transitions

        # obtain transition vector for each label in batch and time-step
        # (except the last ones)
        # [bsz, n_tags, n_tags]
        trn_exp = trn.unsqueeze(0).expand(bsz, *trn.size())
        # [bsz, seq_len + 1]
        lbl_r = tags_ext[:, 1:]
        # [bsz, seq_len + 1, n_tags]
        lbl_r_ext = lbl_r.unsqueeze(dim=-1).expand(*lbl_r.size(), self.num_tags_ext)
        # [bsz, seq_len + 1, n_tags]
        trn_row = torch.gather(trn_exp, 1, lbl_r_ext)

        # obtain transition score from the transition vector for each label
        # in batch and time-step (except the first ones)
        # [bsz, seq_len + 1, 1]
        lbl_l_ext = tags_ext[:, :-1].unsqueeze(dim=-1)
        trn_scr = torch.gather(trn_row, 2, lbl_l_ext)
        trn_scr = trn_scr.squeeze(-1)

        mask_ext = mask.new_full((bsz, seq_len + 1), fill_value=True, dtype=torch.bool)
        mask_ext[:, 1:] = mask
        trn_scr = trn_scr * mask_ext
        transition_scores = trn_scr.sum(1).squeeze(-1)

        return emission_scores + transition_scores

    def neg_log_likelihood_loss(self, logits, mask, tags):
        """
         mini-batch negative log likelihood loss for sequences,
         :param logits: [bsz, seq_len, num_tags]
         :param mask: [bsz, seq_len]
         :param tags: [bsz, seq_len]
         :return: [bsz]
         """
        if not self.include_start_end:
            logits = self._pad_start_end(logits)

        # [bsz]
        real_path_score = self._real_path_score(logits, mask, tags)
        # [bsz]
        all_path_score = self._all_paths_score(logits, mask)

        return torch.mean(all_path_score - real_path_score)

    @staticmethod
    def _pad_start_end(logits):
        """
        padding <start> and <end> scores to input logits
        :param logits: [bsz, seq_len, n_tags],
                        n_tags is the number of tags without <start> and <end>
        :return:
        """
        bsz, seq_len, n_tags = logits.size()
        start_end_logits = logits.new_full((bsz, seq_len, 2), fill_value=0.)
        return torch.cat([logits, start_end_logits], dim=-1)

    def forward(self, logits):
        """
        CRF forward DO NOT change anything
        :param logits: [bsz, seq_len, num_tags]
        :return: [bsz]
        """
        seq_len, bsz, num_tags = logits.size()
        assert self.num_tags_ext == num_tags if self.include_start_end else num_tags + 2

        return logits
