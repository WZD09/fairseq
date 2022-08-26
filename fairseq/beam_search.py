from fairseq.sequence_generator import SequenceGenerator

import math
import sys, os
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from fairseq import search, utils
from fairseq.data import data_utils
from fairseq.models import FairseqIncrementalDecoder
from fairseq.ngram_repeat_block import NGramRepeatBlock

import logging

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("beam search")


class MySequenceGenerator(SequenceGenerator):
    def __init__(
            self,
            models,
            tgt_dict,
            beam_size=1,
            max_len_a=0,
            max_len_b=200,
            max_len=0,
            min_len=1,
            normalize_scores=True,
            len_penalty=1.0,
            unk_penalty=0.0,
            temperature=1.0,
            match_source_len=False,
            no_repeat_ngram_size=0,
            search_strategy=None,
            eos=None,
            symbols_to_strip_from_output=None,
            lm_model=None,
            lm_weight=1.0,
            tokens_to_suppress=(),
    ):
        super(MySequenceGenerator, self).__init__(
            models,
            tgt_dict,
            beam_size,
            max_len_a,
            max_len_b,
            max_len,
            min_len,
            normalize_scores,
            len_penalty,
            unk_penalty,
            temperature,
            match_source_len,
            no_repeat_ngram_size,
            search_strategy,
            eos,
            symbols_to_strip_from_output,
            lm_model,
            lm_weight,
            tokens_to_suppress,
        )
        print("*********************************** my beam search! ******************************")

    def _generate(
            self,
            sample: Dict[str, Dict[str, Tensor]],
            prefix_tokens: Optional[Tensor] = None,
            constraints: Optional[Tensor] = None,
            bos_token: Optional[int] = None,
    ):
        print("*********************************** my beam search _generate! ********************")

        # 从sample中设置encoder的输入
        pad = self.pad
        eos = self.eos
        encoder_input = sample["net_input"]
        src_tokens = encoder_input["src_tokens"]
        src_lengths = (
            (src_tokens.ne(eos) & src_tokens.ne(pad))  # EOS Pad的位置标记为False
                .long()
                .sum(dim=1)  # 按行算True的token有多少
        )
        bsz, src_len = src_tokens.size()[:2]  # 第一维batch dimension，第二维src_len 包括EOS pad
        beam = self.beam_size
        cand_size = 2 * beam  # 2 x beam size 保证如果有EOS还至少还剩beam个cand用于下次搜索


        max_len = min(
            int(self.max_len_a * src_len + self.max_len_b),  # ax + b
            self.max_len - 1,
        )
        # 初始化score为0和token为pad( 1 )（开头放bos 如果没有bos放eos)
        scores = torch.zeros(bsz * beam, max_len + 1).to(src_tokens).float()  # len+1是为了输出最后一个EOS
        tokens = torch.zeros(bsz * beam, max_len + 2).to(src_tokens).long().fill_(pad)
        tokens[:, 0] = bos_token if bos_token else eos
        attn: Optional[Tensor] = None
        # 初始化一些用于标记是否完成的变量
        # 表示需要ignore的candidate，因为有时候我们要sample5个但是已经找到了2个，就标记两个位置然后只完成剩下的三个采样
        cands_to_ignore = torch.zeros(bsz, beam).to(src_tokens).eq(-1)  # 初始都是false
        finalized = [[] for i in range(bsz)]  # 最终返回的结构
        finished = [False for i in range(bsz)]  # 这个句子所有的beam都被找到了就设为True 初始都是false
        num_remaining_sent = bsz  # number of sentences remaining

        # batch,beam 个句子压平到batch*beam，需要一些偏移数组 可以重新取出第几个句子，纪录每个sent的第一个beam压平以后的位置
        # eg. 0,2,4.......2*bsz (if beam=2)
        beam_bsz_offsets = (torch.arange(bsz) * beam).unsqueeze(1).long().to(src_tokens.device)
        # cand_offsets [0, 1, 2, 3]
        cand_offsets = torch.arange(cand_size).long().to(src_tokens.device)

        reorder_state = None
        batch_idxs = None
        original_batch_idxs = torch.arange(bsz).long()
        incremental_states = [{}]

        # 编码src ，bsz行reorder 扩充成bsz*beam行的encoder_outs
        encoder_outs = self.model.forward_encoder(encoder_input)
        encoder_outs = self.model.reorder_encoder_out(
            encoder_outs,
            torch.arange(bsz).view(bsz, -1).repeat(1, beam).to(src_tokens.device).long().view(-1),
        )

        for step in range(max_len + 1):  # one extra step for EOS marker
            if reorder_state is not None:
                # 除了step 0
                if batch_idxs is not None:
                    # update beam indices to take into account removed sentences
                    corr = batch_idxs - torch.arange(batch_idxs.numel()).type_as(
                        batch_idxs
                    )
                    reorder_state.view(-1, beam).add_(
                        corr.unsqueeze(-1) * beam
                    )
                    original_batch_idxs = original_batch_idxs[batch_idxs]
                self.model.reorder_incremental_state(incremental_states, reorder_state)
                encoder_outs = self.model.reorder_encoder_out(
                    encoder_outs, reorder_state
                )
            # decoder输入以前生成的词（ 只有开始符号如果是step0）得到log prob
            raw_lprobs, avg_attn_scores = self.model.forward_decoder(
                tokens[:, : step + 1],
                encoder_outs,
                incremental_states,
                self.temperature,
            )
            if avg_attn_scores is not None:
                if attn is None:
                    attn = torch.empty(
                        bsz * beam, avg_attn_scores.size(1), max_len + 2
                    ).to(scores)
                attn[:, :, step + 1].copy_(avg_attn_scores)
            # 处理lprobs 避免生成过长过短或者pad, 并且检查Nan
            lprobs = self.reset_logprobs(step, raw_lprobs, -math.inf, max_len, self.min_len, self.unk_penalty)

            # do search
            lprobs_bb = lprobs.view(bsz, -1, self.vocab_size)
            vocab_size = lprobs_bb.size()[2]
            scores_bb = scores.view(bsz, beam, -1)[:, :, :step]

            if step == 0:
                # at the first step all hypotheses are equally likely, so use
                # only the first beam
                lprobs_bb = lprobs_bb[:, ::beam, :].contiguous()
            else:
                # make probs contain cumulative scores for each hypothesis
                lprobs_bb = lprobs_bb + scores_bb[:, :, step - 1].unsqueeze(-1)
            # cand_scores, cand_indices, cand_beams 是一个step的search的结果 【bsz, beam*2】
            # cand_score bsz行，每行是2*beam个分数（预测词log score的和）
            # cand_indices 是对应的词的index
            cand_scores, cand_indices = torch.topk(
                lprobs_bb.view(bsz, -1),
                k=min(
                    # Take the best 2 x beam predictions. We'll choose the first
                    # beam of these which don't predict eos to continue with.
                    beam * 2,
                    lprobs_bb.view(bsz, -1).size(1) - 1,  # -1 so we never select pad
                ),
            )
            # Project back into relative indices and beams
            # 除V取整数，因为刚刚把lprobs_bb按照(bsz, -1)拉平了，所有beam的V个p拼在一起找top K，所以现在整除V知道是第几个beam
            cand_beams = torch.div(cand_indices, vocab_size, rounding_mode="trunc")
            # 这个是把所有beam的V个p拼在一起找top K的index重新映射到0~V这个区间（原来每个数字可能是0~beam*V这个区间）
            cand_indices = cand_indices.fmod(vocab_size)

            # cand_beams 是每个bsz内部的beam位置 转换成 bsz*beam的索引
            cand_beam_bsz_idx = cand_beams.add(beam_bsz_offsets)

            eos_bbsz_idx = torch.empty(0).to(tokens)  # indices of hypothesis ending with eos (finished sentences)
            eos_scores = torch.empty(0).to(scores)  # scores of hypothesis ending with eos (finished sentences)
            # 找到生成了结束符的设为True
            eos_mask = cand_indices.eq(eos) & cand_scores.ne(-math.inf)
            eos_mask[:, :beam][cands_to_ignore] = torch.tensor(0).to(eos_mask)
            eos_bbsz_idx = torch.masked_select(cand_beam_bsz_idx[:, :beam], mask=eos_mask[:, :beam])

            # 如果有EOS 放到finalized
            finalized_sents: List[int] = []
            if eos_bbsz_idx.numel() > 0:
                eos_scores = torch.masked_select(
                    cand_scores[:, :beam], mask=eos_mask[:, :beam]
                )

                finalized_sents = self.finalize_hypos(
                    step,
                    eos_bbsz_idx,
                    eos_scores,
                    tokens,
                    scores,
                    finalized,
                    finished,
                    beam,
                    attn,
                    src_lengths,
                    max_len,
                )
                num_remaining_sent -= len(finalized_sents)
            if num_remaining_sent == 0:
                break
            if self.search.stop_on_max_len and step >= max_len:
                break

            # 一句话如果已经找到了beam个结束的生成好的句子，就移除它
            if len(finalized_sents) > 0:
                new_bsz = bsz - len(finalized_sents)

                # construct batch_idxs which holds indices of batches to keep for the next pass
                batch_mask = torch.ones(
                    bsz, dtype=torch.bool, device=cand_indices.device
                )
                batch_mask[finalized_sents] = False
                batch_idxs = torch.arange(
                    bsz, device=cand_indices.device
                ).masked_select(batch_mask)

                # Choose the subset of the hypothesized constraints that will continue
                self.search.prune_sentences(batch_idxs)

                eos_mask = eos_mask[batch_idxs]
                cand_beams = cand_beams[batch_idxs]
                beam_bsz_offsets.resize_(new_bsz, 1)
                cand_beam_bsz_idx = cand_beams.add(beam_bsz_offsets)
                cand_scores = cand_scores[batch_idxs]
                cand_indices = cand_indices[batch_idxs]

                src_lengths = src_lengths[batch_idxs]
                cands_to_ignore = cands_to_ignore[batch_idxs]

                scores = scores.view(bsz, -1)[batch_idxs].view(new_bsz * beam, -1)
                tokens = tokens.view(bsz, -1)[batch_idxs].view(new_bsz * beam, -1)
                if attn is not None:
                    attn = attn.view(bsz, -1)[batch_idxs].view(
                        new_bsz * beam, attn.size(1), -1
                    )
                bsz = new_bsz
            else:
                batch_idxs = None
            # 处理下一次search的输入
            # 将以eos结尾的排在后面, 取前beam个, 作为下次的hypo进行扩展
            # 有EOS的数字最大，所以等下选最小的时候 就选不到
            eos_mask[:, :beam] = ~((~cands_to_ignore) & (~eos_mask[:, :beam]))  # 不ignore的EOS的位置是True
            active_mask = torch.add(  # eos_mask true的位置+ candsize
                eos_mask.long() * cand_size,
                cand_offsets[: eos_mask.size(1)],
            )
            new_cands_to_ignore, active_hypos = torch.topk(
                active_mask, k=beam, dim=1, largest=False
            )
            cands_to_ignore = new_cands_to_ignore.ge(cand_size)[:, :beam]
            # active_bbsz_idx 很重要后面要用，表示哪个beam用来生成下一下
            active_bbsz_idx = torch.gather(cand_beam_bsz_idx, dim=1, index=active_hypos)
            active_scores = torch.gather(cand_scores, dim=1, index=active_hypos)

            # 刚刚作为输入的token重新设置 成 被选择的beam 作为下一个输入
            tokens[:, : step + 1] = torch.index_select(
                tokens[:, : step + 1], dim=0, index=active_bbsz_idx.view(-1)
            )
            # 加上刚刚生成的下一个token
            tokens.view(bsz, beam, -1)[:, :, step + 1] = torch.gather(
                cand_indices, dim=1, index=active_hypos
            )
            # 重新设置之前的score 给下次search用
            if step > 0:
                scores[:, :step] = torch.index_select(scores[:, :step], dim=0, index=active_bbsz_idx.view(-1))
            # 设置当前预测的词的score
            scores.view(bsz, beam, -1)[:, :, step] = active_scores
            self.search.update_constraints(active_hypos)
            # copy attention for active hypotheses
            if attn is not None:
                attn[:, :, : step + 2] = torch.index_select(attn[:, :, : step + 2], dim=0, index=active_bbsz_idx.view(-1))
            # end of step loop

        return self.sort_finalized_descending(finalized)


    def sort_finalized_descending(self, finalized: List[List[Dict[str, Tensor]]]):
        """ 对finalized按照score进行排序 """
        new_finalized = []
        for beam_list in finalized:
            sent_scores = [sentence["score"].item() for sentence in beam_list]
            sent_scores = torch.tensor(sent_scores)
            sorted_scores, sorted_indeces = torch.sort(sent_scores, descending=True)
            new_beam_list = [beam_list[i] for i in sorted_indeces]
            new_finalized.append(new_beam_list)
        return new_finalized


    def reset_logprobs(
            self,
            step: int,
            lprobs: Tensor,
            reset_value: float,
            max_len: int,
            min_len: int,
            unk_penalty: float,
    ):
        """
        对lprobs处理后返回，控制生成EOS pad unk的log prob
        并且检查Nan
        """
        lprobs[:, self.pad] = reset_value  # never select pad 如果预测的词是pad就不输出

        if unk_penalty:
            logger.info(unk_penalty)

        lprobs[:, self.unk] -= unk_penalty  # apply unk penalty
        if step >= max_len:
            lprobs[:, : self.eos] = reset_value
            lprobs[:, self.eos + 1:] = reset_value
        if step < min_len:
            lprobs[:, self.eos] = reset_value

        lprobs[torch.isnan(lprobs)] = torch.tensor(-math.inf).to(lprobs)

        return lprobs
