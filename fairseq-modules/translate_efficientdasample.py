import os, fnmatch, math
import torch
from torch import Tensor
import itertools
import fairseq.tasks.translation
from fairseq.tasks import FairseqTask, register_task,LegacyFairseqTask
from fairseq import metrics, options, utils
from torch import nn
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import sys

from fairseq.data import (
    FairseqDataset,
    AppendTokenDataset,
    ConcatDataset,
    LanguagePairDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
    data_utils,
    encoders,
    indexed_dataset,
)

import numpy as np
import logging
import random
from torch.utils.tensorboard import SummaryWriter

from fairseq.models.fairseq_encoder import EncoderOut

logger = logging.getLogger(__name__)


# THIS IS  A COPY FROM ANOTHER FILE, BE CAREFUL AND TRY TO AVOID THE COPY IN THE FUTURE
class TransformerPerturbationSequenceScorer(object):
    """Scores the target for a given source sentence and perturbates embedings as
    assuming that the model is an instance of TransformerModel"""
    class PerturbationType(Enum):
        NONE = 0
        SOURCE = 1
        TARGET = 2

    def __init__(
        self,
        perturbationType,
        src_dict,
        tgt_dict,
        softmax_batch=None,
        compute_alignment=False,
        eos=None,
        symbols_to_strip_from_output=None,
    ):
        self.src_dict=src_dict
        self.perturbationType=self.PerturbationType[perturbationType]
        self.pad = tgt_dict.pad()
        self.eos = tgt_dict.eos() if eos is None else eos
        self.softmax_batch = softmax_batch or sys.maxsize
        assert self.softmax_batch > 0
        self.compute_alignment = compute_alignment
        self.symbols_to_strip_from_output = (
            symbols_to_strip_from_output.union({self.eos})
            if symbols_to_strip_from_output is not None
            else {self.eos}
        )
        self.lambda_norm=0.01

    #Copied from fairseq Transformer
    def transformer_forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):

        
        encoder_out = self.transformer_encoder_forward(
            src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens
        )
        decoder_out = self.transformer_decoder_forward(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        return decoder_out

    def transformer_decoder_forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
    ):
        x, extra = self.transformer_decoder_extract_features_scriptable(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )
        if not features_only:
            x = self.curmodel.decoder.output_layer(x)
        return x, extra

    def transformer_decoder_extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        if alignment_layer is None:
            alignment_layer = self.curmodel.decoder.num_layers - 1

        # embed positions
        positions = (
            self.curmodel.decoder.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )
            if self.curmodel.decoder.embed_positions is not None
            else None
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        if self.perturbationType== self.PerturbationType.TARGET:
            pass

        # embed tokens and positions
        token_embedding=self.curmodel.decoder.embed_tokens(prev_output_tokens)
        
        if self.perturbationType== self.PerturbationType.TARGET:
            #Add perturbations
            stdevs=torch.linalg.norm(token_embedding,dim=-1)*self.lambda_norm
            addition=torch.randn(token_embedding.size()).to(token_embedding.device)*stdevs.unsqueeze(-1).expand(-1,-1,token_embedding.size(-1))

            token_embedding+=addition

        x = self.curmodel.decoder.embed_scale * token_embedding

        if self.curmodel.decoder.quant_noise is not None:
            x = self.curmodel.decoder.quant_noise(x)

        if self.curmodel.decoder.project_in_dim is not None:
            x = self.curmodel.decoder.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.curmodel.decoder.layernorm_embedding is not None:
            x = self.curmodel.decoder.layernorm_embedding(x)

        x = self.curmodel.decoder.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.curmodel.decoder.cross_self_attention or prev_output_tokens.eq(self.curmodel.decoder.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.curmodel.decoder.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.curmodel.decoder.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.curmodel.decoder.buffered_future_mask(x)
            else:
                self_attn_mask = None

            x, layer_attn, _ = layer(
                x,
                encoder_out.encoder_out if encoder_out is not None else None,
                encoder_out.encoder_padding_mask if encoder_out is not None else None,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.curmodel.decoder.layer_norm is not None:
            x = self.curmodel.decoder.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.curmodel.decoder.project_out_dim is not None:
            x = self.curmodel.decoder.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states}



    def transformer_encoder_forward_embedding(
        self, src_tokens, token_embedding: Optional[torch.Tensor] = None
    ):
        perturbateFirst=True
        if self.src_dict[ src_tokens[0,0]].startswith("TO_"):
            perturbateFirst=False

        # embed tokens and positions
        if token_embedding is None:
            token_embedding = self.curmodel.encoder.embed_tokens(src_tokens)
        
        if self.perturbationType== self.PerturbationType.SOURCE:
            #Add perturbations
            stdevs=torch.linalg.norm(token_embedding,dim=-1)*self.lambda_norm
            addition=torch.randn(token_embedding.size()).to(token_embedding.device)*stdevs.unsqueeze(-1).expand(-1,-1,token_embedding.size(-1))
            mask=torch.ones_like(addition)
            
            if perturbateFirst == False:
                mask[:,0,:]=0.0

            token_embedding+=(addition*mask)

        x = embed = self.curmodel.encoder.embed_scale * token_embedding
        if self.curmodel.encoder.embed_positions is not None:
            x = embed + self.curmodel.encoder.embed_positions(src_tokens)
        if self.curmodel.encoder.layernorm_embedding is not None:
            x = self.curmodel.encoder.layernorm_embedding(x)
        x = self.curmodel.encoder.dropout_module(x)
        if self.curmodel.encoder.quant_noise is not None:
            x = self.curmodel.encoder.quant_noise(x)
        return x, embed

    def transformer_encoder_forward(
        self,
        src_tokens,
        src_lengths,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
    ):

        x, encoder_embedding = self.transformer_encoder_forward_embedding(src_tokens, token_embeddings)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.curmodel.encoder.padding_idx)

        encoder_states = [] if return_all_hiddens else None

        # encoder layers
        for layer in self.curmodel.encoder.layers:
            x = layer(x, encoder_padding_mask)
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.curmodel.encoder.layer_norm is not None:
            x = self.curmodel.encoder.layer_norm(x)

        return EncoderOut(
            encoder_out=x,  # T x B x C
            encoder_padding_mask=encoder_padding_mask,  # B x T
            encoder_embedding=encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
            src_tokens=None,
            src_lengths=None,
        )

    def forward_with_perturbation(self,model, net_input):
        self.curmodel=model
        return self.transformer_forward(**net_input)

    @torch.no_grad()
    def generate(self, models, sample, **kwargs):
        """Score a batch of translations."""
        net_input = sample["net_input"]

        def batch_for_softmax(dec_out, target):
            # assumes decoder_out[0] is the only thing needed (may not be correct for future models!)
            first, rest = dec_out[0], dec_out[1:]
            bsz, tsz, dim = first.shape
            if bsz * tsz < self.softmax_batch:
                yield dec_out, target, True
            else:
                flat = first.contiguous().view(1, -1, dim)
                flat_tgt = target.contiguous().view(flat.shape[:-1])
                s = 0
                while s < flat.size(1):
                    e = s + self.softmax_batch
                    yield (flat[:, s:e],) + rest, flat_tgt[:, s:e], False
                    s = e

        def gather_target_probs(probs, target):
            probs = probs.gather(
                dim=2,
                index=target.unsqueeze(-1),
            )
            return probs

        orig_target = sample["target"]

        # compute scores for each model in the ensemble
        avg_probs = None
        avg_attn = None
        for model in models:
            model.eval()

            decoder_out = self.forward_with_perturbation(model, net_input)

            attn = decoder_out[1] if len(decoder_out) > 1 else None
            if type(attn) is dict:
                attn = attn.get("attn", None)

            batched = batch_for_softmax(decoder_out, orig_target)
            probs, idx = None, 0
            for bd, tgt, is_single in batched:
                sample["target"] = tgt
                curr_prob = model.get_normalized_probs(
                    bd, log_probs=len(models) == 1, sample=sample
                ).data
                if is_single:
                    probs = gather_target_probs(curr_prob, orig_target)
                else:
                    if probs is None:
                        probs = curr_prob.new(orig_target.numel())
                    step = curr_prob.size(0) * curr_prob.size(1)
                    end = step + idx
                    tgt_probs = gather_target_probs(
                        curr_prob.view(tgt.shape + (curr_prob.size(-1),)), tgt
                    )
                    probs[idx:end] = tgt_probs.view(-1)
                    idx = end
                sample["target"] = orig_target

            probs = probs.view(sample["target"].shape)

            if avg_probs is None:
                avg_probs = probs
            else:
                avg_probs.add_(probs)
            if attn is not None:
                if torch.is_tensor(attn):
                    attn = attn.data
                else:
                    attn = attn[0]
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)
        if len(models) > 1:
            avg_probs.div_(len(models))
            avg_probs.log_()
            if avg_attn is not None:
                avg_attn.div_(len(models))

        bsz = avg_probs.size(0)
        hypos = []
        start_idxs = sample["start_indices"] if "start_indices" in sample else [0] * bsz
        for i in range(bsz):
            # remove padding from ref
            ref = (
                utils.strip_pad(sample["target"][i, start_idxs[i] :], self.pad)
                if sample["target"] is not None
                else None
            )
            tgt_len = ref.numel()
            avg_probs_i = avg_probs[i][start_idxs[i] : start_idxs[i] + tgt_len]
            score_i = avg_probs_i.sum() / tgt_len
            if avg_attn is not None:
                avg_attn_i = avg_attn[i]
                if self.compute_alignment:
                    alignment = utils.extract_hard_alignment(
                        avg_attn_i,
                        sample["net_input"]["src_tokens"][i],
                        sample["target"][i],
                        self.pad,
                        self.eos,
                    )
                else:
                    alignment = None
            else:
                avg_attn_i = alignment = None
            hypos.append(
                [
                    {
                        "tokens": ref,
                        "score": score_i,
                        "attention": avg_attn_i,
                        "alignment": alignment,
                        "positional_scores": avg_probs_i,
                    }
                ]
            )
        return hypos


def collate(
    samples,
    pad_idx,
    eos_idx,
    bos_idx,
    left_pad_source=True,
    left_pad_target=False,
    input_feeding=True,
    pad_to_length=None,
    pad_to_multiple=1,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False, pad_to_length=None):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx,
            left_pad,
            move_eos_to_beginning,
            pad_to_length=pad_to_length,
            pad_to_multiple=pad_to_multiple,
        )

    def check_alignment(alignment, src_len, tgt_len):
        if alignment is None or len(alignment) == 0:
            return False
        if (
            alignment[:, 0].max().item() >= src_len - 1
            or alignment[:, 1].max().item() >= tgt_len - 1
        ):
            logger.warning("alignment size mismatch found, skipping alignment!")
            return False
        return True

    def compute_alignment_weights(alignments):
        """
        Given a tensor of shape [:, 2] containing the source-target indices
        corresponding to the alignments, a weight vector containing the
        inverse frequency of each target index is computed.
        For e.g. if alignments = [[5, 7], [2, 3], [1, 3], [4, 2]], then
        a tensor containing [1., 0.5, 0.5, 1] should be returned (since target
        index 3 is repeated twice)
        """
        align_tgt = alignments[:, 1]
        _, align_tgt_i, align_tgt_c = torch.unique(
            align_tgt, return_inverse=True, return_counts=True
        )
        align_weights = align_tgt_c[align_tgt_i[np.arange(len(align_tgt))]]
        return 1.0 / align_weights.float()

    id = torch.LongTensor([s["id"] for s in samples])
    src_tokens = merge(
        "source",
        left_pad=left_pad_source,
        pad_to_length=pad_to_length["source"] if pad_to_length is not None else None,
    )
    # sort by descending source length
    src_lengths = torch.LongTensor(
        [s["source"].ne(pad_idx).long().sum() for s in samples]
    )
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    if samples[0].get("target", None) is not None:
        target = merge(
            "target",
            left_pad=left_pad_target,
            pad_to_length=pad_to_length["target"]
            if pad_to_length is not None
            else None,
        )
        target = target.index_select(0, sort_order)
        tgt_lengths = torch.LongTensor(
            [s["target"].ne(pad_idx).long().sum() for s in samples]
        ).index_select(0, sort_order)
        ntokens = tgt_lengths.sum().item()

        if samples[0].get("prev_output_tokens", None) is not None:
            prev_output_tokens = merge("prev_output_tokens", left_pad=left_pad_target)
        elif input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                "target",
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
                pad_to_length=pad_to_length["target"]
                if pad_to_length is not None
                else None,
            )

    else:
        ntokens = src_lengths.sum().item()

    batch = {
        "id": id,
        "nsentences": len(samples),
        "ntokens": ntokens,
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
        },
        "target": target,
    }
    if prev_output_tokens is not None:
        batch["net_input"]["prev_output_tokens"] = prev_output_tokens.index_select(
            0, sort_order
        )

    #Create auxiliary versions of source and target
    for k in samples[0]:
        if k.startswith("source_aux_"):
            auxname=k[len("source_aux_"):]
            src_tokens_aux = merge(
                k,
                left_pad=left_pad_source,
                pad_to_length=pad_to_length[k] if pad_to_length is not None else None,
            )
            # sort by descending source length
            src_lengths_aux = torch.LongTensor(
                [s[k].ne(pad_idx).long().sum() for s in samples]
            )

            src_lengths_aux= src_lengths_aux.index_select(0, sort_order)
            src_tokens_aux = src_tokens_aux.index_select(0, sort_order)

            batch["net_input"]["src_lengths_aux_"+auxname]=src_lengths_aux
            batch["net_input"]["src_tokens_aux_"+auxname]=src_tokens_aux


        if k.startswith("target_aux_"):
            targetaux = merge(
                k,
                left_pad=left_pad_target,
                pad_to_length=pad_to_length[k]
                if pad_to_length is not None
                else None,
            )
            targetaux = targetaux.index_select(0, sort_order)
            batch[k]=targetaux
            if input_feeding:
                # we create a shifted version of targets for feeding the
                # previous output token(s) into the next decoder step
                prev_output_tokens_aux = merge(
                    k,
                    left_pad=left_pad_target,
                    move_eos_to_beginning=True,
                    pad_to_length=pad_to_length[k]
                    if pad_to_length is not None
                    else None,
                )
                prev_output_tokens_aux = prev_output_tokens_aux.index_select(0, sort_order)
                batch["net_input"]["prev_output_tokens_aux_{}".format(k[len("target_aux_"):])]=prev_output_tokens_aux

    if samples[0].get("alignment", None) is not None:
        #No modification over alignments
        batch["alignments"] = [
            samples[align_idx]["alignment"] for align_idx in sort_order
        ]

    if samples[0].get("constraints", None) is not None:
        # Collate the packed constraints across the samples, padding to
        # the length of the longest sample.
        lens = [sample.get("constraints").size(0) for sample in samples]
        max_len = max(lens)
        constraints = torch.zeros((len(samples), max(lens))).long()
        for i, sample in enumerate(samples):
            constraints[i, 0 : lens[i]] = samples[i].get("constraints")
        batch["constraints"] = constraints

    return batch


class LanguagePairDatasetDA(FairseqDataset):
    """
    A pair of torch.utils.data.Datasets.
    Args:
        src (torch.utils.data.Dataset): source dataset to wrap
        src_sizes (List[int]): source sentence lengths
        src_dict (~fairseq.data.Dictionary): source vocabulary
        tgt (torch.utils.data.Dataset, optional): target dataset to wrap
        tgt_sizes (List[int], optional): target sentence lengths
        tgt_dict (~fairseq.data.Dictionary, optional): target vocabulary
        left_pad_source (bool, optional): pad source tensors on the left side
            (default: True).
        left_pad_target (bool, optional): pad target tensors on the left side
            (default: False).
        shuffle (bool, optional): shuffle dataset elements before batching
            (default: True).
        input_feeding (bool, optional): create a shifted version of the targets
            to be passed into the model for teacher forcing (default: True).
        remove_eos_from_source (bool, optional): if set, removes eos from end
            of source if it's present (default: False).
        append_eos_to_target (bool, optional): if set, appends eos to end of
            target if it's absent (default: False).
        align_dataset (torch.utils.data.Dataset, optional): dataset
            containing alignments.
        constraints (Tensor, optional): 2d tensor with a concatenated, zero-
            delimited list of constraints for each sentence.
        append_bos (bool, optional): if set, appends bos to the beginning of
            source/target sentence.
        num_buckets (int, optional): if set to a value greater than 0, then
            batches will be bucketed into the given number of batch shapes.
        src_lang_id (int, optional): source language ID, if set, the collated batch
            will contain a field 'src_lang_id' in 'net_input' which indicates the
            source language of the samples.
        tgt_lang_id (int, optional): target language ID, if set, the collated batch
            will contain a field 'tgt_lang_id' which indicates the target language
             of the samples.
    """

    def __init__(
        self,
        src,
        src_sizes,
        src_dict,
        tgt=None,
        tgt_sizes=None,
        tgt_dict=None,
        left_pad_source=True,
        left_pad_target=False,
        shuffle=True,
        input_feeding=True,
        remove_eos_from_source=False,
        append_eos_to_target=False,
        align_dataset=None,
        constraints=None,
        append_bos=False,
        eos=None,
        num_buckets=0,
        src_lang_id=None,
        tgt_lang_id=None,
        pad_to_multiple=1,
        src_aux={},
        tgt_aux={},
    ):
        if tgt_dict is not None:
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.eos() == tgt_dict.eos()
            assert src_dict.unk() == tgt_dict.unk()
        if tgt is not None:
            assert len(src) == len(
                tgt
            ), "Source and target must contain the same number of examples"
        self.src = src
        self.tgt = tgt
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.sizes = (
            np.vstack((self.src_sizes, self.tgt_sizes)).T
            if self.tgt_sizes is not None
            else self.src_sizes
        )
        self.src_aux=src_aux
        self.tgt_aux=tgt_aux

        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.remove_eos_from_source = remove_eos_from_source
        self.append_eos_to_target = append_eos_to_target
        self.align_dataset = align_dataset
        if self.align_dataset is not None:
            assert (
                self.tgt_sizes is not None
            ), "Both source and target needed when alignments are provided"
        self.constraints = constraints
        self.append_bos = append_bos
        self.eos = eos if eos is not None else src_dict.eos()
        self.src_lang_id = src_lang_id
        self.tgt_lang_id = tgt_lang_id

        if num_buckets > 0:
            from fairseq.data import BucketPadLengthDataset

            self.src = BucketPadLengthDataset(
                self.src,
                sizes=self.src_sizes,
                num_buckets=num_buckets,
                pad_idx=self.src_dict.pad(),
                left_pad=self.left_pad_source,
            )
            self.src_sizes = self.src.sizes
            logger.info("bucketing source lengths: {}".format(list(self.src.buckets)))
            if self.tgt is not None:
                self.tgt = BucketPadLengthDataset(
                    self.tgt,
                    sizes=self.tgt_sizes,
                    num_buckets=num_buckets,
                    pad_idx=self.tgt_dict.pad(),
                    left_pad=self.left_pad_target,
                )
                self.tgt_sizes = self.tgt.sizes
                logger.info(
                    "bucketing target lengths: {}".format(list(self.tgt.buckets))
                )

            # determine bucket sizes using self.num_tokens, which will return
            # the padded lengths (thanks to BucketPadLengthDataset)
            num_tokens = np.vectorize(self.num_tokens, otypes=[np.compat.long])
            self.bucketed_num_tokens = num_tokens(np.arange(len(self.src)))
            self.buckets = [
                (None, num_tokens) for num_tokens in np.unique(self.bucketed_num_tokens)
            ]
        else:
            self.buckets = None
        self.pad_to_multiple = pad_to_multiple

    def get_batch_shapes(self):
        return self.buckets


    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None
        src_item = self.src[index]

        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        if self.append_eos_to_target:
            eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
            if self.tgt and self.tgt[index][-1] != eos:
                tgt_item = torch.cat([self.tgt[index], torch.LongTensor([eos])])


        if self.append_bos:
            bos = self.tgt_dict.bos() if self.tgt_dict else self.src_dict.bos()
            if self.tgt and self.tgt[index][0] != bos:
                tgt_item = torch.cat([torch.LongTensor([bos]), self.tgt[index]])

            bos = self.src_dict.bos()
            if self.src[index][0] != bos:
                src_item = torch.cat([torch.LongTensor([bos]), self.src[index]])

        src_aux_items={}
        bos = self.src_dict.bos()
        for k in self.src_aux:
            src_aux_items[k]= self.src_aux[k][index]
            if self.append_eos_to_target:
                eos = self.src_dict.eos()
                if src_aux_items[k][-1] != eos:
                    src_aux_items[k]=torch.cat([src_aux_items[k], torch.LongTensor([eos])])
            if self.append_bos:
                if src_aux_items[k][0] != bos:
                    src_aux_items[k]=torch.cat([torch.LongTensor([bos]), src_aux_items[k]])

        tgt_aux_items={}
        bos = self.tgt_dict.bos() if self.tgt_dict else self.src_dict.bos()
        for k in self.tgt_aux:
            tgt_aux_items[k]= self.tgt_aux[k][index]
            if self.append_eos_to_target:
                eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
                if tgt_aux_items[k][-1] != eos:
                    tgt_aux_items[k]=torch.cat([tgt_aux_items[k], torch.LongTensor([eos])])
            if self.append_bos:
                if tgt_aux_items[k][0] != bos:
                    tgt_aux_items[k]=torch.cat([torch.LongTensor([bos]), tgt_aux_items[k]])

        if self.remove_eos_from_source:
            eos = self.src_dict.eos()
            if self.src[index][-1] == eos:
                src_item = self.src[index][:-1]
            for k in src_aux_items:
                if src_aux_items[k][-1] == eos:
                    src_aux_items[k]=src_aux_items[k][:-1]

        example = {
            "id": index,
            "source": src_item,
            "target": tgt_item,
        }
        if self.align_dataset is not None:
            example["alignment"] = self.align_dataset[index]
        if self.constraints is not None:
            example["constraints"] = self.constraints[index]

        for k,v in tgt_aux_items.items():
            example["target_aux_{}".format(k)]=v

        for k,v in src_aux_items.items():
            example["source_aux_{}".format(k)]=v

        return example

    def __len__(self):
        return len(self.src)

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.
        Args:
            samples (List[dict]): samples to collate
            pad_to_length (dict, optional): a dictionary of
                {'source': source_pad_to_length, 'target': target_pad_to_length}
                to indicate the max length to pad to in source and target respectively.
        Returns:
            dict: a mini-batch with the following keys:
                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:
                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one
                    position for teacher forcing, of shape `(bsz, tgt_len)`.
                    This key will not be present if *input_feeding* is
                    ``False``.  Padding will appear on the left if
                    *left_pad_target* is ``True``.
                  - `src_lang_id` (LongTensor): a long Tensor which contains source
                    language IDs of each sample in the batch
                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
                - `tgt_lang_id` (LongTensor): a long Tensor which contains target language
                   IDs of each sample in the batch
        """
        res = collate(
            samples,
            pad_idx=self.src_dict.pad(),
            eos_idx=self.eos,
            bos_idx=self.src_dict.bos(),
            left_pad_source=self.left_pad_source,
            left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
            pad_to_length=pad_to_length,
            pad_to_multiple=self.pad_to_multiple,
        )
        if self.src_lang_id is not None or self.tgt_lang_id is not None:
            src_tokens = res["net_input"]["src_tokens"]
            bsz = src_tokens.size(0)
            if self.src_lang_id is not None:
                res["net_input"]["src_lang_id"] = (
                    torch.LongTensor([[self.src_lang_id]]).expand(bsz, 1).to(src_tokens)
                )
            if self.tgt_lang_id is not None:
                res["tgt_lang_id"] = (
                    torch.LongTensor([[self.tgt_lang_id]]).expand(bsz, 1).to(src_tokens)
                )
        return res

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(
            self.src_sizes[index],
            self.tgt_sizes[index] if self.tgt_sizes is not None else 0,
        )

    def num_tokens_vec(self, indices):
        """Return the number of tokens for a set of positions defined by indices.
        This value is used to enforce ``--max-tokens`` during batching."""
        sizes = self.src_sizes[indices]
        if self.tgt_sizes is not None:
            sizes = np.maximum(sizes, self.tgt_sizes[indices])
        return sizes

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (
            self.src_sizes[index],
            self.tgt_sizes[index] if self.tgt_sizes is not None else 0,
        )

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self)).astype(np.int64)
        else:
            indices = np.arange(len(self), dtype=np.int64)
        if self.buckets is None:
            # sort by target length, then source length
            if self.tgt_sizes is not None:
                indices = indices[np.argsort(self.tgt_sizes[indices], kind="mergesort")]
            return indices[np.argsort(self.src_sizes[indices], kind="mergesort")]
        else:
            # sort by bucketed_num_tokens, which is:
            #   max(padded_src_len, padded_tgt_len)
            return indices[
                np.argsort(self.bucketed_num_tokens[indices], kind="mergesort")
            ]

    @property
    def supports_prefetch(self):
        return getattr(self.src, "supports_prefetch", False) and (
            getattr(self.tgt, "supports_prefetch", False) or self.tgt is None
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)
        if self.tgt is not None:
            self.tgt.prefetch(indices)
        if self.align_dataset is not None:
            self.align_dataset.prefetch(indices)

    def filter_indices_by_size(self, indices, max_sizes):
        """Filter a list of sample indices. Remove those that are longer
            than specified in max_sizes.
        Args:
            indices (np.array): original array of sample indices
            max_sizes (int or list[int] or tuple[int]): max sample size,
                can be defined separately for src and tgt (then list or tuple)
        Returns:
            np.array: filtered sample array
            list: list of removed indices
        """
        return data_utils.filter_paired_dataset_indices_by_size(
            self.src_sizes, self.tgt_sizes, indices, max_sizes,
        )


def load_langpair_dataset(
data_path,
split,
src,
src_dict,
tgt,
tgt_dict,
combine,
dataset_impl,
upsample_primary,
left_pad_source,
left_pad_target,
max_source_positions,
max_target_positions,
prepend_bos=False,
load_alignments=False,
truncate_source=False,
append_source_id=False,
num_buckets=0,
shuffle=True,
pad_to_multiple=1,
prepend_bos_src=None,
):
    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, "{}.{}-{}.{}".format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []
    src_aux_datasets={}
    tgt_aux_datasets={}

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else "")

        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError(
                    "Dataset not found: {} ({})".format(split, data_path)
                )

        src_dataset = data_utils.load_indexed_dataset(
            prefix + src, src_dict, dataset_impl
        )
        if truncate_source:
            src_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(src_dataset, src_dict.eos()),
                    max_source_positions - 1,
                ),
                src_dict.eos(),
            )
        src_datasets.append(src_dataset)

        tgt_dataset = data_utils.load_indexed_dataset(
            prefix + tgt, tgt_dict, dataset_impl
        )
        if tgt_dataset is not None:
            tgt_datasets.append(tgt_dataset)

        #Find auxiliary datasets (if split is train)
        if split == "train":
            #Find .bin files named train_aux_[task]
            files=fnmatch.filter(os.listdir(data_path), "{}_aux_*.{}-{}.{}.bin".format(split_k,src,tgt,tgt)  )
            for file in files:
                split_aux=file.split(".")[0]
                aux_task=split_aux.split("_")[2]

                aux_dataset=data_utils.load_indexed_dataset( os.path.join(data_path,"{}.{}-{}.{}".format(split_aux,src,tgt,src))  ,src_dict, dataset_impl)
                if aux_task not in src_aux_datasets:
                    src_aux_datasets[aux_task]=[]
                src_aux_datasets[aux_task].append(aux_dataset)

                aux_dataset=data_utils.load_indexed_dataset( os.path.join(data_path,"{}.{}-{}.{}".format(split_aux,src,tgt,tgt))  ,tgt_dict, dataset_impl)
                if aux_task not in tgt_aux_datasets:
                    tgt_aux_datasets[aux_task]=[]
                tgt_aux_datasets[aux_task].append(aux_dataset)

        logger.info(
            "{} {} {}-{} {} examples".format(
                data_path, split_k, src, tgt, len(src_datasets[-1])
            )
        )

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets) or len(tgt_datasets) == 0

    src_aux_dataset={}
    tgt_aux_dataset={}
    if len(src_datasets) == 1:
        src_dataset = src_datasets[0]
        tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
        for k in src_aux_datasets:
            src_aux_dataset[k]=src_aux_datasets[k][0]
        for k in tgt_aux_datasets:
            tgt_aux_dataset[k]=tgt_aux_datasets[k][0]

    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        if len(tgt_datasets) > 0:
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
            for k in src_aux_datasets:
                src_aux_dataset[k]=ConcatDataset(src_aux_datasets[k], sample_ratios)
            for k in tgt_aux_datasets:
                tgt_aux_dataset[k]=ConcatDataset(tgt_aux_datasets[k], sample_ratios)
        else:
            tgt_dataset = None

    if prepend_bos:
        assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        for k in src_aux_dataset:
            src_aux_dataset[k]=PrependTokenDataset(src_aux_dataset[k], src_dict.bos())
        if tgt_dataset is not None:
            tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())
            for k in tgt_aux_dataset:
                tgt_aux_dataset[k]=PrependTokenDataset(tgt_aux_dataset[k], tgt_dict.bos())
    elif prepend_bos_src is not None:
        logger.info(f"prepending src bos: {prepend_bos_src}")
        src_dataset = PrependTokenDataset(src_dataset, prepend_bos_src)
        for k in src_aux_dataset:
            src_aux_dataset[k]=PrependTokenDataset(src_aux_dataset[k], prepend_bos_src)


    eos = None
    if append_source_id:
        src_dataset = AppendTokenDataset(
            src_dataset, src_dict.index("[{}]".format(src))
        )
        for k in src_aux_dataset:
            src_aux_dataset[k]=AppendTokenDataset(src_aux_dataset[k], src_dict.index("[{}]".format(src)) )
        if tgt_dataset is not None:
            tgt_dataset = AppendTokenDataset(
                tgt_dataset, tgt_dict.index("[{}]".format(tgt))
            )
            for k in tgt_aux_dataset:
                tgt_aux_dataset[k]=AppendTokenDataset(tgt_aux_dataset[k],tgt_dict.index("[{}]".format(tgt)))
        eos = tgt_dict.index("[{}]".format(tgt))

    align_dataset = None
    if load_alignments:
        align_path = os.path.join(data_path, "{}.align.{}-{}".format(split, src, tgt))
        if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
            align_dataset = data_utils.load_indexed_dataset(
                align_path, None, dataset_impl
            )

    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None
    return LanguagePairDatasetDA(
        src_dataset,
        src_dataset.sizes,
        src_dict,
        tgt_dataset,
        tgt_dataset_sizes,
        tgt_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        align_dataset=align_dataset,
        eos=eos,
        num_buckets=num_buckets,
        shuffle=shuffle,
        pad_to_multiple=pad_to_multiple,
        src_aux=src_aux_dataset,
        tgt_aux=tgt_aux_dataset,
    )

#Code extracted from: https://rlcurriculum.github.io/#appendix
class TeacherExp3(object):
  """Teacher with Exponential-weight algorithm for Exploration and Exploitation.
  """

  def __init__(self, tasks, gamma=0.25):
    self._tasks = tasks
    self._n_tasks = len(self._tasks)
    self._gamma = gamma
    self._log_weights = np.zeros(self._n_tasks)


  @property
  def task_probabilities(self):
    weights = np.exp(self._log_weights - np.sum(self._log_weights))
    probs = ((1 - self._gamma)*weights / np.sum(weights) +
        self._gamma/self._n_tasks)
    return probs


  def get_task(self):
    """Samples a task, according to current Exp3 belief.
    """
    task_i = np.random.choice(self._n_tasks, p=self.task_probabilities)
    return self._tasks[task_i]


  def update(self, task, reward):
    """ Updates the weight of task given current reward observed
    """
    task_i = self._tasks.index(task)
    reward_corrected = reward/self.task_probabilities[task_i]
    self._log_weights[task_i] += self._gamma*reward_corrected/self._n_tasks

'''
Translation with auxiliary tasks included in the original dataset
There is one dataset per epoch
'''
@register_task("translation_efficientsample")
class TranslationEfficientSample(fairseq.tasks.translation.TranslationTask):
    def add_args(parser):
        fairseq.tasks.translation.TranslationTask.add_args(parser)
        parser.add_argument('--tasks', nargs='+', default=None, type=str,help='List of task names. Supported: base, rev, shift, replace, source, tokendropout')
        parser.add_argument('--weights', nargs='+', default=None, type=float,help='Sampling probabilities of tasks.')
        parser.add_argument('--assume-reduced-batch', action='store_true',help='Asume batch size has been reduced to B/n, being n the number of tasks')
        parser.add_argument('--multiple-backward',action='store_true',help='Perform multiple backward passes, one for each task. Saves memory.')
        parser.add_argument('--exp3',action='store_true',help='Exp3 algorithm based on dev set pg to choose between tasks')
        parser.add_argument('--exp3-reward',default='pg',help='Exp3 reward: pg, pgnorm, dummy')
        parser.add_argument('--exp3-exploration-rate',type=float,default=0.25,help='Exp3 exploration rate')
        parser.add_argument('--max-epochs-efficient',default=10,type=int,help='Number of precomputed epochs')
        parser.add_argument('--log-gradients',action='store_true',help='Log gradient cosine similarity between tasks')
        parser.add_argument('--zero-negative-gradients',action='store_true',help='Set to zero gradients of auxiliary tasks if cosine to main task is negative')
        parser.add_argument('--surgery-negative-gradients',action='store_true',help='Re-project (surgery) if cosine to main task is negative')
        parser.add_argument('--vaccine-gradients',action='store_true',help='Re-project (vaccine) using EMA')
        parser.add_argument('--ema-decay-rate',type=float,default=0.01,help='Beta parameter for EMA')
        parser.add_argument('--write-tensorboard',help='Directory where tensorboard data will be stored')
        parser.add_argument('--split-batch',type=int, default=1)
        parser.add_argument('--aux-task-ignore-gradients-sentences-starting',type=str, default='')
        parser.add_argument('--remove-backtranslation-mark',action='store_true')

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args,src_dict,tgt_dict)
        self.tasks=args.tasks
        self.weights=args.weights
        self.assume_reduced=args.assume_reduced_batch
        self.multiple_backward=args.multiple_backward
        self.summary_writer=None
        self.last_train_update_num=None

        self.cos= nn.CosineSimilarity(dim=0)
        if args.write_tensorboard is not None:
            self.summary_writer = SummaryWriter(log_dir=args.write_tensorboard)

        if self.args.exp3:
            self.exp3=TeacherExp3(self.tasks, gamma=args.exp3_exploration_rate)
            self.prev_dev_loss=None
            self.prev_task=None
            self.load_dataset("valid")
            self.dev_batch_iterator=self.get_batch_iterator(
                dataset=self.dataset("valid"),
                max_tokens=4000 #TODO: change this
                )
            self.dev_iterator=self.dev_batch_iterator.next_epoch_itr()

        #task->paragroup->value
        self.ema={}
        for t in self.tasks:
            self.ema[t]={}
        
        #keys: sample_id
        #value: list of lists
        #     each list of "value" corresponds to a run
        #           each sublist corresponds to the sequence of word probabilities of the run.
        #               Each element is a tuple (word, probability)
        self.perturbation_data_source={}
        self.perturbation_data_target={}
        self.n_perturbations=10

        self.backtmark_ignore_gradient=None
        if self.args.aux_task_ignore_gradients_sentences_starting != '':
            self.backtmark_ignore_gradient=self.src_dict.index(self.args.aux_task_ignore_gradients_sentences_starting)


    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).
        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        args.left_pad_source = utils.eval_bool(args.left_pad_source)
        args.left_pad_target = utils.eval_bool(args.left_pad_target)

        paths = utils.split_paths(args.data)
        assert len(paths) > 0

        if args.max_epochs_efficient == 0:
            path_suffix=""
        else:
            path_suffix="1"
        

        # find language pair automatically
        if args.source_lang is None or args.target_lang is None:
            args.source_lang, args.target_lang = data_utils.infer_language_pair(
                paths[0]+path_suffix
            )
        if args.source_lang is None or args.target_lang is None:
            raise Exception(
                "Could not infer language pair, please provide it explicitly"
            )



        # load dictionaries
        src_dict = cls.load_dictionary(
            os.path.join(paths[0]+path_suffix, "dict.{}.txt".format(args.source_lang))
        )
        tgt_dict = cls.load_dictionary(
            os.path.join(paths[0]+path_suffix, "dict.{}.txt".format(args.target_lang))
        )
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        logger.info("[{}] dictionary: {} types".format(args.source_lang, len(src_dict)))
        logger.info("[{}] dictionary: {} types".format(args.target_lang, len(tgt_dict)))

        return cls(args, src_dict, tgt_dict)


    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        if split != getattr(self.args, "train_subset", None):
            # if not training data set, use the first shard for valid and test
            paths = paths[:1]
        else:
            #We are going to ignore all datapaths but first, and append epoch
            #number first path name
            paths=paths[:1]

        #data_path = paths[(epoch - 1) % len(paths)]
        data_path = paths[0]

        # infer langcode
        src, tgt = self.args.source_lang, self.args.target_lang

        # Load as many datasets as tasks
        # Append epoch number
        available_epochs= self.args.max_epochs_efficient

        if available_epochs == 0:
            epoch_num=""
        else:
            epoch_num= str(((epoch-1) % available_epochs ) + 1 ) # We start at epoch 1

        self.datasets[split] = load_langpair_dataset(
            data_path+epoch_num,
            split,
            src,
            self.src_dict,
            tgt,
            self.tgt_dict,
            combine=combine,
            dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            load_alignments=self.args.load_alignments,
            truncate_source=self.args.truncate_source,
            num_buckets=self.args.num_batch_buckets,
            shuffle=(split != "test"),
            pad_to_multiple=self.args.required_seq_len_multiple,
        )
    

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        scorer_source = TransformerPerturbationSequenceScorer('SOURCE',self.source_dictionary,self.target_dictionary, None)
        scorer_target = TransformerPerturbationSequenceScorer('TARGET',self.source_dictionary,self.target_dictionary, None)

        for scorer,perturbation_data in [scorer_source,self.perturbation_data_source],[scorer_target,self.perturbation_data_target]:
            for i in range(self.n_perturbations):
                hypos = scorer.generate([model], sample)

                for i, hypos_i in enumerate(hypos):
                    #import pdb; pdb.set_trace()
                    hypo = hypos_i[0]
                    sample_id = sample["id"][i].item()

                    tokens = hypo["tokens"]
                    tgt_len = tokens.numel()
                    pos_scores = hypo["positional_scores"].float()

                    if getattr(self.args, "add_bos_token", False):
                        assert hypo["tokens"][0].item() == self.target_dictionary.bos()
                        tokens = tokens[1:]
                        pos_scores = pos_scores[1:]

                    skipped_toks = 0

                    inf_scores = pos_scores.eq(float("inf")) | pos_scores.eq(float("-inf"))
                    if inf_scores.any():
                        pos_scores = pos_scores[(~inf_scores).nonzero()]
                    
                    word_prob = []
                    for i in range(len(tokens)):
                        w_ind = tokens[i].item()    
                        w = self.target_dictionary[w_ind]
                        word_prob.append((w, pos_scores[i].item()))
                    
                    if sample_id not in perturbation_data:
                        perturbation_data[sample_id]=[]
                    perturbation_data[sample_id].append(word_prob)


        return loss, sample_size, logging_output
    
    #This could be computed more efficiently with pytorch
    def compute_variance(self, data, logprobs=False):
        if logprobs:
            prob_func=lambda x: x
        else:
            prob_func=math.exp

        result={}
        for k in data:
            sentences=data[k]
            sentlen=len(sentences[0])
            N=len(sentences)
            averages=[0.0 for i in range(sentlen) ]
            variances=[0.0 for i in range(sentlen) ]

            for i in range(N):
                for j in range(sentlen):
                    averages[j]+=prob_func(sentences[i][j][1])
            for j in range(sentlen):
                averages[j]/=N
            
            for i in range(N):
                for j in range(sentlen):
                    variances[j]+= math.pow(prob_func(sentences[i][j][1]) - averages[j], 2  )
            for j in range(sentlen):
                variances[j]/=N

            result[k]=variances
        return result


    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

        if len(self.perturbation_data_source) > 0:
            #Compute average source influence per token
            variances_source=self.compute_variance(self.perturbation_data_source)
            variances_target=self.compute_variance(self.perturbation_data_target)

            total_toks=0
            total_source_infl=0
            for i in sorted(variances_source.keys()):
                source_influences=[ vsource/(vsource+vtarget)  for vsource,vtarget in zip(variances_source[i],variances_target[i]) ]
                total_toks+=len(source_influences)
                total_source_infl+=sum(source_influences)
            
            source_influence=total_source_infl/total_toks
            metrics.log_scalar('source_influence', source_influence)
            if self.summary_writer is not None:
                self.summary_writer.add_scalar('source_influence', source_influence, self.last_train_update_num)

        #keys: sample_id
        #value: list of lists
        #     each list of "value" corresponds to a run
        #           each sublist corresponds to the sequence of word probabilities of the run.
        #               Each element is a tuple (word, probability)
        self.perturbation_data_source={}
        self.perturbation_data_target={}
        


    def has_sharded_data(self, split):
        # Force reloading dataset after each epoch
        return True


    def remove_backtranslation_mark_column(self, tokens):
        return torch.cat( (  tokens[:,0:1], tokens[:,2:]  ), dim=1  )

    def create_sample_for_task(self,sample,task):
        if task == "base":
            src_tokens=sample["net_input"]["src_tokens"]
            src_lengths=sample["net_input"]["src_lengths"]

            if self.args.remove_backtranslation_mark:
                src_tokens=self.remove_backtranslation_mark_column(src_tokens)
                src_lengths=src_lengths - 1
            
            return { "id": sample["id"],  "nsentences": sample["nsentences"], "ntokens": sample["ntokens"], "target": sample["target"], "net_input": { "src_tokens": src_tokens , "src_lengths": src_lengths , "prev_output_tokens": sample["net_input"]["prev_output_tokens"] } }
        else:
            if task != "wrdp2":
                target=sample["target_aux_{}".format(task)]
            else:
                target=sample["target"]
            pad_idx=self.tgt_dict.pad()

            


            #pad complete rows if data is backtranslated
            if self.backtmark_ignore_gradient is not None:
                #token #0 is task token

                backt_tokens=sample["net_input"]["src_tokens"][:,1].detach()

                multiplier=torch.ones_like(target)
                addition=torch.zeros_like(target)
                
                #multiply by (1 or 0), sum (0 or pad)

                #Bool tensor
                masks= (backt_tokens == self.backtmark_ignore_gradient ).unsqueeze(-1) #(bsz, 1)

                # multiply by (1 or 0)
                #For those rows with mask = True, multiply by zero
                inverted_masks=~masks
                multiplier=multiplier*inverted_masks.int()

                #sum 0 or pad
                addition=addition+ masks.int()*pad_idx


                target=target*multiplier + addition

            #modify src_tokens and src_lengths if option is enabled
            src_tokens=sample["net_input"]["src_tokens_aux_{}".format(task)]    
            src_lengths=sample["net_input"]["src_lengths_aux_{}".format(task)]
            
            if self.args.remove_backtranslation_mark:
                src_tokens=self.remove_backtranslation_mark_column(src_tokens)
                src_lengths=src_lengths - 1

            ntokens= target.ne(pad_idx).long().sum().item()
            return { "id": sample["id"],  "nsentences": sample["nsentences"], "ntokens": ntokens, "target": target, "net_input": { "src_tokens": src_tokens, "src_lengths": src_lengths  , "prev_output_tokens": sample["net_input"]["prev_output_tokens_aux_{}".format(task)] } }

    def get_sample_split(self, sample, num_part, total_parts):
        pad_idx=self.tgt_dict.pad()
        slice_target=sample["target"][num_part::total_parts,:].contiguous()
        ntokens= slice_target.ne(pad_idx).long().sum().item()
        outdict={ "id": sample["id"],  "nsentences": slice_target.size(0), "ntokens": ntokens, "target": slice_target, "net_input": { "src_tokens": sample["net_input"]["src_tokens"][num_part::total_parts,:] , "src_lengths": sample["net_input"]["src_lengths"][num_part::total_parts] , "prev_output_tokens": sample["net_input"]["prev_output_tokens"][num_part::total_parts,:] } }
        #import pdb; pdb.set_trace()
        return outdict

    def extract_grads(self, model):
        grads={}
        for name, p in model.named_parameters():
            if name.startswith("encoder.layers.") or name.startswith("decoder.layers."):
                groupname=".".join(name.split(".")[:3])
            else:
                groupname=name
            if groupname not in grads:
                grads[groupname]=[]
            grads[groupname].append( (name, p.shape, p.grad.data.flatten().cpu() ) )
        return grads

    def add_grads(self,model, grad_dict):
        indict={}
        for groupname in grad_dict:
            for name,shape,flatvalue in grad_dict[groupname]:
                indict[name]=(shape, flatvalue)
        for name, p in model.named_parameters():
            if name in indict:
                if p.grad is None:
                    p.grad=indict[name][1].reshape(indict[name][0]).to(p.device)
                else:
                    p.grad+=indict[name][1].reshape(indict[name][0]).to(p.device)


    def train_step_gradient_modifcation(self, sample, model, criterion, optimizer, update_num, ignore_grad=False):
        losses=[]
        sample_sizes=[]

        base_index=None
        #1. Do all the forward passes
        for i,task in enumerate(self.tasks):
            loss, sample_size_loc, logging_output_loc = criterion(model, self.create_sample_for_task(sample,task))
            if ignore_grad:
                loss *= 0

            losses.append(loss)
            sample_sizes.append(sample_size_loc)

            if task == "base":
                sample_size,logging_output=sample_size_loc,logging_output_loc
                base_index=i

        grads_list=[]
        #2. Backward with main task and store gradients
        optimizer.backward(losses[base_index])

        basegrads=self.extract_grads(model)
        grads_list.append(basegrads)

        optimizer.zero_grad()

        #3. Backward for each auxiliary tasks, re-project and store
        for i,task in enumerate(self.tasks):
            if task != "base":
                optimizer.backward(losses[i])
                taskgrads=self.extract_grads(model)

                #Compute cosine distance and re-project or zero
                for group in taskgrads:
                    basegrads_flat=torch.cat( [e[2] for e in  basegrads[group]])
                    taskgrads_flat=torch.cat( [e[2] for e in  taskgrads[group]] )
                    cosine=self.cos(  basegrads_flat   , taskgrads_flat )

                    if group not in self.ema[task]:
                        self.ema[task][group]=0.0

                    if self.args.log_gradients and self.summary_writer is not None:
                        self.summary_writer.add_scalar('cosine-{}/{}'.format(task,group), cosine, update_num)


                    if self.args.surgery_negative_gradients:
                        if cosine < 0:
                            new_taskgrads_flat= taskgrads_flat - torch.dot(taskgrads_flat,basegrads_flat) * basegrads_flat / (basegrads_flat.norm()**2)
                            start=0
                            for i,elem in enumerate(taskgrads[group]):
                                elem[2][:]=new_taskgrads_flat[start:start+len(elem[2])]
                                start+=len(elem[2])
                    if self.args.zero_negative_gradients:
                        if cosine < 0:
                            for i,elem in enumerate(taskgrads[group]):
                                taskgrads[group][i]=(elem[0], elem[1], elem[2]*0)

                    if self.args.vaccine_gradients:
                        ema_t=self.ema[task][group]
                        if cosine < ema_t:
                            #apply gradient vaccine update rule
                            new_taskgrads_flat=  taskgrads_flat + ((taskgrads_flat.norm()*( ema_t*torch.sqrt(1 - cosine**2)  - cosine*torch.sqrt(1- ema_t**2)   ))/( basegrads_flat.norm()*torch.sqrt(1 - ema_t**2)) )* basegrads_flat
                            start=0
                            for i,elem in enumerate(taskgrads[group]):
                                elem[2][:]=new_taskgrads_flat[start:start+len(elem[2])]
                                start+=len(elem[2])

                        #Update EMA
                        self.ema[task][group]=ema_t*(1-self.args.ema_decay_rate) + cosine*self.args.ema_decay_rate

                        if self.args.log_gradients and self.summary_writer is not None:
                            self.summary_writer.add_scalar('ema-{}/{}'.format(task,group), self.ema[task][group], update_num)

                    if self.args.log_gradients and self.summary_writer is not None:
                        taskgrads_flat=torch.cat( [e[2] for e in  taskgrads[group]] )
                        cosine=self.cos(  basegrads_flat   , taskgrads_flat )
                        self.summary_writer.add_scalar('cosine-mod-{}/{}'.format(task,group), cosine, update_num)
                grads_list.append(taskgrads)

                optimizer.zero_grad()

        #import pdb; pdb.set_trace()
        #4. Restore gradients before calling optimizer
        for grads in grads_list:
            self.add_grads(model,grads)
        
        #Combine losses
        loss=sum(losses)
        if self.assume_reduced:
            #Not sure about this
            sample_size=sum(sample_sizes)
        
        return loss, sample_size, logging_output
    

    def train_step_gradient_modifcation_opt(self, sample, model, criterion, optimizer, update_num, ignore_grad=False):
        losses=[]
        sample_sizes=[]

        grads_list=[]

        basegrads=None
        base_index=None
        
        for i,task in enumerate(self.tasks):

            #1. Forward pass
            loss, sample_size_loc, logging_output_loc = criterion(model, self.create_sample_for_task(sample,task))
            if ignore_grad:
                loss *= 0

            losses.append(loss)
            sample_sizes.append(sample_size_loc)

            optimizer.backward(loss)

            if task == "base":
                sample_size,logging_output=sample_size_loc,logging_output_loc
                base_index=i
                basegrads=self.extract_grads(model)
                grads_list.append(basegrads)
            else:
                taskgrads=self.extract_grads(model)

                #Compute cosine distance and re-project or zero
                for group in taskgrads:
                    basegrads_flat=torch.cat( [e[2] for e in  basegrads[group]])
                    taskgrads_flat=torch.cat( [e[2] for e in  taskgrads[group]] )
                    cosine=self.cos(  basegrads_flat   , taskgrads_flat )

                    if group not in self.ema[task]:
                        self.ema[task][group]=0.0

                    if self.args.log_gradients and self.summary_writer is not None:
                        self.summary_writer.add_scalar('cosine-{}/{}'.format(task,group), cosine, update_num)

                    if self.args.surgery_negative_gradients:
                        if cosine < 0:
                            new_taskgrads_flat= taskgrads_flat - torch.dot(taskgrads_flat,basegrads_flat) * basegrads_flat / (basegrads_flat.norm()**2)
                            start=0
                            for i,elem in enumerate(taskgrads[group]):
                                elem[2][:]=new_taskgrads_flat[start:start+len(elem[2])]
                                start+=len(elem[2])
                    if self.args.zero_negative_gradients:
                        if cosine < 0:
                            for i,elem in enumerate(taskgrads[group]):
                                taskgrads[group][i]=(elem[0], elem[1], elem[2]*0)

                    if self.args.vaccine_gradients:
                        ema_t=self.ema[task][group]
                        if cosine < ema_t:
                            #apply gradient vaccine update rule
                            new_taskgrads_flat=  taskgrads_flat + ((taskgrads_flat.norm()*( ema_t*torch.sqrt(1 - cosine**2)  - cosine*torch.sqrt(1- ema_t**2)   ))/( basegrads_flat.norm()*torch.sqrt(1 - ema_t**2)) )* basegrads_flat
                            start=0
                            for i,elem in enumerate(taskgrads[group]):
                                elem[2][:]=new_taskgrads_flat[start:start+len(elem[2])]
                                start+=len(elem[2])

                        #Update EMA
                        self.ema[task][group]=ema_t*(1-self.args.ema_decay_rate) + cosine*self.args.ema_decay_rate

                        if self.args.log_gradients and self.summary_writer is not None:
                            self.summary_writer.add_scalar('ema-{}/{}'.format(task,group), self.ema[task][group], update_num)

                    if self.args.log_gradients and self.summary_writer is not None:
                        taskgrads_flat=torch.cat( [e[2] for e in  taskgrads[group]] )
                        cosine=self.cos(  basegrads_flat   , taskgrads_flat )
                        self.summary_writer.add_scalar('cosine-mod-{}/{}'.format(task,group), cosine, update_num)
                grads_list.append(taskgrads)
             
            optimizer.zero_grad()

        #4. Restore gradients before calling optimizer
        for grads in grads_list:
            self.add_grads(model,grads)
        
        #Combine losses
        loss=sum(losses)
        if self.assume_reduced:
            #Not sure about this
            sample_size=sum(sample_sizes)
        
        return loss, sample_size, logging_output

    def train_step_multiple_backward(self, sample, model, criterion, optimizer, update_num, ignore_grad=False):
        losses=[]
        sample_sizes=[]

        logging_outputs_base=[]
        sample_sizes_base=[]

        if self.args.log_gradients and self.summary_writer is not None:
            #This is very ineffient, but can help us to figure out gradients
            basegrads=None
            basegradsperlayer=None
            for i,(task,weight) in enumerate(zip(self.tasks,self.weights)):
                loss, sample_size_loc, logging_output_loc = criterion(model, self.create_sample_for_task(sample,task))
                if ignore_grad:
                    loss *= 0

                modloss=loss*weight
                if self.assume_reduced:
                    modloss=modloss*len(self.tasks)

                optimizer.backward(modloss)

                #All model parameters
                flatgrads= torch.cat([ p.grad.data.flatten() for name, p in model.named_parameters()  ])

                #Split parameters by layer
                flatgradsperlayer_pre={}
                for name, p in model.named_parameters():
                    if name.startswith("encoder.layers.") or name.startswith("decoder.layers."):
                        groupname=".".join(name.split(".")[:3])
                    else:
                        groupname=name
                    if groupname not in flatgradsperlayer_pre:
                        flatgradsperlayer_pre[groupname]=[]
                    flatgradsperlayer_pre[groupname].append(p.grad.data.flatten())
                flatgradsperlayer={}
                for groupname in flatgradsperlayer_pre:
                    flatgradsperlayer[groupname]=torch.cat(flatgradsperlayer_pre[groupname])

                if i == 0:
                    basegrads=flatgrads
                    basegradsperlayer=flatgradsperlayer
                else:
                    #If this is an auxiliary task, store cosine similarity for all parameters
                    cosine=self.cos(basegrads, flatgrads)
                    self.summary_writer.add_scalar('cosine-{}/all'.format(task), cosine, update_num)

                    #And also per group
                    for groupname in flatgradsperlayer:
                        cosine=self.cos(basegradsperlayer[groupname], flatgradsperlayer[groupname])
                        self.summary_writer.add_scalar('cosine-{}/{}'.format(task,groupname), cosine, update_num)

                #Log gradient norm
                #for name, p in model.named_parameters():
                #    param_norm = p.grad.data.norm(2) if p.grad is not None else None
                #    self.summary_writer.add_scalar('gradnorm-{}/{}'.format(task,name), param_norm, update_num)
                optimizer.zero_grad()

        for i,(task,weight) in enumerate(zip(self.tasks,self.weights)):
            tasksample=self.create_sample_for_task(sample,task)
            if self.args.split_batch > 1:
                for i in range(self.args.split_batch):
                    splitsample=self.get_sample_split(tasksample,i,self.args.split_batch)
                    with torch.autograd.profiler.record_function("forward"):
                        loss, sample_size_loc, logging_output_loc = criterion(model, splitsample)
                        if ignore_grad:
                            loss *= 0

                        modloss=loss*weight
                        if self.assume_reduced:
                            modloss=modloss*len(self.tasks)

                        losses.append(modloss)
                        sample_sizes.append(sample_size_loc)

                        if task == "base":
                            logging_outputs_base.append(logging_output_loc)
                            sample_sizes_base.append(sample_size_loc)

                    with torch.autograd.profiler.record_function("backward"):
                        optimizer.backward(modloss)

                logging_output={}
                for k in logging_outputs_base[0]:
                    logging_output[k]=sum( lob[k] for lob in logging_outputs_base   )
                
                sample_size=sum(sample_sizes_base)
            else:
                with torch.autograd.profiler.record_function("forward"):
                    loss, sample_size_loc, logging_output_loc = criterion(model, tasksample)
                    if ignore_grad:
                        loss *= 0

                    modloss=loss*weight
                    if self.assume_reduced:
                        modloss=modloss*len(self.tasks)

                    losses.append(modloss)
                    sample_sizes.append(sample_size_loc)

                    if task == "base":
                        sample_size,logging_output=sample_size_loc,logging_output_loc

                with torch.autograd.profiler.record_function("backward"):
                    optimizer.backward(modloss)

        #Combine losses
        loss=sum(losses)
        if self.assume_reduced:
            #Not sure about this
            sample_size=sum(sample_sizes)
        
        return loss, sample_size, logging_output

    def train_step_standard(self, sample, model, criterion, optimizer, update_num, ignore_grad=False):
        losses=[]
        sample_sizes=[]
        with torch.autograd.profiler.record_function("forward"):
            for i,(task,weight) in enumerate(zip(self.tasks,self.weights)):
                loss, sample_size_loc, logging_output_loc = criterion(model, self.create_sample_for_task(sample,task))
                modloss=loss*weight
                if self.assume_reduced:
                    modloss=modloss*len(self.tasks)

                losses.append(modloss)
                sample_sizes.append(sample_size_loc)

                if task == "base":
                    sample_size,logging_output=sample_size_loc,logging_output_loc

        #Combine losses
        loss=sum(losses)
        if self.assume_reduced:
            #Not sure about this
            sample_size=sum(sample_sizes)

        if ignore_grad:
            loss *= 0

        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        
        return loss, sample_size, logging_output

    def train_step_exp3(self, sample, model, criterion, optimizer, update_num, ignore_grad=False):
        if self.prev_dev_loss is not None:
            #This is not the first execution

            #6. (from previous execution) compute dev loss after updating parameters
            dev_loss,dev_sample_size, dev_log=criterion(model,self.dev_sample)
            dev_loss=torch.sum(dev_loss).item()/dev_sample_size

            # 7. (from previous execution) update exp3 parameters
            # Compute reward: pg
            if self.args.exp3_reward == "pg":
                reward= self.prev_dev_loss - dev_loss
            elif self.args.exp3_reward == "pgnorm":
                reward=1 - dev_loss/self.prev_dev_loss
            elif self.args.exp3_reward == "dummy" :
                reward=0
            else:
                raise ValueError("--exp3-reward not supported")

            if self.args.exp3_reward != "dummy":
                self.exp3.update(self.prev_task,reward)

            if self.summary_writer is not None:
                probs=self.exp3.task_probabilities
                d={}
                for i,task in enumerate(self.tasks):
                    d[task]=probs[i]
                self.summary_writer.add_scalars('exp3/probs',d,update_num)
                self.summary_writer.add_scalar('exp3/reward',reward,update_num)


        # 1. sample a minibatch from the dev set
        if not self.dev_iterator.has_next():
            self.dev_iterator=self.dev_batch_iterator.next_epoch_itr()
        self.dev_sample=utils.move_to_cuda(self.dev_iterator.__next__())


        # 2. compute dev loss before updating parameters
        dev_loss,dev_sample_size, dev_log=criterion(model,self.dev_sample)
        self.prev_dev_loss=torch.sum(dev_loss).item()/dev_sample_size

        # 3. sample a task
        task=self.exp3.get_task()

        # 4. forward-backward
        with torch.autograd.profiler.record_function("forward"):
            loss, sample_size, logging_output = criterion(model, self.create_sample_for_task(sample,task))

        if ignore_grad:
            loss *= 0

        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)

        self.prev_task=task
        # 5. (in trainer.py) optimizer update

        return loss, sample_size, logging_output

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        """
        Do n forward passes and accumulate gradients

        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.
        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            update_num (int): the current update
            ignore_grad (bool): multiply loss by 0 if this is set to True
        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """

        model.train()
        model.set_num_updates(update_num)
        self.last_train_update_num=update_num

        if self.multiple_backward:
            if self.args.zero_negative_gradients or self.args.surgery_negative_gradients or self.args.vaccine_gradients:
                return self.train_step_gradient_modifcation_opt( sample, model, criterion, optimizer, update_num, ignore_grad)
            else:
                return self.train_step_multiple_backward(sample, model, criterion, optimizer, update_num, ignore_grad)
                
        elif self.args.exp3:
            return self.train_step_exp3(sample, model, criterion, optimizer, update_num, ignore_grad)
        else:
            return self.train_step_standard(sample, model, criterion, optimizer, update_num, ignore_grad)
