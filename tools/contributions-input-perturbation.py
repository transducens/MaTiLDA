"""
Evaluate the perplexity of a trained language model.
"""
from enum import Enum
import logging
import math
import os

import torch
from torch import Tensor

from fairseq import checkpoint_utils, distributed_utils, options, tasks, utils
from fairseq.data import LMContextWindowDataset
from fairseq.logging import progress_bar
from fairseq.logging.meters import StopwatchMeter, TimeMeter
from fairseq.sequence_scorer import SequenceScorer
from typing import Any, Dict, List, Optional, Tuple
from fairseq.models.fairseq_encoder import EncoderOut

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

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
)
logger = logging.getLogger("fairseq_cli.eval_lm")

#This could be computed more efficiently with pytorch
def compute_variance(data, logprobs=False):
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


def main(parsed_args, **unused_kwargs):
    assert parsed_args.path is not None, "--path required for evaluation!"

    if torch.cuda.is_available() and not parsed_args.cpu:
        torch.cuda.set_device(parsed_args.device_id)

    utils.import_user_module(parsed_args)

    logger.info(parsed_args)

    use_cuda = torch.cuda.is_available() and not parsed_args.cpu

    task = tasks.setup_task(parsed_args)

    # Load ensemble
    logger.info("loading model(s) from {}".format(parsed_args.path))
    models, args = checkpoint_utils.load_model_ensemble(
        parsed_args.path.split(os.pathsep),
        arg_overrides=eval(parsed_args.model_overrides),
        task=task,
        suffix=getattr(parsed_args, "checkpoint_suffix", ""),
        strict=(parsed_args.checkpoint_shard_count == 1),
        num_shards=parsed_args.checkpoint_shard_count,
    )

    for arg in vars(parsed_args).keys():
        if arg not in {
            "self_target",
            "future_target",
            "past_target",
            "tokens_per_sample",
            "output_size_dictionary",
            "add_bos_token",
        }:
            setattr(args, arg, getattr(parsed_args, arg))

    
    task = tasks.setup_task(args)

    # Load dataset splits
    task.load_dataset(args.gen_subset)
    dataset = task.dataset(args.gen_subset)
    if args.context_window > 0:
        dataset = LMContextWindowDataset(
            dataset=dataset,
            tokens_per_sample=args.tokens_per_sample,
            context_window=args.context_window,
            pad_idx=task.source_dictionary.pad(),
        )
    logger.info("{} {} {} examples".format(args.data, args.gen_subset, len(dataset)))

    # Optimize ensemble for generation and set the source and dest dicts on the model (required by scorer)
    for model in models:
        if args.fp16:
            model.half()
        if use_cuda and not args.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(args)

    assert len(models) > 0

    logger.info(
        "num. model params: {}".format(sum(p.numel() for p in models[0].parameters()))
    )

    itr = task.get_batch_iterator(
        dataset=dataset,
        max_tokens=args.max_tokens or 36000,
        max_sentences=args.batch_size,
        max_positions=utils.resolve_max_positions(
            *[model.max_positions() for model in models]
        ),
        ignore_invalid_inputs=True,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
        data_buffer_size=args.data_buffer_size,
    ).next_epoch_itr(shuffle=False)
    progress = progress_bar.progress_bar(
        itr,
        log_format=args.log_format,
        log_interval=args.log_interval,
        default_log_format=("tqdm" if not args.no_progress_bar else "none"),
    )

    scorer_source = TransformerPerturbationSequenceScorer('SOURCE',task.source_dictionary,task.target_dictionary, args.softmax_batch)
    scorer_target = TransformerPerturbationSequenceScorer('TARGET',task.source_dictionary,task.target_dictionary, args.softmax_batch)

    score_sum = 0.0
    count = 0

    if args.remove_bpe is not None:
        if args.remove_bpe == "sentencepiece":
            raise NotImplementedError
        else:
            raise NotImplementedError
    else:
        bpe_toks = None
        bpe_len = 0

    
    #keys: sample_id
    #value: list of lists
    #     each list of "value" corresponds to a run
    #           each sublist corresponds to the sequence of word probabilities of the run.
    #               Each element is a tuple (word, probability)
    perturbation_data_source={}
    perturbation_data_target={}

    for sample in progress:
        if "net_input" not in sample:
            continue

        sample = utils.move_to_cuda(sample) if use_cuda else sample
        
        for scorer,perturbation_data in [scorer_source,perturbation_data_source],[scorer_target,perturbation_data_target]:
            #print("Number of pertubations: {}".format(args.n_perturbations))
            for i in range(args.n_perturbations):
                hypos = scorer.generate(models, sample)

                #Just 10 hypos: speed
                for i, hypos_i in enumerate(hypos):
                    #import pdb; pdb.set_trace()
                    hypo = hypos_i[0]
                    sample_id = sample["id"][i].item()

                    tokens = hypo["tokens"]
                    tgt_len = tokens.numel()
                    pos_scores = hypo["positional_scores"].float()

                    if getattr(args, "add_bos_token", False):
                        assert hypo["tokens"][0].item() == task.target_dictionary.bos()
                        tokens = tokens[1:]
                        pos_scores = pos_scores[1:]

                    skipped_toks = 0

                    inf_scores = pos_scores.eq(float("inf")) | pos_scores.eq(float("-inf"))
                    if inf_scores.any():
                        logger.info(
                            "skipping tokens with inf scores:",
                            task.target_dictionary.string(tokens[inf_scores.nonzero()]),
                        )
                        pos_scores = pos_scores[(~inf_scores).nonzero()]
                    score_sum += pos_scores.sum().cpu()
                    count += pos_scores.numel() - skipped_toks

                    word_prob = []
                    for i in range(len(tokens)):
                        w_ind = tokens[i].item()
                        
                        w = task.target_dictionary[w_ind]
                        
                        word_prob.append((w, pos_scores[i].item()))
                    
                    if sample_id not in perturbation_data:
                        perturbation_data[sample_id]=[]
                    perturbation_data[sample_id].append(word_prob)

                    if args.output_word_probs:   
                        logger.info(
                            str(int(sample_id))
                            + " "
                            + (
                                "\t".join(
                                    "{} [{:2f}]".format(x[0], x[1]) for x in word_prob
                                )
                            )
                        )

    avg_nll_loss = -score_sum / count / math.log(2)  # convert to base 2
   
    perturbation_tokens={}
    for i in perturbation_data_source.keys():
        perturbation_tokens[i]= [t[0] for t in perturbation_data_source[i][0] ]
    
    variances_source=compute_variance(perturbation_data_source)
    variances_target=compute_variance(perturbation_data_target)

    total_toks=0
    total_source_infl=0
    for i in sorted(variances_source.keys()):
        print("{} tokens: {}".format(i," ".join(perturbation_tokens[i])))
        print("{} source: {}".format(i,variances_source[i]))
        print("{} target: {}".format(i,variances_target[i]))
        source_influences=[ vsource/(vsource+vtarget)  for vsource,vtarget in zip(variances_source[i],variances_target[i]) ]
        print("{} summary: {}".format(i, " ".join([ token+"("+str(sinfl)+")"  for token,sinfl in zip(perturbation_tokens[i],source_influences) ])   ))
        print()
        total_toks+=len(source_influences)
        total_source_infl+=sum(source_influences)
    
    print("Average source influence: {}".format(total_source_infl/total_toks))


    logger.info(
        "Loss (base 2): {:.4f}, Perplexity: {:.2f}".format(
            avg_nll_loss, 2 ** avg_nll_loss
        )
    )


def cli_main():
    parser = options.get_eval_lm_parser()
    parser.add_argument("--n-perturbations",type=int, default=2)
    args = options.parse_args_and_arch(parser)
    distributed_utils.call_main(args, main)


if __name__ == "__main__":
    cli_main()