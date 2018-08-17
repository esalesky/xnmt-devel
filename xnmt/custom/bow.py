from typing import Optional, Sequence

import dynet as dy
import numpy as np

from xnmt import batchers, event_trigger, inferences, input_readers, losses, sent, vocabs
from xnmt.eval import metrics
from xnmt.modelparts import embedders, transforms
from xnmt.models import base as models
from xnmt.transducers import recurrent, base as transducers
from xnmt.persistence import serializable_init, Serializable, bare, Ref


class BowPredictor(models.ConditionedModel, models.GeneratorModel, Serializable):
  """
  A default translator based on attentional sequence-to-sequence models.

  Args:
    src_reader: A reader for the source side.
    trg_reader: A reader for the target side.
    src_embedder: A word embedder for the input language
    encoder: An encoder to generate encoded inputs
    inference:
    output_layer: final prediction linear layer
    mode: ``avg_mlp``: avg(encoder states) -> linear -> sigmoid -> binary_log_loss                ["vote-then-classify"]
          ``final_mlp``: final encoder states -> linear -> sigmoid -> binary_log_loss                       ["remember"]
          ``lin_sum_sig``: sum ( enc_state -> linear ) -> sigmoid -> binary_log_loss              ["classify-then-vote"]
  """

  yaml_tag = '!BowPredictor'

  @serializable_init
  def __init__(self,
               src_reader: input_readers.InputReader,
               trg_reader: input_readers.InputReader,
               src_embedder: embedders.Embedder = bare(embedders.SimpleWordEmbedder),
               encoder: transducers.SeqTransducer = bare(recurrent.BiLSTMSeqTransducer),
               inference=bare(inferences.IndependentOutputInference),
               hidden_dim: int = Ref("exp_global.default_layer_dim"),
               output_layer: Optional[transforms.Linear] = None,
               generate_per_step: bool = False,
               mode:str="avg_mlp"):
    super().__init__(src_reader=src_reader, trg_reader=trg_reader)
    self.src_embedder = src_embedder
    self.encoder = encoder
    self.output_layer = self.add_serializable_component("output_layer",
                                                        output_layer,
                                                        lambda:transforms.Linear(input_dim=hidden_dim,
                                                                             output_dim=len(trg_reader.vocab)))
    self.inference = inference
    self.mode = mode
    self.generate_per_step = generate_per_step

  def shared_params(self):
    return [{".src_embedder.emb_dim", ".encoder.input_dim"},
            {".encoder.hidden_dim", ".mlp_layer.input_dim"}]

  def calc_nll(self, src, trg):
    event_trigger.start_sent(src)
    embeddings = self.src_embedder.embed_sent(src)
    encodings = self.encoder.transduce(embeddings)
    if not batchers.is_batched(trg): trg = batchers.mark_as_batch([trg])

    if self.mode in ["avg_mlp", "final_mlp"]:
      if self.mode=="avg_mlp":
        if encodings.mask:
          encoding_fixed_size = dy.cdiv(dy.sum_dim(encodings.as_tensor(), [1]),
                                 dy.inputTensor(np.sum(1.0 - encodings.mask.np_arr, axis=1), batched=True))
        else:
          encoding_fixed_size = dy.sum_dim(encodings.as_tensor(), [1]) / encodings.dim()[0][1]
      elif self.mode=="final_mlp":
        encoding_fixed_size = self.encoder.get_final_states()[-1].main_expr()
      scores = dy.logistic(self.output_layer.transform(encoding_fixed_size))
    elif self.mode=="lin_sum_sig":
      enc_lin = []
      for step_i, enc_i in enumerate(encodings):
        step_linear = self.output_layer.transform(enc_i)
        if encodings.mask and np.sum(encodings.mask.np_arr[:,step_i])>0:
          step_linear = dy.cmult(step_linear, dy.inputTensor(1.0 - encodings.mask.np_arr[:,step_i], batched=True))
        enc_lin.append(step_linear)
      if encodings.mask:
        encoding_fixed_size = dy.cdiv(dy.esum(enc_lin),
                                      dy.inputTensor(np.sum(1.0 - encodings.mask.np_arr, axis=1), batched=True))
      else:
        encoding_fixed_size = dy.esum(enc_lin) / encodings.dim()[0][1]
      scores = dy.logistic(encoding_fixed_size)

    else: raise ValueError(f"unknown mode '{self.mode}'")

    idxs = ([], [])
    for batch_i in range(trg.batch_size()):
      for word in set(trg[batch_i]):
        if word not in {vocabs.Vocab.ES, vocabs.Vocab.SS}:
          idxs[0].append(word)
          idxs[1].append(batch_i)
    trg_scores = dy.sparse_inputTensor(idxs, values = np.ones(len(idxs[0])), shape=scores.dim()[0] + (scores.dim()[1],), batched=True, )
    loss_expr = dy.binary_log_loss(scores, trg_scores)
    return loss_expr

  def generate(self, src, forced_trg_ids):
    assert not forced_trg_ids
    assert batchers.is_batched(src) and src.batch_size()==1, "batched generation not fully implemented"
    src = src[0]
    # Generating outputs
    outputs = []
    event_trigger.start_sent(src)
    embeddings = self.src_embedder.embed_sent(src)
    encodings = self.encoder.transduce(embeddings)
    if self.mode in ["avg_mlp", "final_mlp"]:
      if self.generate_per_step:
        assert self.mode == "avg_mlp", "final_mlp not supported with generate_per_step=True"
        scores = [dy.logistic(self.output_layer.transform(enc_i)) for enc_i in encodings]
      else:
        if self.mode == "avg_mlp":
          encoding_fixed_size = dy.sum_dim(encodings.as_tensor(), [1]) * (1.0 / encodings.dim()[0][1])
        elif self.mode == "final_mlp":
          encoding_fixed_size = self.encoder.get_final_states()[-1].main_expr()
        scores = dy.logistic(self.output_layer.transform(encoding_fixed_size))
    elif self.mode == "lin_sum_sig":
      enc_lin = []
      for step_i, enc_i in enumerate(encodings):
        step_linear = self.output_layer.transform(enc_i)
        if encodings.mask and np.sum(encodings.mask.np_arr[:, step_i]) > 0:
          step_linear = dy.cmult(step_linear, dy.inputTensor(1.0 - encodings.mask.np_arr[:, step_i], batched=True))
        enc_lin.append(step_linear)
      if self.generate_per_step:
        scores = [dy.logistic(enc_i) for enc_i in enc_lin]
      else:
        if encodings.mask:
          encoding_fixed_size = dy.cdiv(dy.esum(enc_lin),
                                        dy.inputTensor(np.sum(1.0 - encodings.mask.np_arr, axis=1), batched=True))
        else:
          encoding_fixed_size = dy.esum(enc_lin) / encodings.dim()[0][1]
        scores = dy.logistic(encoding_fixed_size)
    else:
      raise ValueError(f"unknown mode '{self.mode}'")

    if self.generate_per_step:
      output_actions = [np.argmax(score_i.npvalue()) for score_i in scores]
      score = np.sum([np.max(score_i.npvalue()) for score_i in scores])
      outputs.append(sent.SimpleSentence(words=output_actions,
                                         idx=src.idx,
                                         vocab=getattr(self.trg_reader, "vocab", None),
                                         score=score,
                                         output_procs=self.trg_reader.output_procs))
    else:
      scores_arr = scores.npvalue()
      output_actions = list(np.nonzero(scores_arr > 0.5)[0])
      score = np.sum(scores_arr[scores_arr > 0.5])
      outputs.append(sent.SimpleSentence(words=output_actions,
                                         idx=src.idx,
                                         vocab=getattr(self.trg_reader, "vocab", None),
                                         score=score,
                                         output_procs=self.trg_reader.output_procs))
    return outputs

  def set_trg_vocab(self, trg_vocab=None):
    """
    Set target vocab for generating outputs. If not specified, word IDs are generated instead.

    Args:
      trg_vocab (Vocab): target vocab, or None to generate word IDs
    """
    self.trg_vocab = trg_vocab

  def get_nobp_state(self, state):
    output_state = state.rnn_state.output()
    return output_state


class BowFMeasureEvaluator(metrics.SentenceLevelEvaluator, Serializable):
  yaml_tag = "!BowFMeasureEvaluator"
  @serializable_init
  def __init__(self, case_sensitive=False, write_sentence_scores: Optional[str] = None) -> None:
    super().__init__(write_sentence_scores=write_sentence_scores)
    self.case_sensitive = case_sensitive

  def evaluate_one_sent(self, ref:Sequence[str], hyp:Sequence[str]):
    if not self.case_sensitive:
      ref = [ref_i.lower() for ref_i in ref]
      hyp = [hyp_i.lower() for hyp_i in hyp]
    ref_set = set(ref)
    hyp_set = set(hyp)
    return metrics.FMeasure(true_pos=len(hyp_set & ref_set), false_neg=len(ref_set - hyp_set), false_pos=len(hyp_set - ref_set))
