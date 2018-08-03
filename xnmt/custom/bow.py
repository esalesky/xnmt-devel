from typing import Optional

import dynet as dy
import numpy as np

from xnmt import batchers, event_trigger, inferences, input_readers, losses, output, vocabs
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
    self.post_processor = output.PlainTextOutputProcessor()
    self.generate_per_step = generate_per_step

  def shared_params(self):
    return [{".src_embedder.emb_dim", ".encoder.input_dim"},
            {".encoder.hidden_dim", ".mlp_layer.input_dim"}]

  def calc_loss(self, src, trg, loss_calculator):
    event_trigger.start_sent(src)
    embeddings = self.src_embedder.embed_sent(src)
    encodings = self.encoder(embeddings)
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
      scores = dy.logistic(self.output_layer(encoding_fixed_size))
    elif self.mode=="lin_sum_sig":
      enc_lin = []
      for step_i, enc_i in enumerate(encodings):
        step_linear = self.output_layer(enc_i)
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
    for batch_i in range(len(trg)):
      for word in set(trg[batch_i]):
        if word not in {vocabs.Vocab.ES, vocabs.Vocab.SS}:
          idxs[0].append(word)
          idxs[1].append(batch_i)
    trg_scores = dy.sparse_inputTensor(idxs, values = np.ones(len(idxs[0])), shape=scores.dim()[0] + (scores.dim()[1],), batched=True, )
    loss_expr = dy.binary_log_loss(scores, trg_scores)
    bow_loss = losses.FactoredLossExpr({"mle" : loss_expr})

    return bow_loss

  def generate(self, src, idx):
    if not batchers.is_batched(src):
      src = batchers.mark_as_batch([src])
    assert len(src)==1, "batched generation not fully implemented"
    # Generating outputs
    outputs = []
    for sents in src:
      event_trigger.start_sent(src)
      embeddings = self.src_embedder.embed_sent(sents)
      encodings = self.encoder(embeddings)
      if self.mode in ["avg_mlp", "final_mlp"]:
        if self.generate_per_step:
          assert self.mode == "avg_mlp", "final_mlp not supported with generate_per_step=True"
          scores = [dy.logistic(self.output_layer(enc_i)) for enc_i in encodings]
        else:
          if self.mode == "avg_mlp":
            encoding_fixed_size = dy.sum_dim(encodings.as_tensor(), [1]) * (1.0 / encodings.dim()[0][1])
          elif self.mode == "final_mlp":
            encoding_fixed_size = self.encoder.get_final_states()[-1].main_expr()
          scores = dy.logistic(self.output_layer(encoding_fixed_size))
      elif self.mode == "lin_sum_sig":
        enc_lin = []
        for step_i, enc_i in enumerate(encodings):
          step_linear = self.output_layer(enc_i)
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
        # Append output to the outputs
        outputs.append(output.TextOutput(actions=output_actions,
                                         vocab=self.trg_reader.vocab,
                                         score=score))
      else:
        scores_arr = scores.npvalue()
        output_actions = list(np.nonzero(scores_arr > 0.5)[0])
        score = np.sum(scores_arr[scores_arr > 0.5])
        # Append output to the outputs
        outputs.append(output.TextOutput(actions=output_actions,
                                         vocab=self.trg_reader.vocab,
                                         score=score))
    return outputs

  def set_post_processor(self, post_processor):
    self.post_processor = post_processor

  def set_trg_vocab(self, trg_vocab=None):
    """
    Set target vocab for generating outputs. If not specified, word IDs are generated instead.

    Args:
      trg_vocab (Vocab): target vocab, or None to generate word IDs
    """
    self.trg_vocab = trg_vocab

  def get_primary_loss(self):
    return "mle"

  def get_nobp_state(self, state):
    output_state = state.rnn_state.output()
    return output_state
