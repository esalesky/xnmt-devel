from typing import Any, Optional, Sequence, Union
import numbers
import random

import numpy as np
import dynet as dy

from xnmt import batchers, expression_seqs, event_trigger, events, inferences, input_readers, losses, reports, sent, \
  vocabs
from xnmt.modelparts import attenders, bridges, embedders, scorers, transforms
from xnmt.models import base as models
from xnmt.persistence import Serializable, serializable_init, bare, Ref
from xnmt.transducers import recurrent, base as transducers

class SymmetricDecoderState(object):
  def __init__(self, rnn_state: dy.Expression = None, context: dy.Expression = None, out_prob: dy.Expression = None):
    self.rnn_state = rnn_state
    self.context = context
    self.out_prob = out_prob


class SymmetricTranslator(models.ConditionedModel, models.GeneratorModel, Serializable, transducers.SeqTransducer,
                          reports.Reportable):
  """
  Args:
    src_reader:
    trg_reader:
    src_embedder:
    encoder:
    attender:
    dec_lstm:
    bridge:
    transform:
    scorer:
    inference:
    max_dec_len:
    mode: what to feed into the LSTM input
          * ``context``: feed the previous attention context
          * ``expected``: word embeddings weighted by softmax probs from last time step
          * ``argmax``: discrete token lookup based on model predictions
          * ``argmax_st``: as ``argmax``, but use the straight-through gradient estimator for the argmax operation
          * ``teacher``: as ``argmax`` at test time, but use teacher forcing (correct labels) for training
          * ``split``: teacher-forcing to compute attentions, then feed in current (not previous) context
    mode_translate: what to feed into the LSTM input in 'translate' mode. same options as for ``mode``.
    mode_transduce: what to feed into the LSTM input in 'transduce' mode. same options as for ``mode``.
    unfold_until: how long to unfold the RNN at test time for both the transducer and generator model variants, and at
                  training time for the transducer model (train time / generator model is dictated by data)
                  * ``eos``: unfold until EOS token gets largest probability (or max_dec_len is reached)
                  * ``supervised``:
    transducer_loss: if True, add transducer loss as auxiliary loss
    split_regularizer: if True, use additional loss ||E(y_i)-context_i|| (if a float value, it's used to scale the loss)
    split_dual: feed both current context and current label into split (step-2) RNN (with projection to match dimensions)
                - pair of floats: dropout probs, i.e. (context_drop, label_drop)
                - ``True``: equivalent to [0.0, 0.0]
    dropout_dec_state: rate for block dropout applied on decoder state, so that only context vector is passed to output
    split_dual_proj: automatically set
    split_context_transform: run split context through transform before feeding back into RNN
    sampling_prob: for teacher or split mode, probability of sampling from model rather than using teacher forcing
    compute_report:
  """
  yaml_tag = "!SymmetricTranslator"

  @events.register_xnmt_handler
  @serializable_init
  def __init__(self,
               trg_embedder: embedders.DenseWordEmbedder,
               src_reader: input_readers.InputReader = None,
               trg_reader: input_readers.InputReader = None,
               src_embedder=bare(embedders.SimpleWordEmbedder),
               encoder=bare(recurrent.BiLSTMSeqTransducer),
               attender=bare(attenders.MlpAttender),
               dec_lstm=bare(recurrent.UniLSTMSeqTransducer),
               bridge: bridges.Bridge = bare(bridges.CopyBridge),
               transform: transforms.Transform = bare(transforms.AuxNonLinear),
               scorer: scorers.Scorer = bare(scorers.Softmax),
               inference=bare(inferences.IndependentOutputInference),
               max_dec_len: int = 350,
               mode: Optional[str] = None,
               mode_translate: Optional[str] = None,
               mode_transduce: Optional[str] = None,
               unfold_until: str = "eos",
               transducer_loss: bool = False,
               split_regularizer: Union[bool, numbers.Real] = False,
               split_dual: Union[bool, Sequence[numbers.Real]] = False,
               dropout_dec_state: float = 0.0,
               split_dual_proj: Optional[transforms.Linear] = None,
               split_context_transform: Optional[transforms.Transform]= None,
               sampling_prob: numbers.Number = 0.0,
               compute_report: bool = Ref("exp_global.compute_report", default=False)):
    super().__init__(src_reader=src_reader, trg_reader=trg_reader)
    assert mode is None or (mode_translate is None and mode_transduce is None), \
      f"illegal combination: mode={mode}, mode_translate={mode_translate}, mode_transduce={mode_transduce}"
    assert mode or mode_translate or mode_transduce
    if mode_translate or mode_transduce: assert mode_translate and mode_transduce
    assert mode_translate != "split"
    self.src_embedder = src_embedder
    self.trg_embedder = trg_embedder
    self.encoder = encoder
    self.attender = attender
    self.dec_lstm = dec_lstm
    self.bridge = bridge
    self.transform = transform
    self.scorer = scorer
    self.inference = inference
    self.max_dec_len = max_dec_len
    self.mode_translate = mode_translate or mode
    self.mode_transduce = mode_transduce or mode
    if transducer_loss:
      assert self.mode_transduce in ["teacher", "split"], \
        f"mode_transduce='{self.mode_transduce}' not supported with transducer_loss option"
    self.trg_embedder = trg_embedder
    self.unfold_until = unfold_until
    self.transducer_loss = transducer_loss
    if split_regularizer: assert self.mode_transduce == "split"
    self.split_regularizer = split_regularizer
    self.dropout_dec_state = dropout_dec_state
    self.split_dual = [0.0, 0.0] if split_dual is True else split_dual
    self.split_context_transform = split_context_transform
    if self.split_dual:
      assert len(self.split_dual)==2 and max(self.split_dual) <= 1.0 and min(self.split_dual) >= 0.0
      self.split_dual_proj = self.add_serializable_component("split_dual_proj", split_dual_proj,
                                                             lambda: transforms.Linear(input_dim=self.dec_lstm.input_dim*2,
                                                                                           output_dim=self.dec_lstm.input_dim))
    self.sampling_prob = sampling_prob
    self.compute_report = compute_report

  def shared_params(self):
    return [{".src_embedder.emb_dim", ".encoder.input_dim"},
            {".encoder.hidden_dim", ".attender.input_dim", ".dec_lstm.input_dim"},
            {".attender.state_dim", ".dec_lstm.hidden_dim", ".transform.aux_input_dim"}]

  def _encode_src(self, x, apply_emb = True):
    if apply_emb:
      x = self.src_embedder.embed_sent(x)
    encodings = self.encoder.transduce(x)
    self.attender.init_sent(encodings)
    batch_size = encodings.dim()[1]

    enc_final_states = self.encoder.get_final_states()
    rnn_state = self.dec_lstm.initial_state()
    rnn_state = rnn_state.set_s(self.bridge.decoder_init(enc_final_states))
    zeros = dy.zeros(self.dec_lstm.input_dim, batch_size=batch_size)

    ss = batchers.mark_as_batch([vocabs.Vocab.SS] * batch_size)
    first_input = self.trg_embedder.embed(ss)
    self._chosen_rnn_inputs.append(first_input)
    rnn_state = rnn_state.add_input(first_input)
    return SymmetricDecoderState(rnn_state=rnn_state, context=zeros)

  @events.handle_xnmt_event
  def on_start_sent(self, src):
    self.cur_src = src
    self.transducer_losses = []
    self._chosen_rnn_inputs = []
    self.split_reg_penalty_expr = None

  @events.handle_xnmt_event
  def on_set_train(self, val):
    self.train = val

  def transduce(self, x):
    # some preparations
    output_states = []
    current_state = self._encode_src(x, apply_emb=False)
    if self.mode_transduce == "split":
      first_state = SymmetricDecoderState(rnn_state=current_state.rnn_state, context=current_state.context)
    batch_size = x.dim()[1]
    done = [False] * batch_size
    out_mask = batchers.Mask(np_arr=np.zeros((batch_size, self.max_dec_len)))
    out_mask.np_arr.flags.writeable = True
    # teacher / split mode: unfold guided by reference targets
    #  -> feed everything up unto (except) the last token back into the LSTM
    # other modes: unfold until EOS is output or max len is reached
    max_dec_len = self.cur_src.batches[1].sent_len() if self.mode_transduce in ["teacher", "split"] else self.max_dec_len
    atts_list = []
    generated_word_ids = []
    for pos in range(max_dec_len):
      if self.train and self.mode_transduce in ["teacher", "split"]:
        # unroll RNN guided by reference
        prev_ref_action, ref_action = None, None
        if pos > 0:
          prev_ref_action = self._batch_ref_action(pos-1)
        if self.transducer_loss:
          ref_action = self._batch_ref_action(pos)
        step_loss = self.calc_loss_one_step(dec_state=current_state,
                                            batch_size=batch_size,
                                            mode=self.mode_transduce,
                                            ref_action=ref_action,
                                            prev_ref_action=prev_ref_action)
        self.transducer_losses.append(step_loss)
      else: # inference
        # unroll RNN guided by model predictions
        if self.mode_transduce in ["teacher", "split"]:
          prev_ref_action = self._batch_max_action(batch_size, current_state, pos)
        else: prev_ref_action = None
        out_scores = self.generate_one_step(dec_state=current_state,
                                            mask=out_mask,
                                            cur_step=pos,
                                            batch_size=batch_size,
                                            mode=self.mode_transduce,
                                            prev_ref_action=prev_ref_action)
        word_id = np.argmax(out_scores.npvalue(), axis=0)
        word_id = word_id.reshape((word_id.size,))
        generated_word_ids.append(word_id[0])
        for batch_i in range(batch_size):
          if self._terminate_rnn(batch_i=batch_i, pos=pos, batched_word_id=word_id):
            done[batch_i] = True
            out_mask.np_arr[batch_i,pos+1:] = 1.0
        if pos>0 and all(done):
          atts_list.append(self.attender.get_last_attention())
          output_states.append(current_state.rnn_state.h()[-1])
          break
      output_states.append(current_state.rnn_state.h()[-1])
      atts_list.append(self.attender.get_last_attention())
    if self.mode_transduce == "split":
      # split mode: use attentions to compute context, then run RNNs over these context inputs
      if self.split_regularizer:
        assert len(atts_list) == len(self._chosen_rnn_inputs), f"{len(atts_list)} != {len(self._chosen_rnn_inputs)}"
      split_output_states = []
      split_rnn_state = first_state.rnn_state
      for pos, att in enumerate(atts_list):
        lstm_input_context = self.attender.curr_sent.as_tensor() * att # TODO: better reuse the already computed context vecs
        lstm_input_context = dy.reshape(lstm_input_context, (lstm_input_context.dim()[0][0],), batch_size=batch_size)
        if self.split_dual:
          lstm_input_label = self._chosen_rnn_inputs[pos]
          if self.split_dual[0] > 0.0 and self.train:
            lstm_input_context = dy.dropout_batch(lstm_input_context, self.split_dual[0])
          if self.split_dual[1] > 0.0 and self.train:
            lstm_input_label = dy.dropout_batch(lstm_input_label, self.split_dual[1])
          if self.split_context_transform:
            lstm_input_context = self.split_context_transform.transform(lstm_input_context)
          lstm_input_context = self.split_dual_proj.transform(dy.concatenate([lstm_input_context, lstm_input_label]))
        if self.split_regularizer and pos < len(self._chosen_rnn_inputs):
          # _chosen_rnn_inputs does not contain first (empty) input, so this is in fact like comparing to pos-1:
          penalty = dy.squared_norm(lstm_input_context - self._chosen_rnn_inputs[pos])
          if self.split_regularizer != 1:
            penalty = self.split_regularizer * penalty
          self.split_reg_penalty_expr = penalty
        split_rnn_state = split_rnn_state.add_input(lstm_input_context)
        split_output_states.append(split_rnn_state.h()[-1])
      assert len(output_states) == len(split_output_states)
      output_states = split_output_states
    out_mask.np_arr = out_mask.np_arr[:,:len(output_states)]
    self._final_states = []
    if self.compute_report:
      # for symmetric reporter (this can only be run at inference time)
      assert batch_size==1
      atts_matrix = np.asarray([att.npvalue() for att in atts_list]).reshape(len(atts_list), atts_list[0].dim()[0][0]).T
      self.report_sent_info({"symm_att": atts_matrix,
                             "symm_out": sent.SimpleSentence(words=generated_word_ids,
                                                             idx=self.cur_src.batches[0][0].idx,
                                                             vocab=self.cur_src.batches[1][0].vocab,
                                                             output_procs=self.cur_src.batches[1][0].output_procs),
                             "symm_ref": self.cur_src.batches[1][0] if isinstance(self.cur_src,
                                                                                  batchers.CompoundBatch) else None})
    # prepare final outputs
    for layer_i in range(len(current_state.rnn_state.h())):
      self._final_states.append(transducers.FinalTransducerState(main_expr=current_state.rnn_state.h()[layer_i],
                                                         cell_expr=current_state.rnn_state._c[layer_i]))
    out_mask.np_arr.flags.writeable = False
    return expression_seqs.ExpressionSequence(expr_list=output_states, mask=out_mask)

  def _batch_max_action(self, batch_size, current_state, pos):
    if pos == 0:
      return None
    elif batch_size > 1:
      return batchers.mark_as_batch(np.argmax(current_state.out_prob.npvalue(), axis=0))
    else:
      return batchers.mark_as_batch([np.argmax(current_state.out_prob.npvalue(), axis=0)])

  def _batch_ref_action(self, pos):
    ref_action = []
    for src_sent in self.cur_src.batches[1]:
      if src_sent[pos] is None:
        ref_action.append(vocabs.Vocab.ES)
      else:
        ref_action.append(src_sent[pos])
    ref_action = batchers.mark_as_batch(ref_action)
    return ref_action

  def get_final_states(self):
    return self._final_states

  def calc_nll(self, src, trg):
    event_trigger.start_sent(src)
    if isinstance(src, batchers.CompoundBatch):
      src, _ = src.batches
    initial_state = self._encode_src(src)

    dec_state = initial_state
    trg_mask = trg.mask if batchers.is_batched(trg) else None
    losses = []
    seq_len = trg.sent_len()
    if batchers.is_batched(src):
      for j, single_trg in enumerate(trg):
        assert single_trg.sent_len() == seq_len  # assert consistent length
        assert 1 == len([i for i in range(seq_len) if (trg_mask is None or trg_mask.np_arr[j, i] == 0)
                         and single_trg[i] == vocabs.Vocab.ES])  # assert exactly one unmasked ES token
    prev_ref_word = None
    for i in range(seq_len):
      if not batchers.is_batched(trg):
        ref_word = trg[i]
      else:
        ref_word = batchers.mark_as_batch([single_trg[i] for single_trg in trg])
      word_loss = self.calc_loss_one_step(dec_state=dec_state,
                                                     batch_size=ref_word.batch_size(),
                                                     ref_action=ref_word,
                                                     prev_ref_action=prev_ref_word,
                                                     mode=self.mode_translate)
      if batchers.is_batched(src) and trg_mask is not None:
        word_loss = trg_mask.cmult_by_timestep_expr(word_loss, i, inverse=True)
      losses.append(word_loss)
      prev_ref_word = ref_word

    return dy.esum(losses)

  def generate(self, src, forced_trg_ids=None, **kwargs):
    event_trigger.start_sent(src)
    if isinstance(src, batchers.CompoundBatch):
      src = src.batches[0]

    outputs = []

    batch_size = src.batch_size()
    score = batchers.ListBatch([[] for _ in range(batch_size)])
    words = batchers.ListBatch([[] for _ in range(batch_size)])
    done = [False] * batch_size
    initial_state = self._encode_src(src)
    current_state = initial_state
    attentions = []
    for pos in range(self.max_dec_len):
      prev_ref_action = None
      if pos > 0 and self.mode_translate != "context":
        if forced_trg_ids is not None:
          prev_ref_action = batchers.mark_as_batch(
            [forced_trg_ids[batch_i][pos - 1] for batch_i in range(batch_size)])
        elif batch_size > 1:
          prev_ref_action = batchers.mark_as_batch(np.argmax(current_state.out_prob.npvalue(), axis=0))
        else:
          prev_ref_action = batchers.mark_as_batch([np.argmax(current_state.out_prob.npvalue(), axis=0)])

      logsoft = self.generate_one_step(dec_state=current_state,
                                       batch_size=batch_size,
                                       mode=self.mode_translate,
                                       cur_step=pos,
                                       prev_ref_action=prev_ref_action)
      attentions.append(self.attender.get_last_attention().npvalue())
      logsoft = logsoft.npvalue()
      logsoft = logsoft.reshape(logsoft.shape[0],batch_size)
      if forced_trg_ids is None:
        batched_word_id = np.argmax(logsoft, axis=0)
        batched_word_id = batched_word_id.reshape((batched_word_id.size,))
      else:
        batched_word_id = [forced_trg_batch_elem[pos] for forced_trg_batch_elem in forced_trg_ids]
      for batch_i in range(batch_size):
        if done[batch_i]:
          batch_word = vocabs.Vocab.ES
          batch_score = 0.0
        else:
          batch_word = batched_word_id[batch_i]
          batch_score = logsoft[batch_word,batch_i]
          if self._terminate_rnn(batch_i=batch_i, pos=pos, batched_word_id=batched_word_id):
            done[batch_i] = True
        score[batch_i].append(batch_score)
        words[batch_i].append(batch_word)
      if all(done):
        break
    for batch_i in range(batch_size):
      batch_elem_score = sum(score[batch_i])
      outputs.append(sent.SimpleSentence(words=words[batch_i],
                                         idx=src[batch_i].idx,
                                         vocab=getattr(self.trg_reader, "vocab", None),
                                         score=batch_elem_score,
                                         output_procs=self.trg_reader.output_procs))
      if self.compute_report:
        if batch_size > 1:
          cur_attentions = [x[:, :, batch_i] for x in attentions]
        else:
          cur_attentions = attentions
        attention = np.concatenate(cur_attentions, axis=1)
        self.report_sent_info({"attentions": attention,
                               "src": src[batch_i],
                               "output": outputs[-1]})


    return outputs

  def calc_loss_one_step(self,
                         dec_state: SymmetricDecoderState,
                         batch_size: int,
                         prev_ref_action: Optional[batchers.Batch],
                         mode: str,
                         ref_action: Optional[batchers.Batch] = None) -> Optional[dy.Expression]:
    outputs = self._unfold_one_step(dec_state=dec_state, batch_size=batch_size, mode=mode, mask=None, cur_step=None,
                                    prev_ref_action=prev_ref_action)
    if mode not in ["teacher", "split"]:
      dec_state.out_prob = self.scorer.calc_probs(outputs)
    if ref_action:
      word_loss = self.scorer.calc_loss(outputs, batchers.mark_as_batch(ref_action))
      return word_loss
    else: return None

  def generate_one_step(self,
                        dec_state: SymmetricDecoderState,
                        batch_size: int,
                        mode: str,
                        mask: Optional[batchers.Mask] = None,
                        cur_step: Optional[int] = None,
                        prev_ref_action: Optional[batchers.Batch] = None) -> dy.Expression:
    outputs = self._unfold_one_step(dec_state=dec_state, batch_size=batch_size, mode=mode, mask=mask, cur_step=cur_step,
                                    prev_ref_action=prev_ref_action)
    if mode in ["expected", "argmax", "argmax_st", "teacher", "split"]:
      dec_state.out_prob = self.scorer.calc_probs(outputs)
    return self.scorer.calc_scores(outputs)

  def _unfold_one_step(self, dec_state, batch_size, mode, mask, cur_step, prev_ref_action):
    lstm_input = self._choose_rnn_input(dec_state=dec_state, batch_size=batch_size, prev_ref_action=prev_ref_action,
                                        mode=mode)
    new_rnn_state = dec_state.rnn_state.add_input(lstm_input) if lstm_input else dec_state.rnn_state
    if not (mask is None or np.isclose(np.sum(mask.np_arr[:, cur_step:cur_step + 1]), 0.0)):
      new_rnn_state._h = list(new_rnn_state._h)
      new_rnn_state._c = list(new_rnn_state._c)
      for layer_i in range(len(new_rnn_state.h())):
        new_rnn_state._h[layer_i] = mask.cmult_by_timestep_expr(new_rnn_state.h()[layer_i], cur_step, True) \
                                    + mask.cmult_by_timestep_expr(dec_state.rnn_state.h()[layer_i], cur_step, False)
        new_rnn_state._c[layer_i] = mask.cmult_by_timestep_expr(new_rnn_state._c[layer_i], cur_step, True) \
                                    + mask.cmult_by_timestep_expr(dec_state.rnn_state._c[layer_i], cur_step, False)
      new_rnn_state._h = tuple(new_rnn_state._h)
      new_rnn_state._c = tuple(new_rnn_state._c)
    dec_state.rnn_state = new_rnn_state
    rnn_output = dec_state.rnn_state.output()
    dec_state.context = self.attender.calc_context(rnn_output)
    if self.dropout_dec_state and self.train:
      rnn_output = dy.dropout_batch(rnn_output, self.dropout_dec_state)
    outputs = self.transform.transform(dy.concatenate([rnn_output, dec_state.context]))
    return outputs

  def _terminate_rnn(self, batch_i, pos, batched_word_id):
    if self.unfold_until == "supervised":
      return pos >= self.cur_src.batches[1][batch_i].len_unpadded()
    elif self.unfold_until == "eos":
      return batched_word_id[batch_i] == vocabs.Vocab.ES
    else:
      raise ValueError(f"unknown value '{self.unfold_until}' for unfold_until")

  def _choose_rnn_input(self, dec_state, batch_size, prev_ref_action, mode):
    hidden_size = dec_state.context.dim()[0][0]
    vocab_size = self.trg_embedder.vocab_size
    if mode == "context":
      context_vec = dy.reshape(dec_state.context, (hidden_size,), batch_size=batch_size)
      ret = context_vec
    elif dec_state.out_prob is None and prev_ref_action is None:
      ret = None
    elif mode == "expected":
      ret = dy.reshape(dec_state.out_prob, (1,vocab_size), batch_size=batch_size) * dy.parameter(self.trg_embedder.embeddings)
      ret = dy.reshape(ret, (hidden_size,), batch_size=batch_size)
    elif mode in ["argmax", "argmax_st"]:
      gradient_mode = "zero_gradient" if mode == "argmax" else "straight_through_gradient"
      argmax = dy.reshape(dy.argmax(dec_state.out_prob, gradient_mode=gradient_mode), (1, vocab_size),
                          batch_size=batch_size)
      ret = argmax * dy.parameter(self.trg_embedder.embeddings)
      ret = dy.reshape(ret, (hidden_size,), batch_size=batch_size)
    elif mode in ["teacher", "split"]:
      do_sample = self.train and dec_state.out_prob and self.sampling_prob > 0.0 and random.random() < self.sampling_prob
      if not do_sample:
        ret = self.trg_embedder.embed(prev_ref_action)
      else: # do sample
        sampled_vals = []
        npval = dec_state.out_prob.npvalue()
        for bi in range(batch_size):
          npval_bi = npval[:, bi] if batch_size>1 else npval
          sampled_vals.append(
            np.random.choice(vocab_size,
                             p=npval_bi / np.sum(npval_bi)))
        idxs = ([], [])
        for batch_i in range(batch_size):
          idxs[0].append(sampled_vals[batch_i])
          idxs[1].append(batch_i)
        argmax = dy.sparse_inputTensor(idxs, values=np.ones(batch_size), shape=(vocab_size, batch_size), batched=True, )
        argmax = dy.reshape(argmax, (1, vocab_size), batch_size=batch_size)
        ret = argmax * dy.parameter(self.trg_embedder.embeddings)
        ret = dy.reshape(ret, (hidden_size,), batch_size=batch_size)
    else:
      raise ValueError(f"unknown value for mode: {mode}")
    if ret is not None: self._chosen_rnn_inputs.append(ret)
    return ret

  @events.handle_xnmt_event
  def on_calc_additional_loss(self, *args, **kwargs):
    loss_dict = {}
    if self.transducer_loss and self.transducer_losses:
      loss_expr = dy.esum(self.transducer_losses)
      loss_dict["symm_transd_loss"] = loss_expr
    if self.split_reg_penalty_expr is not None:
      loss_dict["symm_transd_reg_penalty"] = self.split_reg_penalty_expr
    if len(loss_dict)==0: return None
    else: return losses.FactoredLossExpr(loss_dict)
