from typing import Sequence

import dynet as dy
import numpy as np

from xnmt import events, expression_seqs, param_collections, param_initializers
from xnmt.transducers import base as transducers
from xnmt.persistence import Serializable, serializable_init, Ref, bare


class QLSTMSeqTransducer(transducers.SeqTransducer, Serializable):
  """
  This implements the quasi-recurrent neural network with input, output, and forget gate.
  https://arxiv.org/abs/1611.01576
  """
  yaml_tag = u'!QLSTMSeqTransducer'

  @events.register_xnmt_handler
  @serializable_init
  def __init__(self,
               input_dim=Ref("exp_global.default_layer_dim"),
               hidden_dim=Ref("exp_global.default_layer_dim"),
               dropout = Ref("exp_global.dropout", default=0.0),
               filter_width=2,
               stride=1,
               param_init=Ref("exp_global.param_init", default=bare(param_initializers.GlorotInitializer)),
               bias_init=Ref("exp_global.bias_init", default=bare(param_initializers.ZeroInitializer))):
    model = param_collections.ParamManager.my_params(self)
    self.hidden_dim = hidden_dim
    self.dropout = dropout
    self.input_dim = input_dim
    self.stride = stride

    dim_f = (filter_width, 1, input_dim, hidden_dim * 3)
    self.p_f = model.add_parameters(dim=dim_f, init=param_init.initializer(dim_f, num_shared=3)) # f, o, z
    dim_b = (hidden_dim * 3,)
    self.p_b = model.add_parameters(dim=dim_b, init=bias_init.initializer(dim_b, num_shared=3))

  @events.handle_xnmt_event
  def on_set_train(self, val):
    self.train = val

  @events.handle_xnmt_event
  def on_start_sent(self, src):
    self._final_states = None

  def get_final_states(self):
    return self._final_states

  def transduce(self, expr_seq: expression_seqs.ExpressionSequence) -> expression_seqs.ExpressionSequence:
    """
    transduce the sequence, applying masks if given (masked timesteps simply copy previous h / c)

    Args:
      expr_seq: expression sequence (will be accessed via tensor_expr)
    Return:
      expression sequence
    """

    if isinstance(expr_seq, list):
      mask_out = expr_seq[0].mask
      seq_len = len(expr_seq[0])
      batch_size = expr_seq[0].dim()[1]
      tensors = [e.as_tensor() for e in expr_seq]
      input_tensor = dy.reshape(dy.concatenate(tensors), (seq_len, 1, self.input_dim), batch_size = batch_size)
    else:
      mask_out = expr_seq.mask
      seq_len = len(expr_seq)
      batch_size = expr_seq.dim()[1]
      input_tensor = dy.reshape(dy.transpose(expr_seq.as_tensor()), (seq_len, 1, self.input_dim), batch_size = batch_size)

    if self.dropout > 0.0 and self.train:
      input_tensor = dy.dropout(input_tensor, self.dropout)

    proj_inp = dy.conv2d_bias(input_tensor, dy.parameter(self.p_f), dy.parameter(self.p_b), stride=(self.stride ,1), is_valid=False)
    reduced_seq_len = proj_inp.dim()[0][0]
    proj_inp = dy.transpose(dy.reshape(proj_inp, (reduced_seq_len, self.hidden_dim *3), batch_size = batch_size))
    # proj_inp dims: (hidden, 1, seq_len), batch_size
    if self.stride > 1 and mask_out is not None:
      mask_out = mask_out.lin_subsampled(trg_len=reduced_seq_len)

    h = [dy.zeroes(dim=(self.hidden_dim ,1), batch_size=batch_size)]
    c = [dy.zeroes(dim=(self.hidden_dim ,1), batch_size=batch_size)]
    for t in range(reduced_seq_len):
      f_t = dy.logistic(dy.strided_select(proj_inp, [], [0, t], [self.hidden_dim, t+ 1]))
      o_t = dy.logistic(dy.strided_select(proj_inp, [], [self.hidden_dim, t], [self.hidden_dim * 2, t + 1]))
      z_t = dy.tanh(dy.strided_select(proj_inp, [], [self.hidden_dim * 2, t], [self.hidden_dim * 3, t + 1]))

      if self.dropout > 0.0 and self.train:
        retention_rate = 1.0 - self.dropout
        dropout_mask = dy.random_bernoulli((self.hidden_dim, 1), retention_rate, batch_size=batch_size)
        f_t = 1.0 - dy.cmult(dropout_mask,
                             1.0 - f_t)  # TODO: would be easy to make a zoneout dynet operation to save memory

      i_t = 1.0 - f_t

      if t == 0:
        c_t = dy.cmult(i_t, z_t)
      else:
        c_t = dy.cmult(f_t, c[-1]) + dy.cmult(i_t, z_t)
      h_t = dy.cmult(o_t, c_t)  # note: LSTM would use dy.tanh(c_t) instead of c_t
      if mask_out is None or np.isclose(np.sum(mask_out.np_arr[:, t:t + 1]), 0.0):
        c.append(c_t)
        h.append(h_t)
      else:
        c.append(mask_out.cmult_by_timestep_expr(c_t, t, True) + mask_out.cmult_by_timestep_expr(c[-1], t, False))
        h.append(mask_out.cmult_by_timestep_expr(h_t, t, True) + mask_out.cmult_by_timestep_expr(h[-1], t, False))

    self._final_states = [transducers.FinalTransducerState(dy.reshape(h[-1], (self.hidden_dim,), batch_size=batch_size), \
                                                           dy.reshape(c[-1], (self.hidden_dim,),
                                                                      batch_size=batch_size))]
    return expression_seqs.ExpressionSequence(expr_list=h[1:], mask=mask_out)


class BiQLSTMSeqTransducer(transducers.SeqTransducer, Serializable):
  yaml_tag = u'!BiQLSTMSeqTransducer'

  @events.register_xnmt_handler
  @serializable_init
  def __init__(self,
               layers,
               input_dim=Ref("exp_global.default_layer_dim"),
               hidden_dim=Ref("exp_global.default_layer_dim"),
               dropout=Ref("exp_global.dropout", default=0.0),
               stride=1,
               filter_width=2,
               param_init=Ref("exp_global.param_init", default=bare(param_initializers.GlorotInitializer)),
               bias_init=Ref("exp_global.bias_init", default=bare(param_initializers.ZeroInitializer)),
               forward_layers=None,
               backward_layers=None):
    self.num_layers = layers
    self.hidden_dim = hidden_dim
    assert hidden_dim % 2 == 0
    self.forward_layers = self.add_serializable_component("forward_layers", forward_layers,
                                                          lambda: self.init_layers(input_dim, hidden_dim, dropout,
                                                                                   stride, filter_width, param_init,
                                                                                   bias_init))
    self.backward_layers = self.add_serializable_component("backward_layers", backward_layers,
                                                           lambda: self.init_layers(input_dim, hidden_dim, dropout,
                                                                                    stride, filter_width, param_init,
                                                                                    bias_init))

  def init_layers(self, input_dim, hidden_dim, dropout, stride, filter_width, param_init, bias_init):
    layers = [QLSTMSeqTransducer(input_dim=input_dim, hidden_dim=hidden_dim / 2, dropout=dropout, stride=stride,
                                 filter_width=filter_width,
                                 param_init=param_init[0] if isinstance(param_init, Sequence) else param_init,
                                 bias_init=bias_init[0] if isinstance(bias_init, Sequence) else bias_init)]
    layers += [QLSTMSeqTransducer(input_dim=hidden_dim, hidden_dim=hidden_dim / 2, dropout=dropout, stride=stride,
                                  filter_width=filter_width,
                                  param_init=param_init[i] if isinstance(param_init, Sequence) else param_init,
                                  bias_init=bias_init[i] if isinstance(bias_init, Sequence) else bias_init) for i in
               range(1, layers)]
    return layers

  @events.handle_xnmt_event
  def on_start_sent(self, src):
    self._final_states = None

  def get_final_states(self):
    return self._final_states

  def transduce(self, es):
    mask = es.mask
    # first layer
    forward_es = self.forward_layers[0].transduce(es)
    rev_backward_es = self.backward_layers[0].transduce(expression_seqs.ReversedExpressionSequence(es))

    # TODO: concat input of each layer to its output; or, maybe just add standard residual connections
    for layer_i in range(1, len(self.forward_layers)):
      new_forward_es = self.forward_layers[layer_i].transduce([forward_es, expression_seqs.ReversedExpressionSequence(rev_backward_es)])
      mask_out = mask
      if mask_out is not None and new_forward_es.mask.np_arr.shape != mask_out.np_arr.shape:
        mask_out = mask_out.lin_subsampled(trg_len=len(new_forward_es))
      rev_backward_es = expression_seqs.ExpressionSequence(self.backward_layers[layer_i].transduce(
        [expression_seqs.ReversedExpressionSequence(forward_es), rev_backward_es]).as_list(), mask=mask_out)
      forward_es = new_forward_es

    self._final_states = [
      transducers.FinalTransducerState(dy.concatenate([self.forward_layers[layer_i].get_final_states()[0].main_expr(),
                                                       self.backward_layers[layer_i].get_final_states()[
                                                         0].main_expr()]),
                                       dy.concatenate([self.forward_layers[layer_i].get_final_states()[0].cell_expr(),
                                                       self.backward_layers[layer_i].get_final_states()[
                                                         0].cell_expr()])) \
      for layer_i in range(len(self.forward_layers))]
    mask_out = mask
    if mask_out is not None and forward_es.mask.np_arr.shape != mask_out.np_arr.shape:
      mask_out = mask_out.lin_subsampled(trg_len=len(forward_es))
    return expression_seqs.ExpressionSequence(
      expr_list=[dy.concatenate([forward_es[i], rev_backward_es[-i - 1]]) for i in range(len(forward_es))],
      mask=mask_out)
