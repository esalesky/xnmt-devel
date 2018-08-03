import dynet as dy
import numpy as np

from xnmt import expression_seqs, param_collections, param_initializers
from xnmt.transducers import base as transducers
from xnmt.persistence import Serializable, serializable_init, Ref, bare

class ConvLSTMSeqTransducer(transducers.SeqTransducer, Serializable):
  """
  This is a convolutional LSTM implementation using a single bidirectional layer.

  Follows Shi et al, 2015: Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting
  https://arxiv.org/pdf/1506.04214.pdf

  Output is an expression sequence of dimensions (hidden_dim, sent_len)

  Args:
    input_dim: input size, product of hidden and channel size
    chn_dim: number of input channel
    num_filters: number of output filters
    param_init: parameter initializer for filters
    bias_init: parameter initializer for biases
  """
  yaml_tag = "!ConvLSTMSeqTransducer"

  @serializable_init
  def __init__(self,
               input_dim: int,
               chn_dim: int = 3,
               num_filters: int = 32,
               param_init: param_initializers.ParamInitializer = Ref("exp_global.param_init", default=bare(param_initializers.GlorotInitializer)),
               bias_init: param_initializers.ParamInitializer = Ref("exp_global.bias_init", default=bare(param_initializers.ZeroInitializer))):
    model = param_collections.ParamManager.my_params(self)
    if input_dim % chn_dim != 0:
      raise RuntimeError("input_dim must be divisible by chn_dim")
    self.input_dim = input_dim

    self.chn_dim = chn_dim
    self.freq_dim = input_dim / chn_dim
    self.num_filters = num_filters
    self.filter_size_time = 1
    self.filter_size_freq = 3

    self.params = {}
    for direction in ["fwd", "bwd"]:
      dim_x = (self.filter_size_time, self.filter_size_freq,
               self.chn_dim, self.num_filters * 4)
      self.params["x2all_" + direction] = \
        model.add_parameters(dim=dim_x,
                             init=param_init.initializer(dim_x, num_shared=4))
      dim_h = (self.filter_size_time, self.filter_size_freq,
               self.num_filters, self.num_filters * 4)
      self.params["h2all_" + direction] = \
        model.add_parameters(dim=dim_h,
                             init=param_init.initializer(dim_h, num_shared=4))
      dim_b = (self.num_filters * 4,)
      self.params["b_" + direction] = \
        model.add_parameters(dim=dim_b, init=bias_init.initializer(dim_b, num_shared=4))

  def transduce(self, es: expression_seqs.ExpressionSequence) -> expression_seqs.ExpressionSequence:
    mask = es.mask
    sent_len = len(es)
    es_expr = es.as_transposed_tensor()
    batch_size = es_expr.dim()[1]

    es_chn = dy.reshape(es_expr, (sent_len, self.freq_dim, self.chn_dim), batch_size=batch_size)

    h_out = {}
    for direction in ["fwd", "bwd"]:
      # input convolutions
      gates_xt_bias = dy.conv2d_bias(es_chn, dy.parameter(self.params["x2all_" + direction]),
                                     dy.parameter(self.params["b_" + direction]), stride=(1, 1), is_valid=False)
      gates_xt_bias_list = [dy.pick_range(gates_xt_bias, i, i + 1) for i in range(sent_len)]

      h = []
      c = []
      for input_pos in range(sent_len):
        directional_pos = input_pos if direction == "fwd" else sent_len - input_pos - 1
        gates_t = gates_xt_bias_list[directional_pos]
        if input_pos > 0:
          # recurrent convolutions
          gates_h_t = dy.conv2d(h[-1], dy.parameter(self.params["h2all_" + direction]), stride=(1, 1), is_valid=False)
          gates_t += gates_h_t

        # standard LSTM logic
        if len(c) == 0:
          c_tm1 = dy.zeros((self.freq_dim * self.num_filters,), batch_size=batch_size)
        else:
          c_tm1 = c[-1]
        gates_t_reshaped = dy.reshape(gates_t, (4 * self.freq_dim * self.num_filters,), batch_size=batch_size)
        c_t = dy.reshape(dy.vanilla_lstm_c(c_tm1, gates_t_reshaped), (self.freq_dim * self.num_filters,),
                         batch_size=batch_size)
        h_t = dy.vanilla_lstm_h(c_t, gates_t_reshaped)
        h_t = dy.reshape(h_t, (1, self.freq_dim, self.num_filters,), batch_size=batch_size)

        if mask is None or np.isclose(np.sum(mask.np_arr[:, input_pos:input_pos + 1]), 0.0):
          c.append(c_t)
          h.append(h_t)
        else:
          c.append(
            mask.cmult_by_timestep_expr(c_t, input_pos, True) + mask.cmult_by_timestep_expr(c[-1], input_pos, False))
          h.append(
            mask.cmult_by_timestep_expr(h_t, input_pos, True) + mask.cmult_by_timestep_expr(h[-1], input_pos, False))

      h_out[direction] = h
    ret_expr = []
    for state_i in range(len(h_out["fwd"])):
      state_fwd = h_out["fwd"][state_i]
      state_bwd = h_out["bwd"][-1 - state_i]
      output_dim = (state_fwd.dim()[0][1] * state_fwd.dim()[0][2],)
      fwd_reshape = dy.reshape(state_fwd, output_dim, batch_size=batch_size)
      bwd_reshape = dy.reshape(state_bwd, output_dim, batch_size=batch_size)
      ret_expr.append(dy.concatenate([fwd_reshape, bwd_reshape], d=0 if self.reshape_output else 2))
    return expression_seqs.ExpressionSequence(expr_list=ret_expr, mask=mask)

  # TODO: implement get_final_states()