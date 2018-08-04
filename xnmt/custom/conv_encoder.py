import math
from collections.abc import Sequence
from typing import Tuple

import dynet as dy

from xnmt import events, expression_seqs, norms, param_collections, param_initializers
from xnmt.transducers import base as transducers
from xnmt.custom import regularizers
from xnmt.persistence import Serializable, bare, serializable_init, Ref


class StridedConvSeqTransducer(transducers.SeqTransducer, Serializable):
  """
  Implements several (possibly strided) CNN layers.

  No padding is performed, thus layer size will shrink even with striding turned off.

  Args:
    weight_noise: apply Gaussian noise of given standard deviation to weights (training time only)
    layers: encoder depth
    input_dim: size of the inputs, before factoring out the channels.
                      We will end up with a convolutional layer of size num_steps X input_dim/chn_dim X chn_dim
    chn_dim: channel input dimension
    num_filters: channel output dimension
    stride: tuple, downsample via striding
    batch_norm : apply batch normalization before the nonlinearity. Normalization is performed over batch, time, and frequency dimensions (and not over the channel dimension).
    nonlinearity: e.g. "rectify" / "silu" / ...  / None
    pre_activation: If True, please BN + nonlinearity before CNN
    output_tensor: True -> output is a expression sequence holding a 3d-tensor (including channel dimension), in transposed form (time is first dimension)
                          False -> output is a expression sequence holding a list of flat vector expressions (frequency and channel dimensions are merged)
    transpose:
    param_init: how to initialize filter parameters

  """
  yaml_tag = u'!StridedConvSeqTransducer'

  @events.register_xnmt_handler
  @serializable_init
  def __init__(self,
               weight_noise: float = Ref("exp_global.weight_noise", default=0.0),
               layers: int = 1,
               input_dim: int = 120,
               chn_dim: int = 3,
               num_filters: int = 32,
               stride: Tuple[int] = (2, 2),
               batch_norm: bool = False,
               nonlinearity: str = "rectify",
               pre_activation: bool = False,
               output_tensor: bool = False,
               transpose: bool = True,
               param_init: param_initializers.ParamInitializer = Ref("exp_global.param_init",
                                                                  default=bare(param_initializers.GlorotInitializer))):
    assert layers > 0
    if input_dim % chn_dim != 0:
      raise ValueError(f"StridedConvEncoder requires input_dim mod chn_dim == 0, got: {input_dim} and {chn_dim}")

    param_col = param_collections.ParamManager.my_params(self)
    self.layers = layers
    self.chn_dim = chn_dim
    self.freq_dim = input_dim / chn_dim
    self.num_filters = num_filters
    self.filter_size_time = 3
    self.filter_size_freq = 3
    self.stride = stride
    self.output_transposed_tensor = output_tensor
    self.nonlinearity = nonlinearity
    self.pre_activation = pre_activation

    self.use_bn = batch_norm
    self.train = True
    self.transpose = transpose
    self.weight_noise = regularizers.WeightNoise(weight_noise)

    self.bn_layers = []
    self.filters_layers = []
    for layer_i in range(layers):
      filter_dim = (self.filter_size_time,
                    self.filter_size_freq,
                    self.chn_dim if layer_i == 0 else self.num_filters,
                    self.num_filters)
      filters = param_col.add_parameters(dim=filter_dim,
                                         init=param_init[layer_i].initializer(filter_dim) if isinstance(param_init,
                                                                                                        Sequence) else param_init.initializer(
                                           filter_dim))
      if self.use_bn:
        self.bn_layers.append(
          norms.BatchNorm(param_col, (self.chn_dim if self.pre_activation else self.num_filters), 3))
      self.filters_layers.append(filters)

  @events.handle_xnmt_event
  def on_set_train(self, val):
    self.train = val

  def get_output_dim(self):
    conv_dim = self.freq_dim
    for layer_i in range(self.layers):
      conv_dim = int(
        math.ceil(float(conv_dim - self.filter_size_freq + 1) / float(self.get_stride_for_layer(layer_i)[1])))
    return conv_dim * self.num_filters

  def get_stride_for_layer(self, layer_i):
    if type(self.stride) == tuple:
      return self.stride
    else:
      assert type(self.stride) == list
      return self.stride[layer_i]

  def get_output_len(self, input_len):
    conv_dim = input_len
    for layer_i in range(self.layers):
      conv_dim = int(
        math.ceil(float(conv_dim - self.filter_size_time + 1) / float(self.get_stride_for_layer(layer_i)[0])))
    return conv_dim

  def pad(self, expr, pad_size):
    assert pad_size >= 0
    if pad_size == 0:
      return expr
    return dy.concatenate([expr, dy.zeroes((pad_size, self.freq_dim * self.chn_dim), batch_size=expr.dim()[
      1])])  # TODO: replicate last frame instead of padding zeros

  def apply_nonlinearity(self, nonlinearity, expr):
    if nonlinearity == "rectify":
      return dy.rectify(expr)
    if nonlinearity == "silu":
      return dy.silu(expr)
    elif nonlinearity is not None:
      raise RuntimeError("unknown nonlinearity: %s" % nonlinearity)
    return expr

  def __call__(self, es):
    es_expr = es.as_tensor()

    sent_len = es_expr.dim()[0][0]
    batch_size = es_expr.dim()[1]

    # convolutions won't work if sentence length is too short; pad if necessary
    pad_size = 0
    while self.get_output_len(sent_len + pad_size) < self.filter_size_time:
      pad_size += 1
    es_expr = self.pad(es_expr, pad_size)
    sent_len += pad_size

    # loop over layers
    if es_expr.dim() == ((sent_len, self.freq_dim, self.chn_dim), batch_size):
      es_chn = es_expr
    else:
      es_chn = dy.reshape(es_expr, (sent_len, self.freq_dim, self.chn_dim), batch_size=batch_size)
    cnn_layer = es_chn
    mask_out = None
    for layer_i in range(len(self.filters_layers)):
      cnn_filter = self.weight_noise(self.filters_layers[layer_i], self.train)

      if not self.pre_activation:
        cnn_layer = dy.conv2d(cnn_layer, cnn_filter, stride=self.get_stride_for_layer(layer_i), is_valid=True)

      if self.use_bn:
        mask_out = None if es.mask is None else es.mask.lin_subsampled(trg_len=cnn_layer.dim()[0][0])
        cnn_layer = self.bn_layers[layer_i](cnn_layer, train=self.train, mask=mask_out)

      cnn_layer = self.apply_nonlinearity(self.nonlinearity, cnn_layer)
      self.last_output.append(cnn_layer)

      if self.pre_activation:
        cnn_layer = dy.conv2d(cnn_layer, cnn_filter, stride=self.get_stride_for_layer(layer_i), is_valid=True)

    mask_out = None if es.mask is None else es.mask.lin_subsampled(trg_len=cnn_layer.dim()[0][0])
    if self.output_transposed_tensor:
      return expression_seqs.ExpressionSequence(expr_transposed_tensor=cnn_layer, mask=mask_out)
    else:
      cnn_out = dy.reshape(cnn_layer, (cnn_layer.dim()[0][0], cnn_layer.dim()[0][1] * cnn_layer.dim()[0][2]),
                           batch_size=batch_size)
      es_list = [cnn_out[i] for i in range(cnn_out.dim()[0][0])]
      return expression_seqs.ExpressionSequence(expr_list=es_list, mask=mask_out)


class PoolingConvSeqTransducer(transducers.SeqTransducer, Serializable):
  """
  Implements several CNN layers, with strided max pooling interspersed.
  """
  yaml_tag = u'!PoolingConvSeqTransducer'

  @serializable_init
  def __init__(self, input_dim, pooling=[None, (1, 1)], chn_dim=3, num_filters=32,
               output_tensor=False, nonlinearity="rectify", param_init=None, bias_init=None):
    """
    :param layers: encoder depth
    :param input_dim: size of the inputs, before factoring out the channels.
                      We will end up with a convolutional layer of size num_steps X input_dim/chn_dim X chn_dim 
    :param model
    :param chn_dim: channel dimension
    :param num_filters
    :param output_tensor: if set, the output is directly given as a 3d-tensor, rather than converted to a list of vector expressions
    :param nonlinearity: "rely" / None
    """
    raise Exception("TODO: buggy, needs proper transposing")
    assert input_dim % chn_dim == 0

    model = exp_global.dynet_param_collection.param_col
    self.layers = len(pooling)
    assert self.layers > 0
    self.chn_dim = chn_dim
    self.freq_dim = input_dim / chn_dim
    self.num_filters = num_filters
    self.filter_size_time = 3
    self.filter_size_freq = 3
    self.pooling = pooling
    self.output_tensor = output_tensor
    self.nonlinearity = nonlinearity
    param_init = param_init or exp_global.param_init
    bias_init = bias_init or exp_global.bias_init

    self.bn_layers = []
    self.filters_layers = []
    self.bn_alt_layers = []
    self.filters_alt_layers = []
    for layer_i in range(self.layers):
      dim_f = (self.filter_size_time,
               self.filter_size_freq,
               self.chn_dim if layer_i == 0 else self.num_filters,
               self.num_filters)
      param_init_i = param_init[i] if isinstance(param_init, Sequence) else param_init
      filters = model.add_parameters(dim=dim_f,
                                     init=param_init_i.initializer(dim_f))
      self.filters_layers.append(filters)

  def get_output_dim(self):
    conv_dim = self.freq_dim
    for layer_i in range(self.layers):
      conv_dim = int(
        math.ceil(float(conv_dim - self.filter_size_freq + 1) / float(self.get_stride_for_layer(layer_i)[1])))
    return conv_dim * self.num_filters

  def get_stride_for_layer(self, layer_i):
    if self.pooling[layer_i]:
      return self.pooling[layer_i]
    else:
      return (1, 1)

  def get_output_len(self, input_len):
    conv_dim = input_len
    for layer_i in range(self.layers):
      conv_dim = int(
        math.ceil(float(conv_dim - self.filter_size_time + 1) / float(self.get_stride_for_layer(layer_i)[0])))
    return conv_dim

  def __call__(self, es):
    es_expr = es.as_tensor()

    sent_len = es_expr.dim()[0][0]
    batch_size = es_expr.dim()[1]

    # convolutions won't work if sentence length is too short; pad if necessary
    pad_size = 0
    while self.get_output_len(sent_len + pad_size) < self.filter_size_time:
      pad_size += 1
    if pad_size > 0:
      es_expr = dy.concatenate(
        [es_expr, dy.zeroes((pad_size, self.freq_dim * self.chn_dim), batch_size=es_expr.dim()[1])])
      sent_len += pad_size

    if es_expr.dim() == ((sent_len, self.freq_dim, self.chn_dim), batch_size):
      es_chn = es_expr
    else:
      es_chn = dy.reshape(es_expr, (sent_len, self.freq_dim, self.chn_dim), batch_size=batch_size)
    cnn_layer = es_chn

    # loop over layers
    for layer_i in range(len(self.filters_layers)):
      cnn_layer_prev = cnn_layer
      filters = self.filters_layers[layer_i]

      # convolution
      cnn_layer = dy.conv2d(cnn_layer, dy.parameter(filters), stride=(1, 1), is_valid=True)

      # non-linearity
      if self.nonlinearity == "rectify":
        cnn_layer = dy.rectify(cnn_layer)
      elif self.nonlinearity == "silu":
        cnn_layer = dy.silu(cnn_layer)
      elif self.nonlinearity is not None:
        raise RuntimeError("unknown nonlinearity: %s" % self.nonlinearity)

      # max pooling
      if self.pooling[layer_i]:
        cnn_layer = dy.maxpooling2d(cnn_layer, (3, 3), stride=self.pooling[layer_i], is_valid=True)

    mask_out = es.mask.lin_subsampled(trg_len=cnn_layer.dim()[0][0])
    if self.output_tensor:
      return expression_seqs.ExpressionSequence(tensor_expr=cnn_layer, mask=mask_out)
    else:
      cnn_out = dy.reshape(cnn_layer, (cnn_layer.dim()[0][0], cnn_layer.dim()[0][1] * cnn_layer.dim()[0][2]),
                           batch_size=batch_size)
      es_list = [cnn_out[i] for i in range(cnn_out.dim()[0][0])]
      return expression_seqs.ExpressionSequence(list_expr=es_list, mask=mask_out)


class ConvStrideTransducer(transducers.SeqTransducer):
  def __init__(self, chn_dim, stride=(1, 1), margin=(0, 0)):
    self.chn_dim = chn_dim
    self.stride = stride
    self.margin = margin

  def __call__(self, expr):
    return dy.strided_select(expr, [self.margin[0], expr.dim()[0][0] - self.margin[0], self.stride[0],
                                    self.margin[1], expr.dim()[0][1] - self.margin[1], self.stride[1],
                                    0, expr.dim()[0][2], 1,
                                    0, expr.dim()[1], 1])
