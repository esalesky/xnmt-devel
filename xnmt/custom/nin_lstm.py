import math
from collections.abc import Sequence

from xnmt import expression_seqs, param_initializers
from xnmt.transducers import base as transducers, recurrent

from xnmt.transducers import network_in_network
from xnmt.events import register_xnmt_handler, handle_xnmt_event
from xnmt.persistence import serializable_init, Serializable, Ref, bare


class ZhangSeqTransducer(transducers.ModularSeqTransducer, Serializable):
  yaml_tag = "!ZhangSeqTransducer"
  @serializable_init
  def __init__(self,
               input_dim, hidden_dim,
               dropout=Ref("exp_global.dropout", default=0.0),
               bottom_layer=None, top_layer=None):
    self.bottom_layer = self.add_serializable_component("bottom_layer", bottom_layer,
                                                        lambda: NinBiLSTMTransducer(layers=2,
                                                                                    input_dim=input_dim,
                                                                                    hidden_dim=hidden_dim,
                                                                                    stride=2,
                                                                                    dropout=dropout))
    self.top_layer = self.add_serializable_component("top_layer", top_layer,
                                                     lambda: recurrent.BiLSTMSeqTransducer(layers=1,
                                                                                           input_dim=hidden_dim,
                                                                                           hidden_dim=hidden_dim,
                                                                                           dropout=dropout))

    self.modules = [self.bottom_layer, self.top_layer]

class NinBiLSTMTransducer(transducers.SeqTransducer, Serializable):
  """
  Builder for NiN-interleaved RNNs that delegates to regular RNNs and wires them together.
  See http://iamaaditya.github.io/2016/03/one-by-one-convolution/
  and https://arxiv.org/pdf/1610.03022.pdf

  Args:
    layers: depth of the network
    input_dim: size of the inputs of bottom layer
    hidden_dim: size of the outputs (and intermediate layer representations)
    stride: in projection layer, concatenate n frames and thus use the projection for downsampling
    dropout: LSTM dropout
    builder_layers: set automatically
    nin_layers: set automatically
    param_init_lstm: a ParamInitializer or list of ParamInitializer objects
                specifying how to initialize weight matrices. If a list is given, each entry denotes one layer.
    bias_init_lstm: a ParamInitializer or list of ParamInitializer objects
               specifying how to initialize bias vectors. If a list is given, each entry denotes one layer.
    param_init_nin: a ParamInitializer or list of ParamInitializer objects
                specifying how to initialize weight matrices. If a list is given, each entry denotes one layer.
  """
  yaml_tag = '!NinBiLSTMTransducer'
  @register_xnmt_handler
  @serializable_init
  def __init__(self,
               layers,
               input_dim=Ref("exp_global.default_layer_dim"),
               hidden_dim=Ref("exp_global.default_layer_dim"),
               stride=1,
               dropout=Ref("exp_global.dropout", default=0.0),
               lstm_layers=None,
               nin_layers=None,
               param_init_lstm=Ref("exp_global.param_init", default=bare(param_initializers.GlorotInitializer)),
               bias_init_lstm=Ref("exp_global.bias_init", default=bare(param_initializers.ZeroInitializer)),
               param_init_nin=Ref("exp_global.param_init", default=bare(param_initializers.GlorotInitializer))):
    """
    """
    assert layers > 0
    assert hidden_dim % 2 == 0

    self.lstm_layers = []
    self.hidden_dim = hidden_dim
    self.stride=stride

    self.lstm_layers = self.add_serializable_component("lstm_layers", lstm_layers,
                                                       lambda: self.init_lstm_layers(layers, input_dim,
                                                                                     hidden_dim, dropout,
                                                                                     param_init_lstm,
                                                                                     bias_init_lstm))

    self.nin_layers = self.add_serializable_component("nin_layers", nin_layers,
                                                      lambda: self.init_nin_layers(layers, hidden_dim,
                                                                                   param_init_nin))

  def init_lstm_layers(self, layers, input_dim, hidden_dim, dropout, param_init_lstm, bias_init_lstm):
    lstm_layers = []
    f = recurrent.UniLSTMSeqTransducer(input_dim=input_dim,
                                       hidden_dim=hidden_dim // 2,
                                       dropout=dropout,
                                       param_init=param_init_lstm[0] if isinstance(param_init_lstm,
                                                                                   Sequence) else param_init_lstm,
                                       bias_init=bias_init_lstm[0] if isinstance(bias_init_lstm,
                                                                                 Sequence) else bias_init_lstm)
    b = recurrent.UniLSTMSeqTransducer(input_dim=input_dim,
                                       hidden_dim=hidden_dim // 2,
                                       dropout=dropout,
                                       param_init=param_init_lstm[0] if isinstance(param_init_lstm,
                                                                                   Sequence) else param_init_lstm,
                                       bias_init=bias_init_lstm[0] if isinstance(bias_init_lstm,
                                                                                 Sequence) else bias_init_lstm)
    lstm_layers.append([f, b])
    for i in range(1, layers):
      f = recurrent.UniLSTMSeqTransducer(input_dim=hidden_dim,
                                         hidden_dim=hidden_dim // 2,
                                         dropout=dropout,
                                         param_init=param_init_lstm[i] if isinstance(param_init_lstm,
                                                                                     Sequence) else param_init_lstm,
                                         bias_init=bias_init_lstm[i] if isinstance(bias_init_lstm,
                                                                                   Sequence) else bias_init_lstm)
      b = recurrent.UniLSTMSeqTransducer(input_dim=hidden_dim,
                                         hidden_dim=hidden_dim // 2,
                                         dropout=dropout,
                                         param_init=param_init_lstm[i] if isinstance(param_init_lstm,
                                                                                     Sequence) else param_init_lstm,
                                         bias_init=bias_init_lstm[i] if isinstance(bias_init_lstm,
                                                                                   Sequence) else bias_init_lstm)
      lstm_layers.append([f, b])
    return lstm_layers

  def init_nin_layers(self, layers, hidden_dim, param_init_nin):
    nin_layers = []
    for i in range(layers):
      nin_layer = network_in_network.NinSeqTransducer(input_dim=hidden_dim // 2,
                                                      hidden_dim=hidden_dim,
                                                      downsample_by=2 * self.stride,
                                                      param_init=param_init_nin[i] if isinstance(param_init_nin,
                                                                                                 Sequence) else param_init_nin)
      nin_layers.append(nin_layer)
    return nin_layers

  @handle_xnmt_event
  def on_start_sent(self, *args, **kwargs):
    self._final_states = None

  def get_final_states(self):
    assert self._final_states is not None, "transduce() must be invoked before get_final_states()"
    return self._final_states
        
  def transduce(self, es: expression_seqs.ExpressionSequence) -> expression_seqs.ExpressionSequence:

    for layer_i, (fb, bb) in enumerate(self.lstm_layers):
      fs = fb.transduce(es)
      bs = bb.transduce(expression_seqs.ReversedExpressionSequence(es))
      interleaved = []

      if es.mask is None: mask = None
      else:
        mask = es.mask.lin_subsampled(0.5) # upsample the mask to encompass interleaved fwd / bwd expressions

      for pos in range(len(fs)):
        interleaved.append(fs[pos])
        interleaved.append(bs[-pos-1])
      
      projected = expression_seqs.ExpressionSequence(expr_list=interleaved, mask=mask)
      projected = self.nin_layers[layer_i].transduce(projected)
      assert math.ceil(len(es) / float(self.stride))==len(projected), \
        f"mismatched len(es)=={len(es)}, stride=={self.stride}, len(projected)=={len(projected)}"
      es = projected

    self._final_states = [transducers.FinalTransducerState(projected[-1])]
    return projected





