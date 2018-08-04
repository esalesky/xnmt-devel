from typing import Optional

import dynet as dy

from xnmt.transducers import base as transducers
from xnmt.modelparts import transforms
from xnmt import events, expression_seqs, losses, param_collections, param_initializers, vocabs
from xnmt.persistence import Serializable, serializable_init, bare, Ref

class SemiDiscreteSeqTransducer(transducers.SeqTransducer, Serializable):
  """
  This implements a semi-discrete transducer as E*(softmax(Wx+b)).

  Args:
    input_dim (int): input dimension
    softmax_dim (int): softmax dimension (intuitively: number of discrete states)
    output_dim (int): hidden dimension
    dropout (float): dropout probability
    gumbel (bool): whether to sample from Gumble softmax
    param_init (ParamInitializer): how to initialize weight matrices
    bias_init (ParamInitializer): how to initialize bias vectors
  """
  yaml_tag = '!SemiDiscreteSeqTransducer'

  @events.register_xnmt_handler
  @serializable_init
  def __init__(self,
               input_dim=Ref("exp_global.default_layer_dim"),
               softmax_dim=Ref("exp_global.default_layer_dim"),
               output_dim=Ref("exp_global.default_layer_dim"),
               dropout = Ref("exp_global.dropout", default=0.0),
               residual=False,
               linear_layer = None,
               vocab = None,
               gumbel = False,
               param_init=Ref("exp_global.param_init", default=bare(param_initializers.GlorotInitializer)),
               bias_init=Ref("exp_global.bias_init", default=bare(param_initializers.ZeroInitializer))):
    param_col = param_collections.ParamManager.my_params(self)
    self.input_dim = input_dim
    if vocab:
      softmax_dim = len(vocab)
    self.softmax_dim = softmax_dim
    self.output_dim = output_dim
    self.dropout_rate = dropout
    self.residual = residual
    self.gumbel = gumbel
    if self.residual: assert self.input_dim == self.output_dim

    self.linear_layer = self.add_serializable_component("linear_layer",
                                                        linear_layer,
                                                        lambda: transforms.Linear(input_dim=self.softmax_dim,
                                                                                  output_dim=self.output_dim,
                                                                                  bias=False,
                                                                                  param_init=param_init,
                                                                                  bias_init=bias_init))

    # self.p_W = param_col.add_parameters(dim=(softmax_dim, input_dim), init=param_init.initializer((softmax_dim, input_dim)))
    # self.p_b = param_col.add_parameters(dim=(softmax_dim), init=bias_init.initializer((softmax_dim,)))
    self.p_E = param_col.add_parameters(dim=(output_dim, softmax_dim), init=param_init.initializer((output_dim, softmax_dim)))

  @events.handle_xnmt_event
  def on_set_train(self, val):
    self.train = val

  def get_final_states(self):
    return self._final_states

  def transduce(self, expr_seq: 'expression_seqs.ExpressionSequence'):
    """
    transduce the sequence

    Args:
      expr_seq: expression sequence
    Returns:
      expression sequence
    """
    batch_size = expr_seq[0].dim()[1]
    seq_len = len(expr_seq)

    output_exps = []
    for pos_i in range(seq_len):
      input_i = expr_seq[pos_i]
      affine = self.linear_layer(input_i)
      # affine = dy.affine_transform([dy.parameter(self.p_b), dy.parameter(self.p_W), input_i])
      if self.train and self.dropout_rate:
        affine = dy.dropout(affine, self.dropout_rate)
      if self.gumbel:
        affine = affine + dy.random_gumbel(dim=affine.dim()[0],batch_size=batch_size)
      softmax_out = dy.softmax(affine)
      # embedded = self.emb_layer(softmax_out)
      embedded = dy.parameter(self.p_E) * softmax_out
      if self.residual:
        embedded = embedded + input_i
      output_exps.append(embedded)

    self._final_states = [transducers.FinalTransducerState(main_expr=embedded)]

    return expression_seqs.ExpressionSequence(expr_list = output_exps, mask=expr_seq.mask)

class EntropyLossSeqTransducer(transducers.SeqTransducer, Serializable):
  """
  This uses a sub-object to transduce the sequence, but adds a loss term that encourages discrete representations.

  Args:
    transducer: transducer
    input_dim: input dimension
    softmax_dim: softmax dimension (intuitively: number of discrete states)
    layer_dim: hidden dimension
    linear_layer:
    vocab:
    scale: scaling factor to apply to loss
    mode: ``entropy`` | ``max``
    param_init: how to initialize weight matrices
    bias_init (ParamInitializer): how to initialize bias vectors
  """
  yaml_tag = '!EntropyLossSeqTransducer'

  @events.register_xnmt_handler
  @serializable_init
  def __init__(self,
               transducer:transducers.SeqTransducer,
               input_dim:int=Ref("exp_global.default_layer_dim"),
               softmax_dim:int=Ref("exp_global.default_layer_dim"),
               layer_dim:int=Ref("exp_global.default_layer_dim"),
               linear_layer:transforms.Linear = None,
               vocab:Optional[vocabs.Vocab] = None,
               scale:float = 1.0,
               mode:str="entropy",
               param_init:param_initializers.ParamInitializer=Ref("exp_global.param_init", default=bare(param_initializers.GlorotInitializer)),
               bias_init:param_initializers.ParamInitializer=Ref("exp_global.bias_init", default=bare(param_initializers.ZeroInitializer))):
    self.transducer = transducer
    self.input_dim = input_dim
    if vocab:
      softmax_dim = len(vocab)
    self.softmax_dim = softmax_dim
    self.layer_dim = layer_dim
    self.scale = scale
    self.mode = mode

    self.linear_layer = self.add_serializable_component("linear_layer",
                                                        linear_layer,
                                                        lambda: transforms.Linear(input_dim=self.softmax_dim,
                                                                                  output_dim=self.layer_dim,
                                                                                  bias=False,
                                                                                  param_init=param_init,
                                                                                  bias_init=bias_init))


  def get_final_states(self):
    return self.transducer.get_final_states()

  def transduce(self, expr_seq):
    self.last_output = self.transducer.transduce(expr_seq)
    return self.last_output

  @events.handle_xnmt_event
  def on_calc_additional_loss(self, *args, **kwargs):
    seq_len = len(self.last_output)

    loss_expr = 0
    for pos_i in range(seq_len):
      input_i = self.last_output[pos_i]
      affine = self.linear_layer(input_i)
      softmax_out = dy.softmax(affine)
      if self.mode == "entropy":
        loss_expr = loss_expr - dy.sum_dim(dy.cmult(dy.log(softmax_out),
                                                    softmax_out),
                                           d=[0])
      elif self.mode == "max":
        loss_expr = loss_expr - dy.log(dy.max_dim(softmax_out))
      else:
        raise ValueError(f"unknown mode {self.mode}")
    # loss_expr = loss_expr * (self.scale / seq_len)
    loss_expr = loss_expr * self.scale

    return losses.FactoredLossExpr({"enc_entropy" : loss_expr })

