import dynet as dy

from xnmt.models import base as models
from xnmt import event_trigger, losses, loss_calculators
from xnmt.persistence import Serializable, serializable_init

class DualEncoderSimilarity(models.ConditionedModel, Serializable):

  yaml_tag = "!DualEncoderSimilarity"

  @serializable_init
  def __init__(self, src_reader, trg_reader, src_embedder, src_encoder, trg_embedder, trg_encoder):
    super().__init__(src_reader, trg_reader)
    self.src_embedder = src_embedder
    self.src_encoder = src_encoder
    self.trg_embedder = trg_embedder
    self.trg_encoder = trg_encoder

  def calc_loss(self, src, trg, loss_calculator):

    event_trigger.start_sent(src)

    src_embeddings = self.src_embedder.embed_sent(src)
    src_encodings = self.src_encoder(src_embeddings)

    trg_embeddings = self.trg_embedder.embed_sent(trg)
    trg_encodings = self.trg_encoder(trg_embeddings)

    model_loss = losses.FactoredLossExpr()
    model_loss.add_loss("dist", loss_calculator(src_encodings, trg_encodings))

    return model_loss


class DistLoss(Serializable, loss_calculators.LossCalculator):
  yaml_tag = '!DistLoss'

  @serializable_init
  def __init__(self, dist_op="squared_norm"):
    if callable(dist_op):
      self.dist_op = dist_op
    else:
      self.dist_op = getattr(dy, dist_op)

  def __call__(self, src_encodings, trg_encodings):
    src_avg = dy.sum_dim(src_encodings.as_tensor(), [1])/(src_encodings.as_tensor().dim()[0][1])
    trg_avg = dy.sum_dim(trg_encodings.as_tensor(), [1])/(trg_encodings.as_tensor().dim()[0][1])
    return self.dist_op(src_avg - trg_avg)
