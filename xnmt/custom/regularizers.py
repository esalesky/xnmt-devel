import dynet as dy

class WeightNoise(object):
  def __init__(self, std):
    self.std = std
  def __call__(self, p, train=True):
    """
    Args:
      DyNet parameter (not expression)
      train: only apply noise if True
    Return:
      DyNet expression with weight noise applied if self.std > 0
    """
    p_expr = dy.parameter(p)
    if self.std > 0.0 and train:
      p_expr = dy.noise(p_expr, self.std)
    return p_expr
