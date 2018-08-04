from xnmt import batcher, input_reader, translator
from xnmt.persistence import Serializable, serializable_init

class EnsembleInputReader(input_reader.InputReader, Serializable):
  yaml_tag = "!EnsembleInputReader"
  @serializable_init
  def __init__(self, input_reader1, input_reader2):
    # TODO: generalize to n input readers
    self.input_reader1 = input_reader1
    self.input_reader2 = input_reader2
    self.vocab = self.input_reader1.vocab if hasattr(self.input_reader1, "vocab") else  self.input_reader2.vocab # TODO: hack to make EnsembleTranslator happy

  def read_sents(self, filename, filter_ids=None):
    for s in zip(self.input_reader1.read_sents(filename[0]), self.input_reader2.read_sents(filename[1])):
      yield translator.EnsembleListDelegate([s[0],s[1]])

  def count_sents(self, filename):
    cnt = self.input_reader1.count_sents()
    assert cnt == self.input_reader2.count_sents()
    return cnt
