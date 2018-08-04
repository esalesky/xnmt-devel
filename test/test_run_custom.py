import unittest
import os, shutil

import xnmt.xnmt_run_experiments as run
import xnmt.events

class TestRunningConfig(unittest.TestCase):

  def setUp(self):
    xnmt.events.clear()

  def test_bow(self):
    run.main(["test/custom/bow.yaml"])

  def test_dual_encoder(self):
    run.main(["test/custom/dual_encoder.yaml"])

  def test_speech_semi_disc(self):
    run.main(["test/custom/speech_semi_disc.yaml"])

  def test_symmetric(self):
    run.main(["test/custom/symmetric.yaml"])


  def tearDown(self):
    try:
      if os.path.isdir("test/tmp"):
        shutil.rmtree("test/tmp")
    except:
      pass

if __name__ == "__main__":
  unittest.main()
