import unittest
import os, shutil

from test.utils import has_cython
import xnmt.xnmt_run_experiments as run
import xnmt.events

class TestRunningConfig(unittest.TestCase):

  def setUp(self):
    xnmt.events.clear()

  def test_bow(self):
    run.main(["test/custom/bow.yaml"])


  def tearDown(self):
    try:
      if os.path.isdir("test/tmp"):
        shutil.rmtree("test/tmp")
    except:
      pass

if __name__ == "__main__":
  unittest.main()
