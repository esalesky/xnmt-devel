import argparse
import sys

import numpy as np
import h5py

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--input", help="Path to npz file to read from")
  parser.add_argument("--output", help="Path to h5 to write out")
  args = parser.parse_args()


  npz_file = np.load(args.input, mmap_mode="r")
  npz_keys = sorted(npz_file.files, key=lambda x: int(x.split('_')[-1]))

  with h5py.File(args.output, "w") as hf:

    for sent_no, key in enumerate(npz_keys):
      features = npz_file[key]
      assert key.startswith("arr_")
      hf.create_dataset(str(int(key[4:])), data=features)

  npz_file.close()

if __name__ == "__main__":
  sys.exit(main())
