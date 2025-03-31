import os
from generators.handshake.speculation.speculator import generate_speculator
from generators.handshake.speculation.spec_save_commit import generate_spec_save_commit
from generators.handshake.merge import generate_merge

base_path = os.path.dirname(os.path.abspath(__file__))

speculator_file = os.path.join(base_path, "speculator.vhd")
with open(speculator_file, "w") as f:
  f.write(generate_speculator("speculator", {
      "bitwidth": 1,
      "fifo_depth": 4,
      "extra_signals": {"spec": 1}
  }))

save_commit_file = os.path.join(base_path, "spec_save_commit.vhd")
with open(save_commit_file, "w") as f:
  f.write(generate_spec_save_commit("spec_save_commit", {
      "bitwidth": 32,
      "fifo_depth": 4,
      "extra_signals": {"spec": 1}
  }))

merge_file = os.path.join(base_path, "merge.vhd")
with open(merge_file, "w") as f:
  f.write(generate_merge("merge", {
      "size": 2,
      "bitwidth": 3
  }))
