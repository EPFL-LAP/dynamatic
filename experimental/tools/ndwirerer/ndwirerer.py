import os
import subprocess

from create_state_wrapper import create_state_wrapper
from get_states import get_states

# TODO clean this up, argparse
OUT_DIR = "experimental/tools/ndwirerer/out"

COMP_DIR = os.path.join(OUT_DIR, "comp")
DOT = os.path.join(COMP_DIR, "miter.dot")

REWRITES = "experimental/test/tools/elastic-miter-generator/rewrites"
MOD = "z"


def exit_on_fail(ret, message):
  if ret != 0:
    print(message)
    exit(1)

def run_nuXmv(cmd, stdout):
  cmd = ["nuXmv", "-source", cmd]
  with open(stdout, "w") as result_file:
    process = subprocess.run(cmd, stdout=result_file)
  return process.returncode


def handshake2smv(mlir, png=False):
  # Convert to DOT
  cmd = ["bin/export-dot", mlir, "--edge-style=spline"]
  with open(DOT, "w") as dot_file:
    process = subprocess.run(cmd, stdout=dot_file)
  exit_on_fail(process.returncode, "Failed to convert to dot")

  if png:
    # Convert to PNG
    cmd = ["dot", "-Tpng", DOT]
    with open(os.path.join(COMP_DIR, "visual.png"), "wb") as png_file:
      process = subprocess.run(cmd, stdout=png_file)
    exit_on_fail(process.returncode, "Failed to convert to PNG")

  # Convert to SMV
  cmd = ["python3", "../dot2smv/dot2smv", DOT]
  process = subprocess.run(cmd)
  exit_on_fail(process.returncode, "Failed to convert to SMV")

  # Move the SMV file
  src_smv = os.path.join(os.path.dirname(DOT), "model.smv")
  dst_smv = os.path.join(os.path.dirname(DOT), f"{MOD}_lhs.smv")
  os.rename(src_smv, dst_smv)

  return dst_smv

    

# TODO do this for LHS/RHS
def main():

  dst_smv = handshake2smv(f"{REWRITES}/{MOD}_lhs.mlir", True)

  # TODO we should probably just pass the filename 
  # Create state wrapper for infinite tokens
  inf_wrapper = create_state_wrapper(dst_smv, f"{REWRITES}/{MOD}_lhs.mlir", inf=True)
  with open("experimental/tools/ndwirerer/out/comp/main_inf.smv", "w") as f:
    f.write(inf_wrapper)
    
  # Run nuXmv for infinite tokens
  ret = run_nuXmv(os.path.join(COMP_DIR, "prove_inf.cmd"), os.path.join(OUT_DIR, f"inf_states.txt"))
  exit_on_fail(ret, "Failed to analyze reachable states with infinite tokens.")


  N = 1
  while True:
    print(f"Checking {N} tokens.")

    fin_wrapper = create_state_wrapper(dst_smv, f"{REWRITES}/{MOD}_lhs.mlir", N=N)
    with open(f"experimental/tools/ndwirerer/out/comp/main_{N}.smv", "w") as f:
      f.write(fin_wrapper)

    # TODO automatically create cmd file
    ret = run_nuXmv(os.path.join(COMP_DIR, f"prove_{N}.cmd"), os.path.join(OUT_DIR, f"{N}_states.txt"))
    exit_on_fail(ret, f"Failed to analyze reachable states with {N} tokens.")

    # Check state differences
    nr_of_differences = get_states(os.path.join(OUT_DIR, f"inf_states.txt"), os.path.join(OUT_DIR, f"{N}_states.txt"))

    if nr_of_differences != 0:
      N += 1
    else:
      print(N)
      break


if(__name__ == "__main__"):
  main()