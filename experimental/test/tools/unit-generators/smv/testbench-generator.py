import sys


def generate_testbench(n):
  print(f""" #include "./module.smv"

MODULE main
  VAR
  {"\n  ".join([f"v{i} : boolean;" for i in range(n)])}
  dut : test_module({", ".join([f"v{i}" for i in range(n)])});
""")


if __name__ == "__main__":
  if len(sys.argv) != 2:
    print("Usage: python testbench-generator.py <input-signals>")
  else:
    try:
      n_sig = int(sys.argv[1])
      generate_testbench(n_sig)
    except ValueError:
      print("Please provide a valid integer for N.")
