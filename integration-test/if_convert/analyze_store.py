import ast
import subprocess

kernel_name = "if_convert"
unit_name = "store1"

base_path = f"duv/{kernel_name}_wrapped/{unit_name}"

file = "test.log"

result = subprocess.run([
    "wlf2log", "-l", base_path, "-o", file, "out/sim/HLS_VERIFY/vsim.wlf"
])


def gen_vector(name, size):
  return [f"{name}({i})" for i in range(size)]


ports = ["addrIn_valid", "addrIn_ready", *gen_vector("addrIn", 5)]

paths = {}
signal_ids = {}
vals = {}

for port in ports:
  paths[port] = f"{base_path}/{port}"
  signal_ids[port] = 0
  vals[port] = 0

prev_vals = vals.copy()


def get_vector_value(name, size):
  val = 0
  for i in range(size):
    val += int(vals[f"{name}({i})"]) << i
  return val


with open(file, "r") as f:
  lines = f.readlines()
  for line in lines:
    tokens = line.strip().split(" ")
    if not tokens:
      continue
    identifier = tokens[0]
    if identifier == "D":
      # Signal Association
      current_path = tokens[1]
      for key, path in paths.items():
        if path == current_path:
          wireId = int(tokens[2])
          signal_ids[key] = wireId
    elif identifier == "S":
      # Change Wire State
      wireId = int(tokens[1])
      for key, id in signal_ids.items():
        if id == wireId:
          val = ast.literal_eval(tokens[2])[0]
          vals[key] = val
    elif identifier == "T":
      if prev_vals != vals:
        if vals["addrIn_ready"] == "1" and vals["addrIn_valid"] == "1":
          time = tokens[1]
          print(time, get_vector_value("addrIn", 5))
      prev_vals = vals.copy()
