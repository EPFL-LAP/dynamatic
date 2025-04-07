import ast

file = "test.log"
kernel_name = "if_convert"
unit_name = "spec_save_commit0"

base_path = f"duv/{kernel_name}_wrapped/{unit_name}"

paths = {
    "head": f"{base_path}/Head",
    "curr": f"{base_path}/Curr",
    "tail": f"{base_path}/Tail"
}

signal_ids = {"head": 0, "curr": 0, "tail": 0}
pointers = {"head": 0, "curr": 0, "tail": 0}

prev_pointers = pointers.copy()


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
          pointers[key] = val
    elif identifier == "T":
      if pointers != prev_pointers:
        # New Timestep
        print("Time: ", tokens[1])
        print(pointers)
        prev_pointers = pointers.copy()
