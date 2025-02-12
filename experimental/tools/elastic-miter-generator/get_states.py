def get_state_set(file):
  with open(file) as f:
    text = f.readlines()

  current_state = None
  states = set()
  for line in text:
    line = line.strip()
    if "-------" in line:
      if current_state is not None:
        states.add(current_state)
      current_state = ""
      continue
    if current_state is None:
      continue
    if not line.startswith("miter."):
      continue
    current_state += line + "\n"

  return states

def get_states(inf_file, fin_file):
  
  set_fin = get_state_set(fin_file)

  set_inf = get_state_set(inf_file)

  diff = set_inf - set_fin

  return len(diff)
