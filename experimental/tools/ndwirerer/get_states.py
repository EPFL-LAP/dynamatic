def get_state_set(file):
  with open(file) as f:
    text = f.readlines()

  current_state = 0
  states = {}
  for line in text:
    line = line.strip()
    if "-------" in line:
      if current_state > 0:
        states[current_state] = "\n".join(states[current_state])
      current_state += 1
      states[current_state] = []
      continue
    if current_state == 0:
      continue
    if not line.startswith("miter."):
      continue
    states[current_state].append(line)

  del states[current_state]
  return set(states.values())

# TODO cache inf state, maybe needs to be class to be clean
def get_states(inf_file, fin_file):
  
  set_fin = get_state_set(fin_file)

  set_inf = get_state_set(inf_file)

  diff = set_inf - set_fin

  return len(diff)
