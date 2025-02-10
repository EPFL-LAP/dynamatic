
def get_states(inf_file, fin_file):


  with open(inf_file) as f:
    inf = f.readlines()

  with open(fin_file) as f:
    fin = f.readlines()

  state = 0
  states_inf = {}
  for line in inf:
    # print(line)
    line = line.strip()
    if "-------" in line:
      if state > 0:
        states_inf[state] = "\n".join(states_inf[state])
      state += 1
      states_inf[state] = []
      continue
    if state == 0:
      continue
    if not line.startswith("miter."):
      continue
    states_inf[state].append(line)

  del states_inf[state]


  state = 0
  states_fin = {}
  for line in fin:
    # print(line)
    line = line.strip()
    if "-------" in line:
      if state > 0:
        states_fin[state] = "\n".join(states_fin[state])
      state += 1
      states_fin[state] = []
      continue
    if state == 0:
      continue
    if not line.startswith("miter."):
      continue
    states_fin[state].append(line)

  del states_fin[state]

  set_inf = set(states_inf.values())
  set_fin = set(states_fin.values())


  diff = set_inf - set_fin

  # for state in diff:
  #   print("--------------")
  #   for number, text in states_inf.items():
  #     if text == state:
  #         print(number)
  #   print(state) 


  print(len(diff))

  return len(diff)
