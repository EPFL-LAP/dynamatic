

import argparse


parser = argparse.ArgumentParser(
                    prog='StateComparer',
                    description='TODO What the program does',
                    epilog='TODO Text at the bottom of help')

parser.add_argument("inf")
parser.add_argument("fin")



# TODO use argparser
with open("experimental/tools/ndwirerer/out/inf_states.txt") as f:
  inf = f.readlines()

with open("experimental/tools/ndwirerer/out/2_states.txt") as f:
  three = f.readlines()

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
states_3 = {}
for line in three:
  # print(line)
  line = line.strip()
  if "-------" in line:
    if state > 0:
      states_3[state] = "\n".join(states_3[state])
    state += 1
    states_3[state] = []
    continue
  if state == 0:
    continue
  if not line.startswith("miter."):
    continue
  states_3[state].append(line)

del states_3[state]

set_inf = set(states_inf.values())
set_3 = set(states_3.values())


diff = set_inf - set_3

for state in diff:
  print("--------------")
  for number, text in states_inf.items():
    if text == state:
        print(number)
  print(state) 


print(len(diff))

exit(len(diff))
