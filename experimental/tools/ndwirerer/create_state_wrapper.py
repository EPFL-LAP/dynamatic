import json
from pprint import pprint
import argparse


parser = argparse.ArgumentParser(
                    prog='StateWrapper',
                    description='TODO What the program does',
                    epilog='TODO Text at the bottom of help')

parser.add_argument("--json")
group = parser.add_mutually_exclusive_group()
group.add_argument("-N", type=int)
group.add_argument("--inf", action="store_true")


args = parser.parse_args()

# TODO add as argument
with open(args.json) as json_data:
  config = json.load(json_data)

# pprint(config)

inf = args.inf

# TODO add as argument
buffer_size = args.N


# TODO proper include
print(
"""#include "c_lhs.smv"

#ifndef BOOL_INPUT
#define BOOL_INPUT
MODULE bool_input(nReady0, max_tokens)
  VAR dataOut0 : boolean;
  VAR counter : 0..31;
  ASSIGN
  init(counter) := 0;
  next(counter) := case
    nReady0 & counter < max_tokens : counter + 1;
    TRUE : counter;
  esac;
  
  -- bool_input persistent
  ASSIGN
  next(dataOut0) := case 
    valid0 & !nReady0 : dataOut0;
    TRUE : {TRUE, FALSE};
  esac;
  DEFINE valid0 := counter < max_tokens;
MODULE bool_input_inf(nReady0)
  VAR dataOut0 : boolean;
  
  -- bool_input persistent
  ASSIGN
  next(dataOut0) := case 
    valid0 & !nReady0 : dataOut0;
    TRUE : {TRUE, FALSE};
  esac;
  DEFINE valid0 := TRUE;
#endif // BOOL_INPUT

MODULE main
"""
)


# TODO remove
config["arguments"] = ["D", "C"]
config["results"] = ["A", "B"]

for i, arg in enumerate(config["arguments"]):
  if inf:
    print(f"VAR seq_generator{i} : bool_input_inf(in_ndw{i}.ready0);")
  else:
    print(f"VAR seq_generator{i} : bool_input(in_ndw{i}.ready0, {buffer_size});")

  print(f"VAR in_ndw{i} : ndw_1_1(seq_generator{i}.dataOut0, seq_generator{i}.valid0, miter.{arg}_ready);")
  # print(f"VAR seq_generator{i} : entry_0_1(miter.{arg}_ready);")

print()


def create_miter_call(arg, res):
  miter = "VAR miter : elastic_miter("
  miter += ", ".join([f"in_ndw{i}.dataOut0, in_ndw{i}.valid0" for i, _ in enumerate(arg)])
  miter += ", "
  miter += ", ".join([f"out_ndw{i}.ready0" for i, _ in enumerate(res)])
  miter += ");\n"
  print(miter)


create_miter_call(config["arguments"], config["results"])

# TODO
print("-- TODO make sure we have sink_1_0")
for i, res in enumerate(config["results"]):
  print(f"VAR out_ndw{i} : ndw_1_1(miter.{res}_out, miter.{res}_valid, sink{i}.ready0);")
  print(f"VAR sink{i} : sink_1_0(out_ndw{i}.dataOut0, out_ndw{i}.valid0);")

print()
