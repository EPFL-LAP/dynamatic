import json
from pprint import pprint

# TODO add as argument
with open("experimental/tools/elastic-miter-generator/out/comp/elastic-miter-config.json") as json_data:
  config = json.load(json_data)

# pprint(config)

# TODO add as argument
buffer_size = 2


print(
"""#include "model.smv"

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
#endif // BOOL_INPUT

MODULE main
"""
)

for i, arg in enumerate(config["arguments"]):
  print(f"VAR seq_generator{i} : bool_input(miter.{arg}_ready, {buffer_size});")
  # print(f"VAR seq_generator{i} : entry_0_1(miter.{arg}_ready);")

print()


def create_miter_call(arg, res):
  miter = "VAR miter : elastic_miter("
  miter += ", ".join([f"seq_generator{i}.dataOut0, seq_generator{i}.valid0" for i, _ in enumerate(arg)])
  miter += ", "
  miter += ", ".join([f"sink{i}.ready0" for i, _ in enumerate(res)])
  miter += ");\n"
  print(miter)


create_miter_call(config["arguments"], config["results"])

# TODO
print("-- TODO make sure we have sink_1_0")
for i, res in enumerate(config["results"]):
  print(f"VAR sink{i} : sink_1_0(miter.{res}_out, miter.{res}_valid);")

print()

# TODO remove
exit()


def create_eq_properties(results):
  for result in results:
    prop = f"AG (miter.{result}_valid -> miter.{result})"
    print('CTLSPEC ' + prop)
    prop = f"AG miter.{result}_valid -> miter.{result}"
    print('CTLSPEC ' + prop)



create_eq_properties(config["results"])


output_prop = ""
# TODO clean this up or change json
for buf in sum(config["output_buffers"], []):
  output_prop += f"miter.{buf}.num = 0 & "

output_prop = output_prop[:-3]
# print(output_prop)

input_prop = ""
for lhs_buffer, rhs_buffer in config["input_buffers"]:
  input_prop += f"miter.{lhs_buffer}.num = miter.{rhs_buffer}.num & "

input_prop = input_prop[:-3]
# print(input_prop)


final_buffer_prop = "AF (AG (" + input_prop + " & " + output_prop + "))"

print('CTLSPEC ' + final_buffer_prop)

final_buffer_prop = "AF AG " + input_prop + " & " + output_prop

print('CTLSPEC ' + final_buffer_prop)

