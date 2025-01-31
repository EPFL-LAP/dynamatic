import json
from pprint import pprint

# TODO add as argument
with open("experimental/tools/elastic-miter-generator/out/comp/elastic-miter-config.json") as json_data:
  config = json.load(json_data)

# pprint(config)


def create_eq_properties(eq_ops):
  for op in eq_ops:
    prop = f"AG ({op}.valid0 -> {op}.dataOut0)"
    print('add_property -c -p "' + prop + '"')



create_eq_properties(config["eq"])


output_prop = ""
# TODO clean this up or change json
for buf in sum(config["output_buffers"], []):
  output_prop += f"{buf}.num = 0 & "

output_prop = output_prop[:-3]
# print(output_prop)

input_prop = ""
for lhs_buffer, rhs_buffer in config["input_buffers"]:
  input_prop += f"{lhs_buffer}.num = {rhs_buffer}.num & "

input_prop = input_prop[:-3]
# print(input_prop)


final_buffer_prop = "AF AG " + input_prop + " & " + output_prop

print('add_property -c -p "' + final_buffer_prop + '"')

