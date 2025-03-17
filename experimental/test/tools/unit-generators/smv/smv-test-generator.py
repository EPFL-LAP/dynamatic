import argparse
import sys
import ast
from utils import *


def parse_parameters(param_list):
  try:
    param_dict = {}
    if param_list is not None:
      for pair in param_list:
        key, value = pair.split("=")
        if value != "":
          param_dict[key.strip()] = ast.literal_eval(value.strip())
    return param_dict
  except ValueError:
    raise ValueError("Invalid parameter format. Use key=value key=value,...\n")


def smv_type_from_bitwidth(bitwidth):
  if bitwidth == 1:
    return "boolean"
  return f"unsigned word [{bitwidth}]"


def create_port_types(ports):
  port_types = {}
  for port in ports:
    if port["direction"] == "in":
      if port["count"] == 1:
        if port["bitwidth"] > 0:
          port_types[port["name"]] = smv_type_from_bitwidth(port["bitwidth"])
        port_types[port["name"] + "_valid"] = "boolean"
      else:
        for i in range(port["count"]):
          if port["bitwidth"] > 0:
            port_types[port["name"] + "_" +
                       str(i)] = smv_type_from_bitwidth(port["bitwidth"])
          port_types[port["name"] + "_" + str(i) + "_valid"] = "boolean"
    else:
      if port["count"] == 1:
        port_types[port["name"] + "_ready"] = "boolean"
      else:
        for i in range(port["count"]):
          port_types[port["name"] + "_" + str(i) + "_ready"] = "boolean"
  return port_types


def create_input_map(ports):
  port_map = {}
  for port in ports:
    if port["direction"] == "in":
      if port["count"] == 1:
        if port["bitwidth"] > 0:
          port_map[port["name"]] = [port["name"]]
        port_map[port["name"] + "_valid"] = [port["name"] + "_valid"]
      else:
        if port["bitwidth"] > 0:
          port_map[port["name"]] = []
          for i in range(port["count"]):
            port_map[port["name"]].append(port["name"] + "_" + str(i))
        port_map[port["name"] + "_valid"] = []
        for i in range(port["count"]):
          port_map[port["name"] +
                   "_valid"].append(port["name"] + "_" + str(i) + "_valid")
    else:
      if port["count"] == 1:
        port_map[port["name"] + "_ready"] = [port["name"] + "_ready"]
      else:
        port_map[port["name"] + "_ready"] = []
        for i in range(port["count"]):
          port_map[port["name"] +
                   "_ready"].append(port["name"] + "_" + str(i) + "_ready")
  return port_map


def create_output_map(ports):
  port_map = {}
  for port in ports:
    if port["direction"] == "out":
      if port["count"] == 1:
        if port["bitwidth"] > 0:
          port_map[port["name"]] = [port["name"]]
        port_map[port["name"] + "_valid"] = [port["name"] + "_valid"]
      else:
        if port["bitwidth"] > 0:
          port_map[port["name"]] = []
          for i in range(port["count"]):
            port_map[port["name"]].append(port["name"] + "_" + str(i))
        port_map[port["name"] + "_valid"] = []
        for i in range(port["count"]):
          port_map[port["name"] +
                   "_valid"].append(port["name"] + "_" + str(i) + "_valid")
    else:
      if port["count"] == 1:
        port_map[port["name"] + "_ready"] = [port["name"] + "_ready"]
      else:
        port_map[port["name"] + "_ready"] = []
        for i in range(port["count"]):
          port_map[port["name"] +
                   "_ready"].append(port["name"] + "_" + str(i) + "_ready")
  return port_map


def needs_cast(name, ports):
  extracted_name = re.sub(r'(_valid|_ready)$', '', name)
  port = next((d for d in ports if d.get("name") == extracted_name), None)
  if port is not None:
    if "_valid" in name or "_ready" in name or port["bitwidth"] == 1:
      return True
  return False


def generate_internal_signals(port_types):
  return "\n  ".join([f"VAR internal_{port_name} : {port_type};" for port_name, port_type in port_types.items()])


def generate_mut_arguments(port_types):
  return ", ".join([f"internal_{port_name}" for port_name in port_types.keys()])


def generate_assignments(port_map, ports):
  return "\n  ".join([f"ASSIGN golden_model._{gm_port} := {" :: ".join([f"word1(internal_{mut_port})" for mut_port in reversed(mut_port_list)]) if needs_cast(gm_port, ports) else " :: ".join([f"internal_{mut_port}" for mut_port in reversed(mut_port_list)])};" for gm_port, mut_port_list in port_map.items()])


def generate_properties(port_map, ports):
  return "\n  ".join([f"INVARSPEC NAME {gm_port}_check := golden_model._{gm_port} = {" :: ".join([f"word1(model_under_test.{mut_port})" for mut_port in reversed(mut_port_list)]) if needs_cast(gm_port, ports) else " :: ".join([f"model_under_test.{mut_port}" for mut_port in reversed(mut_port_list)])};" for gm_port, mut_port_list in port_map.items()])


def generate_test_bench(name, mod_type, parameters):
  port_types = create_port_types(parameters["ports"])
  input_map = create_input_map(parameters["ports"])
  output_map = create_output_map(parameters["ports"])

  return f"""
#include "{name}.smv"
#include "golden_model.smv"


MODULE main
  {generate_internal_signals(port_types)}

  VAR model_under_test : {name}({generate_mut_arguments(port_types)});
  VAR golden_model : _{name}();

  -- input assignments
  ASSIGN golden_model._rst := 0ub_0;
  {generate_assignments(input_map, parameters["ports"])}

  -- properties
  {generate_properties(output_map, parameters["ports"])}
"""


def main():
  parser = argparse.ArgumentParser(description="SMV Test Generator Script")
  parser.add_argument(
      "-n", "--name", required=True, help="Name of the generated module"
  )
  parser.add_argument(
      "-o", "--output", required=True, help="Name of the output file"
  )
  parser.add_argument(
      "-t", "--type", required=True, help="Type of the generated module"
  )
  parser.add_argument(
      "--abstract-data",
      action="store_true",
      help="Enable abstract data mode",
  )
  parser.add_argument(
      "-p",
      "--parameters",
      required=False,
      nargs="*",
      help="Set of parameters in key=value key=value format",
  )

  args = parser.parse_args()

  try:
    parameters = parse_parameters(args.parameters)
  except ValueError as e:
    sys.stderr.write(f"Error parsing parameters: {e}")
    sys.exit(1)

  # Printing parameters for diagnostic purposes
  header = f"-- {args.name} : {args.type}({args.parameters})\n\n"

  with open(args.output, 'w') as file:
    print(header + generate_test_bench(args.name, args.type, parameters), file=file)


if __name__ == "__main__":
  main()
