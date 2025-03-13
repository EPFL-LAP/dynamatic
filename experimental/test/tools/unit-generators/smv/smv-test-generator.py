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


def generate_input_ports(in_port_types, out_port_types):
  smv_ports = {}
  for port_name, port_type in in_port_types.items():
    if SmvScalarType(port_type).bitwidth != 0:
      smv_ports[port_name] = SmvScalarType(port_type).smv_type
    smv_ports[port_name + "_valid"] = "boolean"

  for port_name, port_type in out_port_types.items():
    smv_ports[port_name + "_ready"] = "boolean"
  return smv_ports

def generate_output_ports(in_port_types, out_port_types):
  smv_ports = {}
  for port_name, port_type in out_port_types.items():
    if SmvScalarType(port_type).bitwidth != 0:
      smv_ports[port_name] = SmvScalarType(port_type).smv_type
    smv_ports[port_name + "_valid"] = "boolean"

  for port_name, port_type in in_port_types.items():
    smv_ports[port_name + "_ready"] = "boolean"
  return smv_ports


def generate_test_bench(name, mod_type, parameters):
  input_ports = generate_input_ports(parameters["in_port_types"], parameters["out_port_types"])
  output_ports = generate_output_ports(parameters["in_port_types"], parameters["out_port_types"])
  return f"""
#include "{name}.smv"
#include "golden_model.smv"


MODULE main
  {"\n  ".join([f"VAR internal_{port_name} : {port_type};" for port_name, port_type in input_ports.items()])}

  VAR model_under_test : {name}({", ".join([f"internal_{port_name}" for port_name, _ in input_ports.items()])});
  VAR golden_model : _{name}();
  {"\n  ".join([f"ASSIGN golden_model._{port_name} := {f"word1(internal_{port_name})" if port_type == "boolean" else f"internal_{port_name}"};" for port_name, port_type in input_ports.items()])}

  ASSIGN golden_model._rst := 0ub_0;

  {"\n  ".join([f"INVARSPEC NAME {port_name}_check := model_under_test.{port_name} = {f"bool(golden_model._{port_name})" if port_type == "boolean" else f"golden_model._{port_name}"};" for port_name, port_type in output_ports.items()])}
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
  header = f"// {args.name} : {args.type}({args.parameters})\n\n"

  with open(args.output, 'w') as file:
    print(header + generate_test_bench(args.name, args.type, parameters), file=file)


if __name__ == "__main__":
  main()
