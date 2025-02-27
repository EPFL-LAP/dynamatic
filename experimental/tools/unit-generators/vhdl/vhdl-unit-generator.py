import argparse
import ast
import sys

import generators.handshake.addi as addi
import generators.handshake.cmpi as cmpi
import generators.handshake.cond_br as cond_br
import generators.handshake.fork as fork


def generate_code(name, mod_type, parameters):
  match mod_type:
    case "addi":
      return addi.generate_addi(name, parameters)
    case "cmpi":
      return cmpi.generate_cmpi(name, parameters)
    case "cond_br":
      return cond_br.generate_cond_br(name, parameters)
    case "fork":
      return fork.generate_fork(name, parameters)
    case _:
      raise ValueError(f"Module type {mod_type} not found")


def parse_parameters(param_list):
  try:
    param_dict = {}
    if param_list is not None:
      for pair in param_list:
        key, value = pair.split("=")
        param_dict[key.strip()] = ast.literal_eval(value.strip())
    return param_dict
  except ValueError:
    raise ValueError(
        "Invalid parameter format. Use key=value key=value,...\n")


def main():
  parser = argparse.ArgumentParser(description="VHDL Generator Script")
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
  header = f"-- {args.name} : {args.type}({parameters})\n\n"

  with open(args.output, 'w') as file:
    print(header + generate_code(args.name, args.type, parameters), file=file)


if __name__ == "__main__":
  main()
