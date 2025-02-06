import argparse
import sys
import ast

import generators.handshake.fork as fork
import generators.handshake.buffer as buffer


def generate_code(name, mod_type, parameters):
  match mod_type:
    case "fork":
      return fork.generate_fork(name, parameters)
    case "buffer":
      return buffer.generate_buffer(name, parameters)
    case _:
      raise ValueError(f"Module type {mod_type} not found")


def parse_parameters(param_list):
  try:
    param_dict = {}
    for pair in param_list:
      key, value = pair.split("=")
      if value != "":
        param_dict[key.strip()] = ast.literal_eval(value.strip())
    return param_dict
  except ValueError:
    raise ValueError("Invalid parameter format. Use key=value key=value,...")


def main():
  parser = argparse.ArgumentParser(description="SMV Generator Script")
  parser.add_argument(
      "-n", "--name", required=True, help="Name of the generated module"
  )
  parser.add_argument(
      "-t", "--type", required=True, help="Type of the generated module"
  )
  parser.add_argument(
      "-p",
      "--parameters",
      required=True,
      nargs="+",
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
  print(header + generate_code(args.name, args.type, parameters))


if __name__ == "__main__":
  main()
