import argparse

import generators.handshake.cond_br as cond_br

def generate_code(name, mod_type, parameters):
  match mod_type:
    case "cond_br":
      return cond_br.generate_cond_br(name, parameters)
    case _:
      raise ValueError(f"Module type {mod_type} not found")

def parse_key_value(key_value):
  key, value = key_value.split("=")
  return key, value

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
      type=parse_key_value,
      required=True,
      nargs="+",
      help="Set of parameters in key=value key=value format",
  )

  args = parser.parse_args()

  parameters = dict(args.parameters)

  # Printing parameters for diagnostic purposes
  header = f"-- {args.name} : {args.type}({parameters})\n\n"
  print(header + generate_code(args.name, args.type, parameters))


if __name__ == "__main__":
  main()
