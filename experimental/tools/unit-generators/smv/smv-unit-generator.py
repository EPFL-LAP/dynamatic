import argparse
import sys
import ast

import generators.handshake.br as br
import generators.handshake.buffer as buffer
import generators.handshake.cond_br as cond_br
import generators.handshake.constant as constant
import generators.handshake.control_merge as control_merge
import generators.handshake.join as join
import generators.handshake.fork as fork
import generators.handshake.lazy_fork as lazy_fork
import generators.handshake.load as load
import generators.handshake.merge as merge
import generators.handshake.mux as mux
import generators.handshake.select as select
import generators.handshake.sink as sink
import generators.handshake.source as source
import generators.handshake.store as store

import generators.arith.addf as addf
import generators.arith.addi as addi
import generators.arith.andi as andi
import generators.arith.cmpi as cmpi
import generators.arith.divf as divf
import generators.arith.divsi as divsi
import generators.arith.divui as divui
import generators.arith.maximumf as maximumf
import generators.arith.muli as muli


def generate_code(name, mod_type, parameters):
  match mod_type:
    case "br":
      return br.generate_br(name, parameters)
    case "buffer":
      return buffer.generate_buffer(name, parameters)
    case "cond_br":
      return cond_br.generate_cond_br(name, parameters)
    case "constant":
      return constant.generate_constant(name, parameters)
    case "control_merge":
      return control_merge.generate_control_merge(name, parameters)
    case "join":
      return join.generate_join(name, parameters)
    case "fork":
      return fork.generate_fork(name, parameters)
    case "lazy_fork":
      return lazy_fork.generate_lazy_fork(name, parameters)
    case "load":
      return load.generate_load(name, parameters)
    case "merge":
      return merge.generate_merge(name, parameters)
    case "mux":
      return mux.generate_mux(name, parameters)
    case "select":
      return select.generate_select(name, parameters)
    case "sink":
      return sink.generate_sink(name, parameters)
    case "source":
      return source.generate_source(name, parameters)
    case "store":
      return store.generate_store(name, parameters)
    case "addf":
      return addf.generate_addf(name, parameters)
    case "addi":
      return addi.generate_addi(name, parameters)
    case "andi":
      return andi.generate_andi(name, parameters)
    case "cmpi":
      return cmpi.generate_cmpi(name, parameters)
    case "divf":
      return divf.generate_divf(name, parameters)
    case "divsi":
      return divsi.generate_divsi(name, parameters)
    case "divui":
      return divui.generate_divui(name, parameters)
    case "maximumf":
      return maximumf.generate_maximumf(name, parameters)
    case "muli":
      return muli.generate_muli(name, parameters)
    case _:
      raise ValueError(f"Module type {mod_type} not found")


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


def main():
  parser = argparse.ArgumentParser(description="SMV Generator Script")
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
  header = f"// {args.name} : {args.type}({args.parameters})\n\n"

  with open(args.output, 'w') as file:
    print(header + generate_code(args.name, args.type, parameters), file=file)


if __name__ == "__main__":
  main()
