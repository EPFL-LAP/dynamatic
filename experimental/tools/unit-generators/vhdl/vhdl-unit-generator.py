import argparse

import generators.handshake.addi as addi
import generators.handshake.buffer as buffer
import generators.handshake.cmpi as cmpi
import generators.handshake.cond_br as cond_br
import generators.handshake.constant as constant
import generators.handshake.control_merge as control_merge
import generators.handshake.extsi as extsi
import generators.handshake.fork as fork
import generators.handshake.load as load
import generators.handshake.merge as merge
import generators.handshake.muli as muli
import generators.handshake.mux as mux
import generators.handshake.sink as sink
import generators.handshake.source as source
import generators.handshake.store as store
import generators.handshake.trunci as trunci

def generate_code(name, mod_type, parameters):
  match mod_type:
    case "addi":
      return addi.generate_addi(name, parameters)
    case "buffer":
      return buffer.generate_buffer(name, parameters)
    case "cmpi":
      return cmpi.generate_cmpi(name, parameters)
    case "cond_br":
      return cond_br.generate_cond_br(name, parameters)
    case "constant":
      return constant.generate_constant(name, parameters)
    case "control_merge":
      return control_merge.generate_control_merge(name, parameters)
    case "extsi":
      return extsi.generate_extsi(name, parameters)
    case "fork":
      return fork.generate_fork(name, parameters)
    case "load":
      return load.generate_load(name, parameters)
    case "merge":
      return merge.generate_merge(name, parameters)
    case "muli":
      return muli.generate_muli(name, parameters)
    case "mux":
      return mux.generate_mux(name, parameters)
    case "sink":
      return sink.generate_sink(name, parameters)
    case "source":
      return source.generate_source(name, parameters)
    case "store":
      return store.generate_store(name, parameters)
    case "trunci":
      return trunci.generate_trunci(name, parameters)
    case _:
      return f"Module type {mod_type} not found"
      # raise ValueError(f"Module type {mod_type} not found")

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
      nargs="*",
      default=[],
      help="Set of parameters in key=value key=value format",
  )

  args = parser.parse_args()

  parameters = dict(args.parameters)

  # Printing parameters for diagnostic purposes
  header = f"-- {args.name} : {args.type}({parameters})\n\n"
  print(header + generate_code(args.name, args.type, parameters))


if __name__ == "__main__":
  main()
