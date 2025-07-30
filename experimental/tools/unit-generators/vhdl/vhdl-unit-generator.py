import argparse
import ast
import sys

import importlib

class Generators():
    def __init__(self):
        self._data = {}

    def __getitem__(self, key):
        return self._data[key]

    def __contains__(self, key):
        return key in self._data

    def add(self, category, mod):
        imported = importlib.import_module(f"generators.{category}.{mod}")
        self._data[mod] = getattr(imported, f"generate_{mod}")


generators = Generators()
generators.add("handshake", "absf")
generators.add("handshake", "addf")
generators.add("handshake", "addi")
generators.add("handshake", "andi")
generators.add("handshake", "buffer")
generators.add("handshake", "cmpi")
generators.add("handshake", "cmpf")
generators.add("handshake", "cond_br")
generators.add("handshake", "br")
generators.add("handshake", "constant")
generators.add("handshake", "control_merge")
generators.add("handshake", "divf")
generators.add("handshake", "divsi")
generators.add("handshake", "divui")
generators.add("handshake", "negf")
generators.add("handshake", "extsi")
generators.add("handshake", "extf")
generators.add("handshake", "fork")
generators.add("handshake", "lazy_fork")
generators.add("handshake", "load")
generators.add("handshake", "maximumf")
generators.add("handshake", "minimumf")
generators.add("handshake", "mem_controller")
generators.add("handshake", "merge")
generators.add("handshake", "mulf")
generators.add("handshake", "muli")
generators.add("handshake", "mux")
generators.add("handshake", "ndwire")
generators.add("handshake", "ori")
generators.add("handshake", "xori")
generators.add("handshake", "logical_not")
generators.add("handshake", "select")
generators.add("handshake", "sink")
generators.add("handshake", "source")
generators.add("handshake", "store")
generators.add("handshake", "subf")
generators.add("handshake", "subi")
generators.add("handshake", "trunci")
generators.add("handshake", "truncf")
generators.add("handshake.speculation", "spec_commit")
generators.add("handshake.speculation", "spec_save_commit")
generators.add("handshake.speculation", "speculating_branch")
generators.add("handshake.speculation", "speculator")
generators.add("handshake.speculation", "non_spec")
generators.add("support", "mem_to_bram")
generators.add("handshake", "extui")
generators.add("handshake", "shli")
generators.add("handshake", "shrsi")
generators.add("handshake", "shrui")
generators.add("handshake", "blocker")
generators.add("handshake", "sitofp")
generators.add("handshake", "fptosi")
generators.add("handshake", "rigidifier")
generators.add("handshake", "valid_merger")
generators.add("handshake", "top_join")
generators.add("handshake", "remsi")



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
    parser.add_argument("-o", "--output", required=True,
                        help="Name of the output file")
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

    if args.type not in generators:
        raise ValueError(f"Module type {args.type} not found")

    generate_code = generators[args.type]

    with open(args.output, "w") as file:
        print(header + generate_code(args.name, parameters), file=file)


if __name__ == "__main__":
    main()
