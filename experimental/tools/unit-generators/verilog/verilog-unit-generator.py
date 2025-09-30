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
#generators.add("handshake", "join")
generators.add("handshake", "addi")
generators.add("handshake", "andi")
generators.add("handshake", "ori")
generators.add("handshake", "xori")
generators.add("handshake", "muli")
generators.add("handshake", "shli")
generators.add("handshake", "shrui")
generators.add("handshake", "shrsi")
generators.add("handshake", "subi")
generators.add("handshake", "cmpi")
generators.add("handshake", "extui")
generators.add("handshake", "extsi")
generators.add("handshake", "trunci")
generators.add("handshake", "divsi")
generators.add("handshake", "remsi")
generators.add("handshake", "divui")

generators.add("handshake", "not")
generators.add("handshake", "sink")
generators.add("handshake", "join")
generators.add("handshake", "constant")
generators.add("handshake", "select")
generators.add("handshake", "ndwire")
generators.add("handshake", "fork")
generators.add("handshake", "lazy_fork")
generators.add("handshake", "mux")
generators.add("handshake", "br")
generators.add("handshake", "cond_br")
generators.add("handshake", "source")
generators.add("handshake", "control_merge")
generators.add("handshake", "load")
generators.add("handshake", "store")
generators.add("handshake", "tehb")
generators.add("handshake", "oehb")
generators.add("handshake.dataless", "dataless_oehb")
generators.add("handshake", "mem_controller")
generators.add("handshake", "mem_controller_loadless")
generators.add("handshake", "mem_controller_storeless")

generators.add("handshake", "tfifo")
#generators.add("handshake.dataless", "dataless_tfifo")
generators.add("handshake", "one_slot_break_dvr")
#generators.add("handshake.dataless", "dataless_one_slot_break_dvr")
generators.add("handshake", "shift_reg_break_dv")
#generators.add("handshake.dataless", "dataless_shift_reg_break_dv")

generators.add("support", "mem_to_bram")
generators.add("support", "elastic_fifo_inner")
#generators.add("support.dataless", "dataless_elastic_fifo_inner")

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
    header = f"// {args.name} : {args.type}({parameters})\n\n"

    if args.type not in generators:
        raise ValueError(f"Module type {args.type} not found")

    generate_code = generators[args.type]
    
    with open(args.output, "w") as file:
        print(header + generate_code(args.name, parameters), file=file)
    # with open(args.output, "w") as file:
    #     print(header + generate_code(args.name, parameters)[0], file=file)

    # with open(args.output.split(".")[0] + "_join.v", "w") as file:
    #     print(header + generate_code(args.name, parameters)[1], file=file)


if __name__ == "__main__":
    main()
