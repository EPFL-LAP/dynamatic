import argparse
import ast
import sys

import importlib

def gen(mod_path, func):
    mod = importlib.import_module(f"generate.{mod_path}.{func}")
    return getattr(mod, func)

generators = {
    "absf": gen("handshake", "absf"),
    "addf": gen("handshake", "addf"),
    "addi": gen("handshake", "addi"),
    "andi": gen("handshake", "andi"),
    "buffer": gen("handshake", "buffer"),
    "cmpi": gen("handshake", "cmpi"),
    "cmpf": gen("handshake", "cmpf"),
    "cond_br": gen("handshake", "cond_br"),
    "constant": gen("handshake", "constant"),
    "control_merge": gen("handshake", "control_merge"),
    "divf": gen("handshake", "divf"),
    "divsi": gen("handshake", "divsi"),
    "divui": gen("handshake", "divui"),
    "negf": gen("handshake", "negf"),
    "extsi": gen("handshake", "extsi"),
    "extf": gen("handshake", "extf"),
    "fork": gen("handshake", "fork"),
    "lazy_fork": gen("handshake", "lazy_fork"),
    "load": gen("handshake", "load"),
    "maximumf": gen("handshake", "maximumf"),
    "minimumf": gen("handshake", "minimumf"),
    "mem_controller": gen("handshake", "mem_controller"),
    "merge": gen("handshake", "merge"),
    "mulf": gen("handshake", "mulf"),
    "muli": gen("handshake", "muli"),
    "mux": gen("handshake", "mux"),
    "ori": gen("handshake", "ori"),
    "xori": gen("handshake", "xori"),
    "not": gen("handshake", "logical_not"),
    "select": gen("handshake", "select"),
    "sink": gen("handshake", "sink"),
    "source": gen("handshake", "source"),
    "store": gen("handshake", "store"),
    "subf": gen("handshake", "subf"),
    "subi": gen("handshake", "subi"),
    "trunci": gen("handshake", "trunci"),
    "truncf": gen("handshake", "truncf"),
    "spec_commit": gen("handshake.speculation", "spec_commit"),
    "spec_save_commit": gen("handshake.speculation", "spec_save_commit"),
    "speculating_branch": gen("handshake.speculation", "speculating_branch"),
    "speculator": gen("handshake.speculation", "speculator"),
    "non_spec": gen("handshake.speculation", "non_spec"),
    "mem_to_bram": gen("support", "mem_to_bram"),
    "extui": gen("handshake", "extui"),
    "shli": gen("handshake", "shli"),
    "shrsi": gen("handshake", "shrsi"),
    "shrui": gen("handshake", "shrui"),
    "blocker": gen("handshake", "blocker"),
    "sitofp": gen("handshake", "sitofp"),
    "fptosi": gen("handshake", "fptosi"),
    "ready_remover": gen("handshake", "ready_remover"),
    "valid_merger": gen("handshake", "valid_merger"),
}


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
        print(header + generate_code(args.name, args.type, parameters), file=file)


if __name__ == "__main__":
    main()
