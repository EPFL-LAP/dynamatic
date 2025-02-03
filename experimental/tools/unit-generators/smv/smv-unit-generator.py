import argparse
import sys
import ast

from generators import *

def generate_code(name, mod_type, parameters):
    match mod_type:
        case "fork": return generate_fork(name, parameters)
        case _ : print(f"Module type {mod_type} not found")


def parse_parameters(param_string):
    try:
        param_dict = {}
        for pair in param_string.split(','):
            key, value = pair.split('=')
            if (value != ""):
                param_dict[key.strip()] = ast.literal_eval(value.strip())
        return param_dict
    except ValueError:
        raise ValueError("Invalid parameter format. Use key=value,key=value,...")

def main():
    parser = argparse.ArgumentParser(description="SMV Generator Script")
    parser.add_argument("-n", "--name", required=True, help="Name of the generated module")
    parser.add_argument("-t", "--type", required=True, help="Type of the generated module")
    parser.add_argument("-p", "--parameters", required=True, help="Set of parameters in key=value,key=value format")

    args = parser.parse_args()

    try:
        parameters = parse_parameters(args.parameters)
    except ValueError as e:
        print(f"Error parsing parameters: {e}")
        sys.exit(1)

    print(generate_code(args.name, args.type, parameters))

if __name__ == "__main__":
    main()