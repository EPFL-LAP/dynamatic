import argparse
from vhdl_gen.configs   import GetConfigs
from vhdl_gen.codegen   import codeGen

def parse_args():    
    # Parse the arguments
    # python3 main.py [-h] [--target-dir PATH_RTL] --spec-file PATH_CONFIGS
    parser = argparse.ArgumentParser()
    parser.add_argument('--target-dir', '-t', dest='path_rtl', default = './', type = str)
    parser.add_argument('--spec-file', '-s', required = True, dest='path_configs', default = '', type = str)
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    lsqConfig = GetConfigs(args.path_configs)
    codeGen(args.path_rtl, lsqConfig)

if __name__ == "__main__":
    main() 