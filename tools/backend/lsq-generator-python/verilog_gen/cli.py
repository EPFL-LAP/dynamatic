import argparse
import vhdl_gen.configs as configs
import vhdl_gen.codegen as code_gen


def parse_args():
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--target-dir', '-t', dest='target_dir', default='./', type=str)
    parser.add_argument('--spec-file', '-s', required=True, dest='spec_file', default='', type=str)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    lsqConfig = configs.GetConfigs(args.spec_file)
    code_gen.codeGen(args.target_dir, lsqConfig)


if __name__ == "__main__":
    main()
