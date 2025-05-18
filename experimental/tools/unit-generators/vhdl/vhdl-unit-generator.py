import argparse
import ast
import sys

import generators.handshake.addf as addf
import generators.handshake.addi as addi
import generators.handshake.andi as andi
import generators.handshake.buffer as buffer
import generators.handshake.cmpf as cmpf
import generators.handshake.cmpi as cmpi
import generators.handshake.cond_br as cond_br
import generators.handshake.constant as constant
import generators.handshake.control_merge as control_merge
import generators.handshake.extsi as extsi
import generators.handshake.fork as fork
import generators.handshake.load as load
import generators.handshake.mem_controller as mem_controller
import generators.handshake.merge as merge
import generators.handshake.mulf as mulf
import generators.handshake.muli as muli
import generators.handshake.mux as mux
import generators.handshake.select as select
import generators.handshake.sink as sink
import generators.handshake.source as source
import generators.handshake.store as store
import generators.handshake.subf as subf
import generators.handshake.subi as subi
import generators.handshake.trunci as trunci
import generators.handshake.speculation.spec_commit as spec_commit
import generators.handshake.speculation.spec_save_commit as spec_save_commit
import generators.handshake.speculation.speculating_branch as speculating_branch
import generators.handshake.speculation.speculator as speculator
import generators.handshake.speculation.non_spec as non_spec
import generators.support.mem_to_bram as mem_to_bram
import generators.handshake.extui as extui
import generators.handshake.shli as shli
import generators.handshake.blocker as blocker
import generators.handshake.sitofp as sitofp
import generators.handshake.fptosi as fptosi


def generate_code(name, mod_type, parameters):
    match mod_type:
        case "addf":
            return addf.generate_addf(name, parameters)
        case "addi":
            return addi.generate_addi(name, parameters)
        case "andi":
            return andi.generate_andi(name, parameters)
        case "buffer":
            return buffer.generate_buffer(name, parameters)
        case "cmpi":
            return cmpi.generate_cmpi(name, parameters)
        case "cmpf":
            return cmpf.generate_cmpf(name, parameters)
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
        case "mem_controller":
            return mem_controller.generate_mem_controller(name, parameters)
        case "merge":
            return merge.generate_merge(name, parameters)
        case "mulf":
            return mulf.generate_mulf(name, parameters)
        case "muli":
            return muli.generate_muli(name, parameters)
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
        case "subf":
            return subf.generate_subf(name, parameters)
        case "subi":
            return subi.generate_subi(name, parameters)
        case "trunci":
            return trunci.generate_trunci(name, parameters)
        case "spec_commit":
            return spec_commit.generate_spec_commit(name, parameters)
        case "spec_save_commit":
            return spec_save_commit.generate_spec_save_commit(name, parameters)
        case "speculating_branch":
            return speculating_branch.generate_speculating_branch(name, parameters)
        case "speculator":
            return speculator.generate_speculator(name, parameters)
        case "non_spec":
            return non_spec.generate_non_spec(name, parameters)
        case "mem_to_bram":
            return mem_to_bram.generate_mem_to_bram(name, parameters)
        case "extui":
            return extui.generate_extui(name, parameters)
        case "shli":
            return shli.generate_shli(name, parameters)
        case "blocker":
            return blocker.generate_blocker(name, parameters)
        case "sitofp":
            return sitofp.generate_sitofp(name, parameters)
        case "fptosi":
            return fptosi.generate_fptosi(name, parameters)
        case _:
            raise ValueError(f"Module type {mod_type} not found")


def parse_parameters(param_list):
    try:
        param_dict = {}
        if param_list is not None:
            for pair in param_list:
                key, value = pair.split("=")
                param_dict[key.strip()] = ast.literal_eval(value.strip())
        return param_dict
    except ValueError:
        raise ValueError("Invalid parameter format. Use key=value key=value,...\n")


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

    with open(args.output, "w") as file:
        print(header + generate_code(args.name, args.type, parameters), file=file)


if __name__ == "__main__":
    main()
