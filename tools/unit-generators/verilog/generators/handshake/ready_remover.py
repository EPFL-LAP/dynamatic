from generators.support.utils import data

def generate_ready_remover(name, params):
    bitwidth = params["bitwidth"]
    extra_signals = params["extra_signals"]

    def generate_inner(name): return _generate_ready_remover(name, bitwidth)
    def generate(): return generate_inner(name)

    if extra_signals:
        assert False, "Extra signal is not currently supported in ready_remover!"
    else:
        return generate()


def _generate_ready_remover(name, bitwidth):
    potential_input = f"input [{bitwidth} - 1 : 0] ins,"
    potential_output = f"output [{bitwidth} - 1 : 0] outs,"
    potential_assignment = "assign outs = ins;"

    return f"""
module {name} (
    input clk,
    input rst,
    {data(potential_input, bitwidth)}
    input ins_valid,
    input outs_ready,
    {data(potential_output, bitwidth)}
    output outs_valid,
    output ins_ready
);
  {data(potential_assignment, bitwidth)}
  assign outs_valid = ins_valid;
  assign ins_ready = 1'b1;
endmodule
"""

