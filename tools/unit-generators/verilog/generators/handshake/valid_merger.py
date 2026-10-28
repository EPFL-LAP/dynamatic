from generators.support.utils import data

def generate_valid_merger(name, params):
    lhs_bitwidth = params["left_bitwidth"]
    rhs_bitwidth = params["right_bitwidth"]

    lhs_extra_signals = params.get("lhs_extra_signals", None)
    rhs_extra_signals = params.get("rhs_extra_signals", None)

    def generate_inner(name): return _generate_valid_merger(name, lhs_bitwidth, rhs_bitwidth)
    def generate(): return generate_inner(name)

    if lhs_extra_signals or rhs_extra_signals:
        assert False, "Extra signal is not currently supported for valid merger!"
    else:
        return generate()


def _generate_valid_merger(name, lhs_bitwidth, rhs_bitwidth):
    possible_lhs_ins = f"input [{lhs_bitwidth} - 1 : 0] lhs_ins,"
    possible_rhs_ins = f"input [{rhs_bitwidth} - 1 : 0] rhs_ins,"
    possible_lhs_outs = f"output [{lhs_bitwidth} - 1 : 0] lhs_outs,"
    possible_rhs_outs = f"output [{rhs_bitwidth} - 1 : 0] rhs_outs,"

    possible_lhs_assignment = "assign lhs_outs = lhs_ins;"
    possible_rhs_assignment = "assign rhs_outs = rhs_ins;"

    return f"""
module {name} (
    input clk,
    input rst,

    // Inputs
    {data(possible_lhs_ins, lhs_bitwidth)}
    input lhs_ins_valid,
    {data(possible_rhs_ins, rhs_bitwidth)}
    input rhs_ins_valid,
    input lhs_outs_ready,
    input rhs_outs_ready,

    // Outputs
    {data(possible_lhs_outs, lhs_bitwidth)}
    output lhs_outs_valid,
    output rhs_ins_ready,
    {data(possible_rhs_outs, rhs_bitwidth)}
    output rhs_outs_valid,
    output lhs_ins_ready
);
  {data(possible_lhs_assignment, lhs_bitwidth)}
  assign lhs_outs_valid = lhs_ins_valid;
  assign lhs_ins_ready = lhs_outs_ready;

  {data(possible_rhs_assignment, rhs_bitwidth)}
  assign rhs_outs_valid = lhs_ins_valid; // merge happens here
  assign rhs_ins_ready = rhs_outs_ready;
endmodule
"""
