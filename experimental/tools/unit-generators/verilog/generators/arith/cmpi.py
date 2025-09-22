from generators.handshake.binop import generate_binop
from generators.handshake.join import generate_join


def parse_predicate(predicate_string: str):
    mapping = {
        "eq":  ("==", False),
        "ne":  ("!=", False),
        "ult": ("<",  False),
        "slt": ("<",  True),
        "ule": ("<=", False),
        "sle": ("<=", True),
        "ugt": (">",  False),
        "sgt": (">",  True),
        "uge": (">=", False),
        "sge": (">=", True),
    }

    try:
        return mapping[predicate_string]
    except KeyError:
        raise ValueError(f"Unknown predicate: {predicate_string}")

def generate_cmpi(name, params):
    bitwidth = params["bitwidth"]
    predicate_string = params["predicate"]


    predicate, signed = parse_predicate(predicate_string)
    
    if signed:
        body = f"""
    wire constant_one = 1'b1;
    wire constant_zero = 1'b0;
    wire signed [{bitwidth} - 1 : 0] signed_lhs;
    wire signed [{bitwidth} - 1 : 0] signed_rhs;
    assign signed_lhs = lhs;
    assign signed_rhs = rhs;

    assign result = (signed_lhs {predicate} signed_rhs) ? constant_one : constant_zero;
"""

    else:
        body = f"""
    wire constant_one = 1'b1;
    wire constant_zero = 1'b0;
    assign result = (lhs {predicate} rhs) ? constant_one : constant_zero;
"""
    
    return generate_binop(
        name=name,
        op_body=body,
        handshake_op="cmpi",
        lhs_bitwidth=bitwidth,
        rhs_bitwidth=bitwidth,
        output_bitwidth=1,
        extra_signals=params.get("extra_signals", None))
