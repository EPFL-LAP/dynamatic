from generators.support.signal_manager import generate_arith2_signal_manager
from generators.support.arith2 import generate_arith2


def generate_subi(name, params):
    bitwidth = params["bitwidth"]

    body = """
  result <= std_logic_vector(unsigned(lhs) - unsigned(rhs));
    """

    return generate_arith2(
        name=name,
        modType="subi",
        bitwidth=bitwidth,
        body=body,
        extra_signals=params.get("extra_signals", None),
    )
