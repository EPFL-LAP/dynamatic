from generators.handshake.addf import generate_addf
from generators.support.arith2 import generate_arith2


def generate_subf(name, params):
    impl = params["fpu_impl"]
    latency = params["latency"]

    # only used by flopoco
    is_double = params.get("is_double", None)

    if impl == "vivado":
        bitwidth = 32
    elif impl == "flopoco":
        if is_double is None:
            raise ValueError("is_double was missing for generating a flopoco subf")
        bitwidth = 64 if is_double else 32
    else:
        raise ValueError(f"Invalid FPU implementation: {impl}")

    addf_name = f"{name}_addf"
    addf_params = {k: v for k, v in params.items() if k != "extra_signals"}
    dependencies = generate_addf(addf_name, addf_params)

    signals = f"""
  -- subf is the same as addf, but we flip the sign bit of rhs
  signal rhs_neg : std_logic_vector({bitwidth} - 1 downto 0);
    """

    body = f"""
  rhs_neg <= not rhs({bitwidth} - 1) & rhs({bitwidth} - 2 downto 0);

  FloatingPointAdder_U1: entity work.{addf_name}
  port map (
    clk => clk,
    rst => rst,
    -- input channel from "lhs"
    lhs => lhs,
    lhs_valid => lhs_valid,
    lhs_ready => lhs_ready,
    -- input channel from "rhs", made negative
    rhs => rhs_neg,
    rhs_valid => rhs_valid,
    rhs_ready => rhs_ready,
    --output channel to "result"
    result => result,
    result_valid => result_valid,
    result_ready => result_ready
  );
    """

    return generate_arith2(
        name=name,
        modType="subf",
        dependencies=dependencies,
        signals=signals,
        body=body,
        bitwidth=bitwidth,
        latency=latency,
        extra_signals=params.get("extra_signals", None)
    )
