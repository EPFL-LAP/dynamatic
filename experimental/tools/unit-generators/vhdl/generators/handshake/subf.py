from generators.handshake.addf import generate_addf
from generators.support.signal_manager import generate_arith2_signal_manager


def generate_subf(name, params):
    latency = params["latency"]

    fpu_impl = params["fpu_impl"]

    extra_signals = params.get("extra_signals", None)

    # flopoco only
    is_double = params.get("is_double", None)

    if fpu_impl == "vivado":
        bitwidth = 32
    elif fpu_impl == "flopoco":
        bitwidth = 64 if is_double else 32

    generate_inner = lambda name : _generate_subf(name, params)
    generate = lambda : generate_inner(name)

    if extra_signals:
        return generate_arith2_signal_manager(
            name=name,
            bitwidth=bitwidth,
            extra_signals=extra_signals,
            generate_inner=generate_inner,
            latency=latency
        )
    else:
        return generate()

def _generate_subf(name, params):
    impl = params["fpu_impl"]

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

    entity = f"""
-- Entity of subf
entity {name} is
  port(
    clk: in std_logic;
    rst: in std_logic;
    -- input channel lhs
    lhs: in std_logic_vector({bitwidth} - 1 downto 0);
    lhs_valid: in std_logic;
    lhs_ready: out std_logic;
    -- input channel rhs
    rhs: in std_logic_vector({bitwidth} - 1 downto 0);
    rhs_valid: in std_logic;
    rhs_ready: out std_logic;
    -- output channel result
    result : out std_logic_vector({bitwidth} - 1 downto 0);
    result_valid: out std_logic;
    result_ready: in std_logic
  );
end entity;
"""

    architecture = f"""
-- Architecture of subf
architecture arch of {name} is
  -- subf is the same as addf, but we flip the sign bit of rhs
  signal rhs_neg : std_logic_vector({bitwidth} - 1 downto 0);
begin

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

end architecture;
"""

    return dependencies + entity + architecture