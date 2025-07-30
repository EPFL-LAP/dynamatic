from generators.handshake.cmpf import generate_cmpf
from generators.support.utils import VIVADO_IMPL, FLOPOCO_IMPL
from generators.support.signal_manager import generate_arith2_signal_manager


def generate_minimumf(name, params):
    latency = params["latency"]

    fpu_impl = params["fpu_impl"]

    extra_signals = params.get("extra_signals", None)

    # flopoco only
    is_double = params.get("is_double", None)

    if fpu_impl == VIVADO_IMPL:
        bitwidth = 32
    elif fpu_impl == FLOPOCO_IMPL:
        if is_double is None:
            raise ValueError("is_double was missing for generating a flopoco minimumf")
        bitwidth = 64 if is_double else 32

    def generate_inner(name): return _generate_minimumf(name, params, bitwidth)
    def generate(): return generate_inner(name)

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


def _generate_minimumf(name, params, bitwidth):
    cmpf_name = f"{name}_cmp"
    cmpf_params = {k: v for k, v in params.items() if k != "extra_signals"}
    cmpf_params["predicate"] = "olt"
    dependencies = generate_cmpf(cmpf_name, cmpf_params)

    entity = f"""
library ieee;
use ieee.std_logic_1164.all;

-- Entity of minimumf
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
-- Architecture of minimumf
architecture arch of {name} is

  signal cmp_result : std_logic_vector(0 downto 0);

begin

  cmp_inst: entity work.{cmpf_name}
    port map (
      clk          => clk,
      rst          => rst,
      lhs          => lhs,
      lhs_valid    => lhs_valid,
      lhs_ready    => lhs_ready,
      rhs          => rhs,
      rhs_valid    => rhs_valid,
      rhs_ready    => rhs_ready,
      result       => cmp_result,
      result_valid => result_valid,
      result_ready => result_ready
    );

  result <= lhs when cmp_result = "1" else rhs;

end architecture;
"""

    return dependencies + entity + architecture
