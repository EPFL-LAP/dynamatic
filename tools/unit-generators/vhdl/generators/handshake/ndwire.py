from generators.support.utils import data
from generators.support.signal_manager import generate_unary_signal_manager


def generate_ndwire(name, params):
    bitwidth = params["bitwidth"]
    extra_signals = params.get("extra_signals", None)

    def generate_inner(name): return _generate_ndwire(name, bitwidth)
    def generate(): return generate_inner(name)

    if extra_signals:
        return generate_unary_signal_manager(
            name=name,
            bitwidth=bitwidth,
            generate_inner=generate_inner,
            extra_signals=extra_signals
        )
    else:
        return generate()


def _generate_ndwire(name: str, bitwidth: int) -> str:
    # only included if bitwidth > 0
    potential_input = f"ins       : in  std_logic_vector({bitwidth} - 1 downto 0);"
    potential_output = f"outs       : out std_logic_vector({bitwidth} - 1 downto 0);"
    potential_assignment = "outs <= ins;"

    entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of ndwire
entity {name} is
  port (
    clk, rst : in std_logic;
    -- input channel
    ins_valid : in  std_logic;
    ins_ready : out std_logic;
    {data(potential_input, bitwidth)}
    -- output channel
    outs_valid : out std_logic;
    outs_ready : in  std_logic
    {data(potential_output, bitwidth)}
  );
end entity;

"""

    architecture = f"""
-- Architecture of ndwire
architecture arch of {name} is
  type nd_state_t is (SLEEPING, RUNNING);
  signal state, next_state : nd_state_t;

  -- This is the source of non-determism.
  -- It needs to be set to a primary input in a formal tool
  -- If the formal tools does not implicitly treat undriven signals
  -- like primary inputs this needs to be done explicitly.
  signal nd_next_state : nd_state_t;

begin
  process (clk, rst)
  begin
    -- The initialization of the state is non-deterministic
    if rst = '1' then
      state <= nd_next_state;
    elsif rising_edge(clk) then
      state <= next_state;
    end if;
  end process;

  process (state, ins_valid, outs_ready)
  begin
    -- If the wire is sleeping it can always switch to the running state.
    -- If (ins_valid and outs_ready) we either have a transaction
    -- and can freely choose the state again.
    if (state = SLEEPING) then
      next_state <= nd_next_state;
    elsif (ins_valid and outs_ready) then
      next_state <= nd_next_state;
    else
      next_state <= state;
    end if;
  end process;

  ins_ready <= outs_ready and (state = RUNNING);
  outs_valid <= ins_valid and (state = RUNNING);
  {data(potential_assignment, bitwidth)}

end architecture;
"""
    return entity + architecture
