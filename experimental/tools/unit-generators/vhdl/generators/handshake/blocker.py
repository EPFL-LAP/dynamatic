from generators.support.signal_manager import generate_signal_manager
from generators.support.logic import generate_and_n


def generate_blocker(name, params):
    # Number of input ports
  size = params["size"]

  bitwidth = params["bitwidth"]
  extra_signals = params.get("extra_signals", None)

  if extra_signals:
    return _generate_blocker_signal_manager(name, size, bitwidth, extra_signals)
  else:
    return _generate_blocker(name, size, bitwidth)


def _generate_blocker(name, size, bitwidth):
  and_n_module_name = f"{name}_and_n"
  dependencies = generate_and_n(and_n_module_name, {"size": size})

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use work.types.all;

-- Entity of blocker
entity {name} is
  port (
    clk          : in std_logic;
    rst          : in std_logic;
    -- input channels
    ins        : in  data_array({size} - 1 downto 0)({bitwidth} - 1 downto 0);
    ins_valid  : in std_logic_vector({size} - 1 downto 0);
    outs_ready : in std_logic;
    -- output channel
    outs       : out std_logic_vector({bitwidth} - 1 downto 0);
    outs_valid : out std_logic;
    ins_ready  : out std_logic_vector({size} - 1 downto 0)
  );
end entity;
"""

  architecture = f"""
-- Architecture of blocker
architecture arch of {name} is
  signal allValid : std_logic;
begin
  allValidAndGate : entity work.{and_n_module_name} port map(ins_valid, allValid);
  outs_valid <= allValid;

  process (ins_valid, outs_ready)
    variable singlePValid : std_logic_vector({size} - 1 downto 0);
  begin
    for i in 0 to {size} - 1 loop
      singlePValid(i) := '1';
      for j in 0 to {size} - 1 loop
        if (i /= j) then
          singlePValid(i) := (singlePValid(i) and ins_valid(j));
        end if;
      end loop;
    end loop;
    for i in 0 to {size} - 1 loop
      ins_ready(i) <= (singlePValid(i) and outs_ready);
    end loop;
  end process;

  -- Propagate the first input directly after validation
  process(allValid, ins)
  begin
    if (allValid = '1') then
      outs <= ins(0);
    end if;
  end process;

end architecture;
"""

  return dependencies + entity + architecture


def _generate_blocker_signal_manager(name, size, bitwidth, extra_signals):
  return generate_signal_manager(name, {
      "type": "normal",
      "in_ports": [{
          "name": "ins",
          "bitwidth": bitwidth,
          "extra_signals": extra_signals,
          "2d": True,
          "size": size
      }],
      "out_ports": [{
          "name": "outs",
          "bitwidth": bitwidth,
          "extra_signals": extra_signals
      }],
      "extra_signals": extra_signals
  }, lambda name: _generate_blocker(name, size, bitwidth))
