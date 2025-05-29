from generators.support.signal_manager import generate_default_signal_manager
from generators.handshake.join import generate_join


def generate_cond_br(name, params):
    bitwidth = params["bitwidth"]
    extra_signals = params.get("extra_signals", None)

    if extra_signals:
        return _generate_cond_br_signal_manager(name, bitwidth, extra_signals)
    elif bitwidth == 0:
        return _generate_cond_br_dataless(name)
    else:
        return _generate_cond_br(name, bitwidth)


def _generate_cond_br_dataless(name):
    join_name = f"{name}_join"

    dependencies = generate_join(join_name, {"size": 2})

    entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of cond_br_dataless
entity {name} is
  port (
    clk, rst : in std_logic;
    -- data input channel
    data_valid : in  std_logic;
    data_ready : out std_logic;
    -- condition input channel
    condition       : in  std_logic_vector(0 downto 0);
    condition_valid : in  std_logic;
    condition_ready : out std_logic;
    -- true output channel
    trueOut_valid : out std_logic;
    trueOut_ready : in  std_logic;
    -- false output channel
    falseOut_valid : out std_logic;
    falseOut_ready : in  std_logic
  );
end entity;
"""

    architecture = f"""
-- Architecture of cond_br_dataless
architecture arch of {name} is
  signal branchInputs_valid, branch_ready : std_logic;
begin

  join : entity work.{join_name}(arch)
    port map(
      -- input channels
      ins_valid(0) => data_valid,
      ins_valid(1) => condition_valid,
      ins_ready(0) => data_ready,
      ins_ready(1) => condition_ready,
      -- output channel
      outs_valid => branchInputs_valid,
      outs_ready => branch_ready
    );

  trueOut_valid  <= condition(0) and branchInputs_valid;
  falseOut_valid <= (not condition(0)) and branchInputs_valid;
  branch_ready   <= (falseOut_ready and not condition(0)) or (trueOut_ready and condition(0));
end architecture;
"""

    return dependencies + entity + architecture


def _generate_cond_br(name, bitwidth):
    inner_name = f"{name}_inner"

    dependencies = _generate_cond_br_dataless(inner_name)

    entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of cond_br
entity {name} is
  port (
    clk, rst : in std_logic;
    -- data input channel
    data       : in  std_logic_vector({bitwidth} - 1 downto 0);
    data_valid : in  std_logic;
    data_ready : out std_logic;
    -- condition input channel
    condition       : in  std_logic_vector(0 downto 0);
    condition_valid : in  std_logic;
    condition_ready : out std_logic;
    -- true output channel
    trueOut       : out std_logic_vector({bitwidth} - 1 downto 0);
    trueOut_valid : out std_logic;
    trueOut_ready : in  std_logic;
    -- false output channel
    falseOut       : out std_logic_vector({bitwidth} - 1 downto 0);
    falseOut_valid : out std_logic;
    falseOut_ready : in  std_logic
  );
end entity;
"""

    architecture = f"""
-- Architecture of cond_br
architecture arch of {name} is
begin
  control : entity work.{inner_name}
    port map(
      clk             => clk,
      rst             => rst,
      data_valid      => data_valid,
      data_ready      => data_ready,
      condition       => condition,
      condition_valid => condition_valid,
      condition_ready => condition_ready,
      trueOut_valid   => trueOut_valid,
      trueOut_ready   => trueOut_ready,
      falseOut_valid  => falseOut_valid,
      falseOut_ready  => falseOut_ready
    );

  trueOut  <= data;
  falseOut <= data;
end architecture;
"""

    return dependencies + entity + architecture


def _generate_cond_br_signal_manager(name, bitwidth, extra_signals):
    return generate_default_signal_manager(
        name,
        [{
            "name": "data",
            "bitwidth": bitwidth,
            "extra_signals": extra_signals
        }, {
            "name": "condition",
            "bitwidth": 1,
            "extra_signals": extra_signals
        }],
        [{
            "name": "trueOut",
            "bitwidth": bitwidth,
            "extra_signals": extra_signals
        }, {
            "name": "falseOut",
            "bitwidth": bitwidth,
            "extra_signals": extra_signals
        }],
        extra_signals,
        lambda name:
            (_generate_cond_br_dataless(name) if bitwidth == 0
             else _generate_cond_br(name, bitwidth)))
