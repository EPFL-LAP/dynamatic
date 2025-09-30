from generators.support.signal_manager import generate_default_signal_manager
from generators.handshake.join import generate_join


def generate_passer(name, params):
    bitwidth = params["bitwidth"]
    extra_signals = params.get("extra_signals", None)

    if extra_signals:
        return _generate_passer_signal_manager(name, bitwidth, extra_signals)
    else:
        if bitwidth == 0:
            return _generate_passer_dataless(name)
        else:
            return _generate_passer(name, bitwidth)


def _generate_passer_dataless(name):
    join_name = f"{name}_join"
    dependency = generate_join(join_name, {"size": 2})

    return dependency + f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of passer_dataless
entity {name} is
  port (
    clk, rst : in std_logic;
    data_valid : in std_logic;
    data_ready : out std_logic;
    ctrl : in std_logic_vector(0 downto 0);
    ctrl_valid : in std_logic;
    ctrl_ready : out std_logic;
    result_valid : out std_logic;
    result_ready : in std_logic
  );
end entity;

-- Architecture of passer_dataless
architecture arch of {name} is
  signal branch_valid, branch_ready : std_logic;
begin
  branch_ready <= not ctrl(0) or result_ready;
  result_valid <= branch_valid and ctrl(0);
  join_inputs : entity work.{join_name}(arch)
    port map(
      -- inputs
      ins_valid(0) => data_valid,
      ins_valid(1) => ctrl_valid,
      outs_ready   => branch_ready,
      -- outputs
      outs_valid   => branch_valid,
      ins_ready(0) => data_ready,
      ins_ready(1) => ctrl_ready
    );
end architecture;
"""


def _generate_passer(name, bitwidth):
    inner_name = f"{name}_inner"
    dependency = _generate_passer_dataless(inner_name)

    return dependency + f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of passer
entity {name} is
  port (
    clk, rst : in std_logic;
    data : in std_logic_vector({bitwidth} - 1 downto 0);
    data_valid : in std_logic;
    data_ready : out std_logic;
    ctrl : in std_logic_vector(0 downto 0);
    ctrl_valid : in std_logic;
    ctrl_ready : out std_logic;
    result : out std_logic_vector({bitwidth} - 1 downto 0);
    result_valid : out std_logic;
    result_ready : in std_logic
  );
end entity;

-- Architecture of passer
architecture arch of {name} is
begin
  inner : entity work.{inner_name}(arch)
    port map(
      clk          => clk,
      rst          => rst,
      data_valid   => data_valid,
      data_ready   => data_ready,
      ctrl         => ctrl,
      ctrl_valid   => ctrl_valid,
      ctrl_ready   => ctrl_ready,
      result_valid => result_valid,
      result_ready => result_ready
    );

  result <= data;
end architecture;
"""


def _generate_passer_signal_manager(name, bitwidth, extra_signals):
    return generate_default_signal_manager(
        name,
        [{
            "name": "data",
            "bitwidth": bitwidth,
            "extra_signals": extra_signals
        }, {
            "name": "ctrl",
            "bitwidth": 1,
            "extra_signals": extra_signals
        }],
        [{
            "name": "result",
            "bitwidth": bitwidth,
            "extra_signals": extra_signals
        }],
        extra_signals,
        lambda name:
            (_generate_passer_dataless(name) if bitwidth == 0
             else _generate_passer(name, bitwidth)))
