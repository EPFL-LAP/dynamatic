
def generate_ready_remover(name, params):
    bitwidth = params["bitwidth"]

    if bitwidth > 0:
        return _generate_rigidifier(name, bitwidth)
    else:
        return _generate_rigidifier_dataless(name)


def _generate_rigidifier(name, bitwidth):

    entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of rigidifier
entity {name} is
  port (
    -- inputs
    clk          : in std_logic;
    rst          : in std_logic;
    ins          : in std_logic_vector({bitwidth} - 1 downto 0);
    ins_valid    : in std_logic;
    outs_ready : in std_logic;
    -- outputs
    outs       : out std_logic_vector({bitwidth} - 1 downto 0);
    outs_valid : out std_logic;
    ins_ready    : out std_logic
  );
end entity;
"""

    architecture = f"""
-- Architecture of rigidifier
architecture arch of {name} is
begin
  outs <= ins;
  outs_valid <= ins_valid;
  ins_ready <= '1';
end architecture;
"""

    return entity + architecture


def _generate_rigidifier_dataless(name):

    entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of rigidifier
entity {name} is
  port (
    -- inputs
    clk          : in std_logic;
    rst          : in std_logic;
    ins_valid    : in std_logic;
    outs_ready : in std_logic;
    -- outputs
    outs_valid : out std_logic;
    ins_ready    : out std_logic
  );
end entity;
"""

    architecture = f"""
-- Architecture of rigidifier
architecture arch of {name} is
begin
  outs_valid <= ins_valid;
  ins_ready <= '1';
end architecture;
"""

    return entity + architecture
