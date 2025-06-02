
def generate_valid_merger(name, params):
    left_bitwidth = params["left_bitwidth"]
    right_bitwidth = params["right_bitwidth"]

    if left_bitwidth > 0 and right_bitwidth > 0:
        return _generate_valid_merger(name, left_bitwidth, right_bitwidth)
    elif left_bitwidth > 0:
        return _generate_valid_merger_right_dataless(name, left_bitwidth)
    elif right_bitwidth > 0:
        return _generate_valid_merger_left_dataless(name, right_bitwidth)
    else:
        return _generate_valid_merger_dataless(name)


def _generate_valid_merger(name, left_bitwidth, right_bitwidth):

    entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of valid_merger
entity {name} is
  port (
    -- inputs
    clk              : in std_logic;
    rst              : in std_logic;
    lhs_ins          : in std_logic_vector({left_bitwidth} - 1 downto 0);
    lhs_ins_valid    : in std_logic;
    lhs_outs_ready   : in std_logic;
    rhs_ins          : in std_logic_vector({right_bitwidth} - 1 downto 0);
    rhs_ins_valid    : in std_logic;
    rhs_outs_ready   : in std_logic;
    -- outputs
    lhs_outs         : out std_logic_vector({left_bitwidth} - 1 downto 0);
    lhs_outs_valid   : out std_logic;
    lhs_ins_ready    : out std_logic;
    rhs_outs         : out std_logic_vector({right_bitwidth} - 1 downto 0);
    rhs_outs_valid   : out std_logic;
    rhs_ins_ready    : out std_logic
  );
end entity;
"""

    architecture = f"""
-- Architecture of valid_merger
architecture arch of {name} is
begin
  lhs_outs <= lhs_ins;
  rhs_outs <= rhs_ins;
  lhs_outs_valid <= lhs_ins_valid;
  rhs_outs_valid <= lhs_ins_valid; -- merge happens here
  lhs_ins_ready <= lhs_outs_ready;
  rhs_ins_ready <= rhs_outs_ready;
end architecture;
"""

    return entity + architecture


def _generate_valid_merger_right_dataless(name, left_bitwidth):

    entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of valid_merger
entity {name} is
  port (
    -- inputs
    clk              : in std_logic;
    rst              : in std_logic;
    lhs_ins          : in std_logic_vector({left_bitwidth} - 1 downto 0);
    lhs_ins_valid    : in std_logic;
    lhs_outs_ready   : in std_logic;
    rhs_ins_valid    : in std_logic;
    rhs_outs_ready   : in std_logic;
    -- outputs
    lhs_outs         : out std_logic_vector({left_bitwidth} - 1 downto 0);
    lhs_outs_valid   : out std_logic;
    lhs_ins_ready    : out std_logic;
    rhs_outs_valid   : out std_logic;
    rhs_ins_ready    : out std_logic
  );
end entity;
"""

    architecture = f"""
-- Architecture of valid_merger
architecture arch of {name} is
begin
  lhs_outs <= lhs_ins;
  lhs_outs_valid <= lhs_ins_valid;
  rhs_outs_valid <= lhs_ins_valid; -- merge happens here
  lhs_ins_ready <= lhs_outs_ready;
  rhs_ins_ready <= rhs_outs_ready;
end architecture;
"""

    return entity + architecture


def _generate_valid_merger_left_dataless(name, right_bitwidth):

    entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of valid_merger
entity {name} is
  port (
    -- inputs
    clk              : in std_logic;
    rst              : in std_logic;
    lhs_ins_valid    : in std_logic;
    lhs_outs_ready   : in std_logic;
    rhs_ins          : in std_logic_vector({right_bitwidth} - 1 downto 0);
    rhs_ins_valid    : in std_logic;
    rhs_outs_ready   : in std_logic;
    -- outputs
    lhs_outs_valid   : out std_logic;
    lhs_ins_ready    : out std_logic;
    rhs_outs         : out std_logic_vector({right_bitwidth} - 1 downto 0);
    rhs_outs_valid   : out std_logic;
    rhs_ins_ready    : out std_logic
  );
end entity;
"""

    architecture = f"""
-- Architecture of valid_merger
architecture arch of {name} is
begin
  rhs_outs <= rhs_ins;
  lhs_outs_valid <= lhs_ins_valid;
  rhs_outs_valid <= lhs_ins_valid; -- merge happens here
  lhs_ins_ready <= lhs_outs_ready;
  rhs_ins_ready <= rhs_outs_ready;
end architecture;
"""

    return entity + architecture


def _generate_valid_merger_dataless(name):
    entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of valid_merger
entity {name} is
  port (
    -- inputs
    clk              : in std_logic;
    rst              : in std_logic;
    lhs_ins_valid    : in std_logic;
    lhs_outs_ready   : in std_logic;
    rhs_ins_valid    : in std_logic;
    rhs_outs_ready   : in std_logic;
    -- outputs
    lhs_outs_valid   : out std_logic;
    lhs_ins_ready    : out std_logic;
    rhs_outs_valid   : out std_logic;
    rhs_ins_ready    : out std_logic
  );
end entity;
"""

    architecture = f"""
-- Architecture of valid_merger
architecture arch of {name} is
begin
  lhs_outs_valid <= lhs_ins_valid;
  rhs_outs_valid <= lhs_ins_valid; -- merge happens here
  lhs_ins_ready <= lhs_outs_ready;
  rhs_ins_ready <= rhs_outs_ready;
end architecture;
"""

    return entity + architecture
