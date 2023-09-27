library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;
-- #CST_VALUE#
entity constant_node_#CST_NAME# is
  generic (
    BITWIDTH : integer
  );
  port (
    -- inputs
    clk          : in std_logic;
    rst          : in std_logic;
    ctrl_valid   : in std_logic;
    result_ready : in std_logic;
    -- outputs
    ctrl_ready   : out std_logic;
    result       : out std_logic_vector(BITWIDTH - 1 downto 0);
    result_valid : out std_logic);
end entity;

architecture arch of constant_node_#CST_NAME# is
begin
  result       <= #CST_VALUE#;
  result_valid <= ctrl_valid;
  ctrl_ready   <= result_ready;
end architecture;
