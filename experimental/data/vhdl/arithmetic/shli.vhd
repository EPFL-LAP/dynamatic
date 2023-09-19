library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity shli_node is
  generic (
    BITWIDTH : integer
  );
  port (
    -- inputs
    clk          : in std_logic;
    rst          : in std_logic;
    lhs          : in std_logic_vector(BITWIDTH - 1 downto 0);
    lhs_valid    : in std_logic;
    rhs          : in std_logic_vector(BITWIDTH - 1 downto 0);
    rhs_valid    : in std_logic;
    result_ready : in std_logic;
    -- outputs
    lhs_ready    : out std_logic;
    rhs_ready    : out std_logic;
    result       : out std_logic_vector(BITWIDTH - 1 downto 0);
    result_valid : out std_logic);
end entity;

architecture arch of shli_node is

  signal join_valid : std_logic;
  signal out_array  : std_logic_vector(1 downto 0);

begin
  out_array(0) <= lhs_ready;
  out_array(1) <= rhs_ready;

  join_write_temp : entity work.join(arch) generic map(2)
    port map(
    (lhs_valid,
      rhs_valid),
      result_ready,
      join_valid,
      out_array);
  result       <= std_logic_vector(shift_left(unsigned(lhs), to_integer(unsigned('0' & rhs(BITWIDTH - 2 downto 0)))));
  result_valid <= join_valid;
end architecture;
