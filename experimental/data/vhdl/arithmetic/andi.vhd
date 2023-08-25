library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity andi is
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

architecture arch of andi is

  signal join_valid : std_logic;

begin

  join_write_temp : entity work.join(arch) generic map(2)
    port map(
    (lhs_valid,
      rhs_valid),
      result_ready,
      join_valid,
      (lhs_ready,
      rhs_ready));
  result       <= lhs and rhs;
  result_valid <= join_valid;
end architecture;
