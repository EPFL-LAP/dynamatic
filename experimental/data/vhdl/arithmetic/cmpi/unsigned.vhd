library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;
-- #PREDICATE# = uge, ugt, ule, ult
-- #TYPEOP# = >=, >, <=, <
entity cmpi_#PREDICATE# is
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

architecture arch of cmpi_#PREDICATE# is
  signal join_valid : std_logic;
  signal one        : std_logic := "1";
  signal zero       : std_logic := "0";
begin

  join_write_temp : entity work.join(arch) generic map(2)
    port map(
    (lhs_valid,
      rhs_valid),
      result_ready,
      join_valid,
      (lhs_ready,
      rhs_ready));

  result <= one when (unsigned(lhs) #TYPEOP# unsigned(rhs)) else
    zero;
  result_valid <= join_valid;

end architecture;
