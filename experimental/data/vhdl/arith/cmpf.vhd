library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity ENTITY_NAME is
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
    result       : out std_logic_vector(0 downto 0);
    result_valid : out std_logic;
    lhs_ready    : out std_logic;
    rhs_ready    : out std_logic
  );
begin
  assert BITWIDTH=32
  report "ENTITY_NAME currently only support 32-bit floating point operands"
  severity failure;
end entity;

architecture arch of ENTITY_NAME is

  signal ip_lhs : std_logic_vector(BIT_WIDTH downto 0);
  signal ip_rhs : std_logic_vector(BIT_WIDTH downto 0);

  signal ip_unordered : std_logic;
  signal ip_result : std_logic;

begin
  join_inputs : entity work.join(arch) generic map(2)
    port map(
      -- inputs
      ins_valid(0) => lhs_valid,
      ins_valid(1) => rhs_valid,
      outs_ready   => result_ready,
      -- outputs
      outs_valid   => result_valid,
      ins_ready(0) => lhs_ready,
      ins_ready(1) => rhs_ready
    );

  ieee2nfloat_0: entity work.InputIEEE_32bit(arch)
    port map (
        --input
        X => lhs,
        --output
        R => ip_lhs
    );

  ieee2nfloat_1: entity work.InputIEEE_32bit(arch)
    port map (
        --input
        X => rhs,
        --output
        R => ip_rhs
    );

  operator : entity work.FloatingPointComparatorCOMPARATOR(arch)
  port map(
    clk, "1", 
    ip_lhs, ip_rhs, ip_unordered, ip_result
  );
  result <= not ip_unordered and ip_result;

end architecture;
