library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity ENTITY_NAME is
  generic (
    DATA_TYPE : integer
  );
  port (
    -- inputs
    clk          : in std_logic;
    rst          : in std_logic;
    lhs          : in std_logic_vector(DATA_TYPE - 1 downto 0);
    lhs_valid    : in std_logic;
    rhs          : in std_logic_vector(DATA_TYPE - 1 downto 0);
    rhs_valid    : in std_logic;
    result_ready : in std_logic;
    -- outputs
    result       : out std_logic_vector(0 downto 0);
    result_valid : out std_logic;
    lhs_ready    : out std_logic;
    rhs_ready    : out std_logic
  );
begin
  assert DATA_TYPE=32
  report "ENTITY_NAME currently only supports 32-bit floating point operands"
  severity failure;
end entity;

architecture arch of ENTITY_NAME is

  signal ip_lhs : std_logic_vector(DATA_TYPE + 1 downto 0);
  signal ip_rhs : std_logic_vector(DATA_TYPE + 1 downto 0);

  signal ip_unordered : std_logic;
  signal ip_result : std_logic;

  constant cmp_predicate : string := "COMPARATOR";

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

  gen_flopoco_ip :
    if cmp_predicate = "OEQ" or cmp_predicate = "UEQ" generate
      operator : entity work.FloatingPointComparatorEQ(arch)
      port map(
        clk, '1', 
        ip_lhs, ip_rhs, ip_unordered, ip_result
      );
    elsif cmp_predicate = "OGT" or cmp_predicate = "UGT" generate 
      operator : entity work.FloatingPointComparatorGT(arch)
      port map(
        clk, '1', 
        ip_lhs, ip_rhs, ip_unordered, ip_result
      );
    elsif cmp_predicate = "OGE" or cmp_predicate = "UGE" generate 
      operator : entity work.FloatingPointComparatorGE(arch)
      port map(
        clk, '1', 
        ip_lhs, ip_rhs, ip_unordered, ip_result
      );
    elsif cmp_predicate = "OLT" or cmp_predicate = "ULT" generate 
      operator : entity work.FloatingPointComparatorLT(arch)
      port map(
        clk, '1', 
        ip_lhs, ip_rhs, ip_unordered, ip_result
      );
    elsif cmp_predicate = "OLE" or cmp_predicate = "ULE" generate 
      operator : entity work.FloatingPointComparatorLE(arch)
      port map(
        clk, '1', 
        ip_lhs, ip_rhs, ip_unordered, ip_result
      );
    elsif cmp_predicate = "ONE" or cmp_predicate = "UNE" generate 
      operator : entity work.FloatingPointComparatorEQ(arch)
      port map(
        clk, '1', 
        ip_lhs, ip_rhs, ip_unordered, ip_result
      );
    elsif cmp_predicate = "ORD" generate 
      -- This predicate only tests if all inputs are ordered,
      -- hence in principle any IP would work
      operator : entity work.FloatingPointComparatorEQ(arch)
      port map(
        clk, '1', 
        ip_lhs, ip_rhs, ip_unordered, ip_result
      );
    elsif cmp_predicate = "UNO" generate
      -- This predicate only tests if any input is unordered,
      -- hence in principle any IP would work
      operator : entity work.FloatingPointComparatorEQ(arch)
      port map(
        clk, '1', 
        ip_lhs, ip_rhs, ip_unordered, ip_result
      );
    else generate -- cmp_predicate = "UNO"
      assert false
      report "COMPARATOR is an invalid predicate!"
      severity failure;
    end generate;
     
  gen_result_signal :
    if
    cmp_predicate = "OEQ" or
    cmp_predicate = "OGT" or
    cmp_predicate = "OGE" or
    cmp_predicate = "OLT" or
    cmp_predicate = "OLE" generate
      result(0) <= not ip_unordered and ip_result;
    elsif
    cmp_predicate = "ONE" generate
      result(0) <= not ip_unordered and not ip_result;
    elsif cmp_predicate = "ORD" generate
      result(0) <= not ip_unordered;
    elsif 
    cmp_predicate = "UEQ" or
    cmp_predicate = "UGT" or
    cmp_predicate = "UGE" or
    cmp_predicate = "ULT" or
    cmp_predicate = "ULE" generate
      result(0) <= ip_unordered or ip_result;
    elsif
    cmp_predicate = "UNE" generate
      result(0) <= ip_unordered or not ip_result;
    elsif
    cmp_predicate = "UNO" generate
      result(0) <= ip_unordered;
    else generate 
      assert false
      report "COMPARATOR is an invalid predicate!"
      severity failure;
    end generate;

end architecture;
