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

  signal ip_lhs : std_logic_vector(BITWIDTH + 1 downto 0);
  signal ip_rhs : std_logic_vector(BITWIDTH + 1 downto 0);

  signal ip_unordered : std_logic;
  signal ip_result : std_logic;

  constant alu_opcode : std_logic_vector(4 downto 0) := "COMPARATOR";

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
    if alu_opcode = "00001" or alu_opcode = "01000" generate
      operator : entity work.FloatingPointComparatorEQ(arch)
      port map(
        clk, '1', 
        ip_lhs, ip_rhs, ip_unordered, ip_result
      );
    elsif alu_opcode = "00010" or alu_opcode = "01001" generate 
      operator : entity work.FloatingPointComparatorGT(arch)
      port map(
        clk, '1', 
        ip_lhs, ip_rhs, ip_unordered, ip_result
      );
    elsif alu_opcode = "00011" or alu_opcode = "01010" generate 
      operator : entity work.FloatingPointComparatorGE(arch)
      port map(
        clk, '1', 
        ip_lhs, ip_rhs, ip_unordered, ip_result
      );
    elsif alu_opcode = "00100" or alu_opcode = "01011" generate 
      operator : entity work.FloatingPointComparatorLT(arch)
      port map(
        clk, '1', 
        ip_lhs, ip_rhs, ip_unordered, ip_result
      );
    elsif alu_opcode = "00101" or alu_opcode = "01100" generate 
      operator : entity work.FloatingPointComparatorLE(arch)
      port map(
        clk, '1', 
        ip_lhs, ip_rhs, ip_unordered, ip_result
      );
    elsif alu_opcode = "00110" or alu_opcode = "01101" generate 
      operator : entity work.FloatingPointComparatorEQ(arch)
      port map(
        clk, '1', 
        ip_lhs, ip_rhs, ip_unordered, ip_result
      );
    elsif alu_opcode = "00111" generate 
      operator : entity work.FloatingPointComparatorEQ(arch)
      port map(
        clk, '1', 
        ip_lhs, ip_rhs, ip_unordered, ip_result
      );
    else generate 
      operator : entity work.FloatingPointComparatorEQ(arch)
      port map(
        clk, '1', 
        ip_lhs, ip_rhs, ip_unordered, ip_result
      );
    end generate;
     
  gen_result_signal :
    if
    alu_opcode = "00001" or
    alu_opcode = "00010" or
    alu_opcode = "00011" or
    alu_opcode = "00100" or
    alu_opcode = "00101" generate
      result(0) <= not ip_unordered and ip_result;
    elsif
    alu_opcode = "00110" generate
      result(0) <= not ip_unordered and not ip_result;
    elsif alu_opcode = "00111" generate
      result(0) <= not ip_unordered;
    elsif 
    alu_opcode = "01000" or
    alu_opcode = "01001" or
    alu_opcode = "01010" or
    alu_opcode = "01011" or
    alu_opcode = "01100" generate
      result(0) <= ip_unordered and ip_result;
    elsif
    alu_opcode = "01101" generate
      result(0) <= ip_unordered and not ip_result;
    else generate -- "01110" UNO
      result(0) <= ip_unordered;
    end generate;

end architecture;
