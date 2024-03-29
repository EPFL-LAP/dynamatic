library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;
-- #PREDICATE# = oeq, ogt, oge, olt, ole, one, ord, ueq, ugt, uge, ult, ule, une, uno
-- #CONST# = 00001, 00010, 00011, 00100, 00101, 00110, 00111, 01000, 01001, 01010, 01011, 01100, 01101, 01110 
entity cmpf_node_#PREDICATE# is
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
    result       : out std_logic;
    result_valid : out std_logic);
end entity;

architecture arch of cmpf_node_#PREDICATE# is

  component array_RAM_fcmp_32cud is
    generic (
      ID         : integer := 1;
      NUM_STAGE  : integer := 2;
      din0_WIDTH : integer := 32;
      din1_WIDTH : integer := 32;
      dout_WIDTH : integer := 1
    );
    port (
      clk    : in std_logic;
      reset  : in std_logic;
      ce     : in std_logic;
      din0   : in std_logic_vector(din0_WIDTH - 1 downto 0);
      din1   : in std_logic_vector(din1_WIDTH - 1 downto 0);
      opcode : in std_logic_vector(4 downto 0);
      dout   : out std_logic_vector(dout_WIDTH - 1 downto 0)
    );
  end component;

  signal join_valid   : std_logic;
  constant alu_opcode : std_logic_vector(4 downto 0) := "#CONST#";
  signal out_array    : std_logic_vector(1 downto 0);

begin
  lhs_ready <= out_array(0);
  rhs_ready <= out_array(1);

  result <= '0';

  array_RAM_fcmp_32ns_32ns_1_2_1_u1 : component array_RAM_fcmp_32cud
    generic map(
      ID         => 1,
      NUM_STAGE  => 2,
      din0_WIDTH => 32,
      din1_WIDTH => 32,
      dout_WIDTH => 1)
    port map(
      clk     => clk,
      reset   => rst,
      din0    => lhs,
      din1    => rhs,
      ce      => result_ready,
      opcode  => alu_opcode,
      dout(0) => result);

    join_write_temp : entity work.join(arch) generic map(2)
      port map(
      (lhs_valid,
        rhs_valid),
        result_ready,
        join_valid,
        out_array);

    buff : entity work.delay_buffer(arch)
      generic map(1)
      port map(
        clk,
        rst,
        join_valid,
        result_ready,
        result_valid);
  end architecture;
