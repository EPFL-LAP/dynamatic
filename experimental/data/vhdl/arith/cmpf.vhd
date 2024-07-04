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
end entity;

architecture arch of ENTITY_NAME is

  component array_RAM_fcmp_32cud is
    generic (
      ID         : integer := 1;
      NUM_STAGE  : integer := 2;
      din0_WIDTH : integer := 32;
      din1_WIDTH : integer := 32;
      dout_WIDTH : integer := 1
    );
    port (
      clk    : in  std_logic;
      reset  : in  std_logic;
      ce     : in  std_logic;
      din0   : in  std_logic_vector(din0_WIDTH - 1 downto 0);
      din1   : in  std_logic_vector(din1_WIDTH - 1 downto 0);
      opcode : in  std_logic_vector(4 downto 0);
      dout   : out std_logic_vector(dout_WIDTH - 1 downto 0)
    );
  end component;

  signal join_valid   : std_logic;
  constant alu_opcode : std_logic_vector(4 downto 0) := "COMPARATOR";

begin
  join_inputs : entity work.join(arch) generic map(2)
    port map(
      -- inputs
      ins_valid(0) => lhs_valid,
      ins_valid(1) => rhs_valid,
      outs_ready   => result_ready,
      -- outputs
      outs_valid   => join_valid,
      ins_ready(0) => lhs_ready,
      ins_ready(1) => rhs_ready
    );

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
      dout(0) => result(0)
    );

  buff : entity work.delay_buffer(arch) generic map(1)
    port map(
      clk,
      rst,
      join_valid,
      result_ready,
      result_valid
    );
end architecture;
