library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity divsi_node is
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

architecture arch of divsi_node is

  component array_RAM_sdiv_32ns_32ns_32_36_1 is
    generic (
      ID         : integer;
      NUM_STAGE  : integer;
      din0_WIDTH : integer;
      din1_WIDTH : integer;
      dout_WIDTH : integer);
    port (
      clk   : in std_logic;
      reset : in std_logic;
      ce    : in std_logic;
      din0  : in std_logic_vector(din0_WIDTH - 1 downto 0);
      din1  : in std_logic_vector(din1_WIDTH - 1 downto 0);
      dout  : out std_logic_vector(dout_WIDTH - 1 downto 0));
  end component;

  signal join_valid : std_logic;
  signal out_array  : std_logic_vector(1 downto 0);

begin
  lhs_ready <= out_array(0);
  rhs_ready <= out_array(1);
  array_RAM_sdiv_32ns_32ns_32_36_1_U1 : component array_RAM_sdiv_32ns_32ns_32_36_1
    generic map(
      ID         => 1,
      NUM_STAGE  => 36,
      din0_WIDTH => 32,
      din1_WIDTH => 32,
      dout_WIDTH => 32)
    port map(
      clk   => clk,
      reset => rst,
      ce    => result_ready,
      din0  => lhs,
      din1  => rhs,
      dout  => result);

    join_write_temp : entity work.join(arch) generic map(2)
      port map(
      (lhs_valid,
        rhs_valid),
        result_ready,
        join_valid,
        out_array);

    buff : entity work.delay_buffer(arch)
      generic map(35)
      port map(
        clk,
        rst,
        join_valid,
        result_ready,
        result_valid);
  end architecture;
