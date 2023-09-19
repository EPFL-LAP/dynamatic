library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity muli_node is
  generic (
    BITWIDTH : integer
  );
  port (
    clk : in std_logic;
    rst : in std_logic;

    lhs       : in std_logic_vector(BITWIDTH - 1 downto 0);
    lhs_valid : in std_logic;
    lhs_ready : out std_logic;

    rhs       : in std_logic_vector(BITWIDTH - 1 downto 0);
    rhs_valid : in std_logic;
    rhs_ready : out std_logic;

    result       : out std_logic_vector(BITWIDTH - 1 downto 0);
    result_valid : out std_logic;
    result_ready : in std_logic);
end entity;

architecture arch of muli_node is

  signal join_valid : std_logic;

  signal buff_valid, oehb_valid, oehb_ready : std_logic;
  signal oehb_dataOut, oehb_datain          : std_logic_vector(BITWIDTH - 1 downto 0);

  constant LATENCY : integer := 4;
  signal out_array : std_logic_vector(1 downto 0);

begin
  out_array(0) <= lhs_ready;
  out_array(1) <= rhs_ready;
  join : entity work.join(arch) generic map(2)
    port map(
    (lhs_valid,
      rhs_valid),
      oehb_ready,
      join_valid,
      out_array);
  multiply_unit : entity work.mul_4_stage(behav) generic map (BITWIDTH)

    port map(
      clk => clk,
      ce  => oehb_ready,
      a   => lhs,
      b   => rhs,
      p   => result);

  buff : entity work.delay_buffer(arch) generic map(LATENCY - 1)
    port map(
      clk,
      rst,
      join_valid,
      oehb_ready,
      buff_valid);

  oehb : entity work.OEHB(arch) generic map (1)
    port map(
      clk        => clk,
      rst        => rst,
      ins_valid  => buff_valid,
      outs_ready => result_ready,
      outs_valid => result_valid,
      ins_ready  => oehb_ready,
      ins        => oehb_datain,
      outs       => oehb_dataOut);
end architecture;
