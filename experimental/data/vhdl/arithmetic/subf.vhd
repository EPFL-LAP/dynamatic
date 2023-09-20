library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity subf_node is
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

architecture arch of subf_node is

  component array_RAM_subf_32bkb is
    generic (
      ID         : integer := 1;
      NUM_STAGE  : integer := 10;
      din0_WIDTH : integer := 32;
      din1_WIDTH : integer := 32;
      dout_WIDTH : integer := 32
    );
    port (
      clk   : in std_logic;
      reset : in std_logic;
      ce    : in std_logic;
      din0  : in std_logic_vector(din0_WIDTH - 1 downto 0);
      din1  : in std_logic_vector(din1_WIDTH - 1 downto 0);
      dout  : out std_logic_vector(dout_WIDTH - 1 downto 0)
    );
  end component;

  signal join_valid : std_logic;

  signal buff_valid, oehb_valid, oehb_ready : std_logic;
  signal oehb_dataOut, oehb_datain          : std_logic;
  signal out_array                          : std_logic_vector(1 downto 0);

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

  buff : entity work.delay_buffer(arch) generic map(8)
    port map(
      clk,
      rst,
      join_valid,
      oehb_ready,
      buff_valid);

  oehb : entity work.OEHB(arch) generic map (BITWIDTH)
    port map(
      clk        => clk,
      rst        => rst,
      ins_valid  => buff_valid,
      outs_ready => result_ready,
      outs_valid => result_valid,
      ins_ready  => oehb_ready,
      ins        => oehb_datain,
      outs       => oehb_dataOut
    );

  array_RAM_fsub_32ns_32ns_32_10_full_dsp_1_U1 : component array_RAM_subf_32bkb
    port map(
      clk   => clk,
      reset => rst,
      ce    => oehb_ready,
      din0  => lhs,
      din1  => rhs,
      dout  => result);

  end architecture;
