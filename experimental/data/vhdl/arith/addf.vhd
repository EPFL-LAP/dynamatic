library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity addf is
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
    result       : out std_logic_vector(BITWIDTH - 1 downto 0);
    result_valid : out std_logic;
    lhs_ready    : out std_logic;
    rhs_ready    : out std_logic
  );
end entity;

architecture arch of addf is

  component array_RAM_addf_32bkb is
    generic (
      ID         : integer := 1;
      NUM_STAGE  : integer := 10;
      din0_WIDTH : integer := 32;
      din1_WIDTH : integer := 32;
      dout_WIDTH : integer := 32
    );
    port (
      clk   : in  std_logic;
      reset : in  std_logic;
      ce    : in  std_logic;
      din0  : in  std_logic_vector(din0_WIDTH - 1 downto 0);
      din1  : in  std_logic_vector(din1_WIDTH - 1 downto 0);
      dout  : out std_logic_vector(dout_WIDTH - 1 downto 0)
    );
  end component;

  signal join_valid                         : std_logic;
  signal out_array                          : std_logic_vector(1 downto 0);
  signal buff_valid, oehb_valid, oehb_ready : std_logic;
  signal oehb_dataOut, oehb_datain          : std_logic;

begin
  join_inputs : entity work.join(arch) generic map(2)
    port map(
      -- inputs
      ins_valid(0) => lhs_valid,
      ins_valid(1) => rhs_valid,
      outs_ready   => oehb_ready,
      -- outputs
      outs_valid   => join_valid,
      ins_ready(0) => lhs_ready,
      ins_ready(1) => rhs_ready
    );

  buff : entity work.delay_buffer(arch) generic map(8)
    port map(
      clk,
      rst,
      join_valid,
      oehb_ready,
      buff_valid
    );

  oehb : entity work.oehb(arch) generic map(1)
    port map(
      clk        => clk,
      rst        => rst,
      ins_valid  => buff_valid,
      outs_ready => result_ready,
      outs_valid => result_valid,
      ins_ready  => oehb_ready,
      ins(0)     => oehb_datain,
      outs(0)    => oehb_dataOut
    );

  array_RAM_fadd_32ns_32ns_32_10_full_dsp_1_U1 : component array_RAM_addf_32bkb
    port map(
      clk   => clk,
      reset => rst,
      ce    => oehb_ready,
      din0  => lhs,
      din1  => rhs,
      dout  => result
    );

end architecture;
