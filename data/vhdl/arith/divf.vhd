library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity divf is
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
    result       : out std_logic_vector(DATA_TYPE - 1 downto 0);
    result_valid : out std_logic;
    lhs_ready    : out std_logic;
    rhs_ready    : out std_logic
  );
begin
  assert DATA_TYPE=32
  report "divf currently only supports 32-bit floating point operands"
  severity failure;
end entity;

architecture arch of divf is

  signal join_valid                         : std_logic;
  signal buff_valid, oehb_valid, oehb_ready : std_logic;

  -- intermediate input signals for IEEE-754 to Flopoco-simple-float conversion
  signal ip_lhs, ip_rhs : std_logic_vector(DATA_TYPE + 1 downto 0);

  -- intermediate output signal for Flopoco-simple-float to IEEE-754 conversion
  signal ip_result : std_logic_vector(DATA_TYPE + 1 downto 0);

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

  buff : entity work.delay_buffer(arch) generic map(28)
    port map(
      clk,
      rst,
      join_valid,
      oehb_ready,
      buff_valid
    );

  oehb : entity work.oehb(arch) generic map (1)
    port map(
      clk        => clk,
      rst        => rst,
      ins_valid  => buff_valid,
      outs_ready => result_ready,
      outs_valid => result_valid,
      --outputs
      ins_ready => oehb_ready,
      ins(0)    => '0',
      outs   => open
    );

  ieee2nfloat_lhs: entity work.InputIEEE_32bit(arch)
    port map (
        X => lhs,
        R => ip_lhs
    );

  ieee2nfloat_rhs: entity work.InputIEEE_32bit(arch)
    port map (
        X => rhs,
        R => ip_rhs
    );

  nfloat2ieee_result : entity work.OutputIEEE_32bit(arch)
    port map (
        X => ip_result,
        R => result
    );

  ip : entity work.FloatingPointDivider(arch)
    port map (
        clk => clk,
        ce  => oehb_ready,
        X   => ip_lhs,
        Y   => ip_rhs,
        R   => ip_result
    );

end architecture;
