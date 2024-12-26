library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.float_pkg.all;

entity fptosi is
  generic (
    DATA_TYPE : integer
  );
  port (
    -- inputs
    clk        : in std_logic;
    rst        : in std_logic;
    ins        : in std_logic_vector(DATA_TYPE - 1 downto 0);
    ins_valid  : in std_logic;
    outs_ready : in std_logic;
    -- outputs
    outs       : out std_logic_vector(DATA_TYPE - 1 downto 0);
    outs_valid : out std_logic;
    ins_ready  : out std_logic
  );
begin
  assert DATA_TYPE=32
  report "fptosi currently only supports 32-bit floating point operands"
  severity failure;
end entity;

architecture arch of fptosi is
  constant LATENCY : integer := 5;
  signal converted : std_logic_vector(DATA_TYPE - 1 downto 0);
  signal q0 : std_logic_vector(DATA_TYPE - 1 downto 0);
  signal q1 : std_logic_vector(DATA_TYPE - 1 downto 0);
  signal q2 : std_logic_vector(DATA_TYPE - 1 downto 0);
  signal q3 : std_logic_vector(DATA_TYPE - 1 downto 0);
  signal q4 : std_logic_vector(DATA_TYPE - 1 downto 0);
  signal buff_valid, oehb_ready : std_logic;
  signal oehb_dataOut, oehb_datain          : std_logic_vector(DATA_TYPE - 1 downto 0);
  signal float_value : float32;
begin

  float_value <= to_float(ins);
  converted <= std_logic_vector(to_signed(float_value, 32));
  outs <= q4;

  process (clk)
  begin
    if (clk'event and clk = '1') then
      if (rst) then
        q0 <= (others => '0');
        q1 <= (others => '0');
        q2 <= (others => '0');
        q3 <= (others => '0');
        q4 <= (others => '0');
      elsif (oehb_ready) then
        q0 <= converted;
        q1 <= q0;
        q2 <= q1;
        q3 <= q2;
        q4 <= q3;
      end if;
    end if;
  end process;

  buff : entity work.delay_buffer(arch) generic map(LATENCY - 1)
    port map(
      clk,
      rst,
      ins_valid,
      oehb_ready,
      buff_valid
    );

  oehb : entity work.oehb(arch) generic map (DATA_TYPE)
    port map(
      clk        => clk,
      rst        => rst,
      ins_valid  => buff_valid,
      outs_ready => outs_ready,
      outs_valid => outs_valid,
      ins_ready  => oehb_ready,
      ins        => oehb_datain,
      outs       => oehb_dataOut
    );

  ins_ready <= oehb_ready;

end architecture;
