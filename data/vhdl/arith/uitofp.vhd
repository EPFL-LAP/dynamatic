library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.float_pkg.all;

entity uitofp is
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
    outs       : out std_logic_vector(32 - 1 downto 0);
    outs_valid : out std_logic;
    ins_ready  : out std_logic
  );
begin
end entity;

architecture arch of uitofp is
  constant LATENCY : integer := 5;
  signal converted : std_logic_vector(32 - 1 downto 0);
  signal q0 : std_logic_vector(32 - 1 downto 0);
  signal q1 : std_logic_vector(32 - 1 downto 0);
  signal q2 : std_logic_vector(32 - 1 downto 0);
  signal q3 : std_logic_vector(32 - 1 downto 0);
  signal q4 : std_logic_vector(32 - 1 downto 0);
  signal buff_valid, oehb_ready : std_logic;
  signal oehb_dataOut, oehb_datain : std_logic;
  signal float_value : float32;
begin

  float_value <= to_float(resize(unsigned(ins), 32));
  converted <= to_std_logic_vector(float_value);
  outs <= q4;

  -- NOTE: This 5 stage latency is used to imitate the latency of the Vitis
  -- HLS for Kintex-7 FPGA devices. This should be revised.
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

  -- This OEHB is necessary to make the unit elastic. If not placed (i.e.,
  -- adding one more cycle of latency), this unit would deadlock in a fork-join
  -- pattern.
  oehb : entity work.oehb(arch) generic map (1)
    port map(
      clk        => clk,
      rst        => rst,
      ins_valid  => buff_valid,
      outs_ready => outs_ready,
      outs_valid => outs_valid,
      ins_ready  => oehb_ready,
      ins(0)     => oehb_datain,
      outs(0)    => oehb_dataOut
    );
  ins_ready <= oehb_ready;

end architecture;
