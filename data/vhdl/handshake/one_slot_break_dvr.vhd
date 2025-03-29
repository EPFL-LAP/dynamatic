library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity one_slot_break_dvr is
  generic (
    DATA_TYPE : integer
  );
  port (
    clk, rst : in std_logic;
    -- input channel
    ins       : in  std_logic_vector(DATA_TYPE - 1 downto 0);
    ins_valid : in  std_logic;
    ins_ready : out std_logic;
    -- output channel
    outs       : out std_logic_vector(DATA_TYPE - 1 downto 0);
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;

architecture arch of one_slot_break_dvr is
  signal enable, inputReady : std_logic;
  signal dataReg: std_logic_vector(DATA_TYPE - 1 downto 0);
begin

  control : entity work.one_slot_break_dvr_dataless
    port map(
      clk        => clk,
      rst        => rst,
      ins_valid  => ins_valid,
      ins_ready  => inputReady,
      outs_valid => outs_valid,
      outs_ready => outs_ready
    );

  p_data : process (clk) is
  begin
    if (rising_edge(clk)) then
      if (rst = '1') then
        dataReg <= (others => '0');
      elsif (enable) then
        dataReg <= ins;
      end if;
    end if;
  end process;

  ins_ready <= inputReady;
  enable <= ins_valid and inputReady;
  outs <= dataReg;

end architecture;