library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity one_slot_break_dvr_dataless is 
  port (
    clk, rst : in std_logic;
    -- input channel
    ins_valid : in  std_logic;
    ins_ready : out std_logic;
    -- output channel
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;

architecture arch of one_slot_break_dvr_dataless is

  signal enable, stop : std_logic;
  signal outputValid, inputReady : std_logic;

begin

  p_ready : process(clk) is
  begin
    if (rising_edge(clk)) then
      if (rst = '1') then
        inputReady <= '1';
      else
        inputReady <= (not stop) and (not enable);
      end if;
    end if;
  end process; 

  p_valid : process(clk) is
  begin
    if (rising_edge(clk)) then
      if (rst = '1') then
        outputValid <= '0';
      else
        outputValid <= enable or stop;
      end if;
    end if;
  end process;

  enable <= ins_valid and inputReady;
  stop <= outputValid and not outs_ready;
  ins_ready <= inputReady;
  outs_valid <= outputValid;

end architecture;