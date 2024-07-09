library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity tehb_dataless is
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

architecture arch of tehb_dataless is
  signal fullReg, outputValid : std_logic;
begin
  outputValid <= ins_valid or fullReg;

  process (clk, rst) is
  begin
    if (rst = '1') then
      fullReg <= '0';
    elsif (rising_edge(clk)) then
      fullReg <= outputValid and not outs_ready;
    end if;
  end process;

  ins_ready  <= not fullReg;
  outs_valid <= outputValid;
end architecture;
