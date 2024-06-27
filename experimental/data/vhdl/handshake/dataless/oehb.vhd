library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity oehb_dataless is
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

architecture arch of oehb_dataless is
  signal full_reg, mux_sel : std_logic;
begin
  process (clk, rst) is
  begin
    if (rst = '1') then
      outs_valid <= '0';
    elsif (rising_edge(clk)) then
      outs_valid <= ins_valid or not ins_ready;
    end if;
  end process;

  ins_ready <= not outs_valid or outs_ready;
end architecture;
