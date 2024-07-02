library ieee;
use ieee.std_logic_1164.all;

entity lazy_fork_dataless is
  generic (
    SIZE : integer
  );
  port (
    clk, rst : in std_logic;
    -- input channel
    ins_valid : in  std_logic;
    ins_ready : out std_logic;
    -- output channels
    outs_valid : out std_logic_vector(SIZE - 1 downto 0);
    outs_ready : in  std_logic_vector(SIZE - 1 downto 0)
  );
end entity;

architecture arch of lazy_fork_dataless is
  signal allnReady : std_logic;
begin
  genericAnd : entity work.and_n generic map (SIZE)
    port map(outs_ready, allnReady);

  valids : process (ins_valid, outs_ready, allnReady)
  begin
    for i in 0 to SIZE - 1 loop
      outs_valid(i) <= ins_valid and allnReady;
    end loop;
  end process;

  ins_ready <= allnReady;
end architecture;
