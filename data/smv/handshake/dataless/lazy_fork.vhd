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

  valids : process (ins_valid, outs_ready)
    variable tmp_ready : std_logic_vector(SIZE - 1 downto 0);
  begin
    for i in tmp_ready'range loop
      tmp_ready(i) := '1';
      for j in outs_ready'range loop
        if i /= j then
          tmp_ready(i) := (tmp_ready(i) and outs_ready(j));
        end if;
      end loop;
    end loop;
    for i in outs_valid'range loop
      outs_valid(i) <= ins_valid and tmp_ready(i);
    end loop;
  end process;

  ins_ready <= allnReady;
end architecture;
