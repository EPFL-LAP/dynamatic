library ieee;
use ieee.std_logic_1164.all;
use work.customTypes.all;

entity lazy_fork_node is generic (
  OUTPUTS  : integer;
  BITWIDTH : integer);
port (
  -- inputs
  ins        : in std_logic_vector(BITWIDTH - 1 downto 0);
  ins_valid  : in std_logic;
  clk        : in std_logic;
  rst        : in std_logic;
  outs_ready : in std_logic_vector(OUTPUTS - 1 downto 0);
  -- outputs
  ins_ready  : out std_logic;
  outs       : out data_array (OUTPUTS - 1 downto 0)(BITWIDTH - 1 downto 0);
  outs_valid : out std_logic_vector(OUTPUTS - 1 downto 0)
);

end entity;

architecture arch of lazy_fork_node is
  signal allnReady : std_logic;
begin

  genericAnd : entity work.andn generic map (OUTPUTS)
    port map(outs_ready, allnReady);

  valids : process (ins_valid, outs_ready, allnReady)
  begin
    for i in 0 to OUTPUTS - 1 loop
      outs_valid(i) <= ins_valid and allnReady;
    end loop;
  end process;

  ins_ready <= allnReady;

  process (ins)
  begin
    for I in 0 to OUTPUTS - 1 loop
      outs(I) <= ins;
    end loop;
  end process;
end arch;
