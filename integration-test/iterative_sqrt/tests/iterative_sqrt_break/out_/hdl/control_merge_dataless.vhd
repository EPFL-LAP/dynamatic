library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity control_merge_dataless is
  generic (
    SIZE        : integer;
    INDEX_TYPE : integer
  );
  port (
    clk, rst : in std_logic;
    -- input channels
    ins_valid : in  std_logic_vector(SIZE - 1 downto 0);
    ins_ready : out std_logic_vector(SIZE - 1 downto 0);
    -- data output channel
    outs_valid : out std_logic;
    outs_ready : in  std_logic;
    -- index output channel
    index       : out std_logic_vector(INDEX_TYPE - 1 downto 0);
    index_valid : out std_logic;
    index_ready : in  std_logic
  );
end entity;

architecture arch of control_merge_dataless is
  signal index_tehb                                               : std_logic_vector (INDEX_TYPE - 1 downto 0);
  signal dataAvailable, readyToFork, tehbOut_valid, tehbOut_ready : std_logic;
begin
  process (ins_valid)
  begin
    index_tehb <= (INDEX_TYPE - 1 downto 0 => '0');
    for i in 0 to (SIZE - 1) loop
      if (ins_valid(i) = '1') then
        index_tehb <= std_logic_vector(to_unsigned(i, INDEX_TYPE));
        exit;
      end if;
    end loop;
  end process;

  merge_ins : entity work.merge_dataless(arch) generic map (SIZE)
    port map(
      clk        => clk,
      rst        => rst,
      ins_valid  => ins_valid,
      outs_ready => tehbOut_ready,
      ins_ready  => ins_ready,
      outs_valid => dataAvailable
    );

  tehb : entity work.tehb(arch) generic map (INDEX_TYPE)
    port map(
      clk        => clk,
      rst        => rst,
      ins_valid  => dataAvailable,
      outs_ready => readyToFork,
      outs_valid => tehbOut_valid,
      ins_ready  => tehbOut_ready,
      ins        => index_tehb,
      outs       => index
    );

  fork_valid : entity work.fork_dataless(arch) generic map (2)
    port map(
      clk           => clk,
      rst           => rst,
      ins_valid     => tehbOut_valid,
      outs_ready(0) => outs_ready,
      outs_ready(1) => index_ready,
      ins_ready     => readyToFork,
      outs_valid(0) => outs_valid,
      outs_valid(1) => index_valid
    );
end architecture;
