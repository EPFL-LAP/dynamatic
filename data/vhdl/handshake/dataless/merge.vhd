library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity merge_dataless is
  generic (
    SIZE : integer
  );
  port (
    clk, rst : in std_logic;
    -- input channels
    ins_valid : in  std_logic_vector(SIZE - 1 downto 0);
    ins_ready : out std_logic_vector(SIZE - 1 downto 0);
    -- output channel
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;

architecture arch of merge_dataless is
  signal tehb_pvalid : std_logic;
  signal tehb_ready  : std_logic;
begin
  merge_ins : entity work.merge_notehb_dataless(arch) generic map (SIZE)
    port map(
      clk        => clk,
      rst        => rst,
      ins_valid  => ins_valid,
      outs_ready => tehb_ready,
      ins_ready  => ins_ready,
      outs_valid => tehb_pvalid
    );

  tehb : entity work.tehb_dataless(arch)
    port map(
      clk        => clk,
      rst        => rst,
      ins_valid  => tehb_pvalid,
      outs_ready => outs_ready,
      outs_valid => outs_valid,
      ins_ready  => tehb_ready
    );
end architecture;
