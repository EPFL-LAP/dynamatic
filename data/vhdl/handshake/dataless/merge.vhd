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
  process (ins_valid)
    variable tmp_valid_out : std_logic;
  begin
    tmp_valid_out := '0';
    for i in SIZE - 1 downto 0 loop
      if (ins_valid(i) = '1') then
        tmp_valid_out := ins_valid(I);
      end if;
    end loop;
    tehb_pvalid <= tmp_valid_out;
  end process;

  process (tehb_ready)
  begin
    for i in 0 to SIZE - 1 loop
      ins_ready(i) <= tehb_ready;
    end loop;
  end process;

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
