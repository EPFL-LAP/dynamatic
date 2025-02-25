library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

entity merge is
  generic (
    SIZE     : integer;
    DATA_TYPE : integer
  );
  port (
    clk, rst : in std_logic;
    -- input channels
    ins       : in  data_array(SIZE - 1 downto 0)(DATA_TYPE - 1 downto 0);
    ins_valid : in  std_logic_vector(SIZE - 1 downto 0);
    ins_ready : out std_logic_vector(SIZE - 1 downto 0);
    -- output channel
    outs       : out std_logic_vector(DATA_TYPE - 1 downto 0);
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;

architecture arch of merge is
  signal tehb_data_in : std_logic_vector(DATA_TYPE - 1 downto 0);
  signal tehb_pvalid  : std_logic;
  signal tehb_ready   : std_logic;
begin
  
  merge_ins : entity work.merge_notehb(arch) generic map (SIZE, DATA_TYPE)
    port map(
      clk        => clk,
      rst        => rst,
      ins        => ins,
      ins_valid  => ins_valid,
      outs_ready => tehb_ready,
      ins_ready  => ins_ready,
      outs       => tehb_data_in,
      outs_valid => tehb_pvalid
    );

  tehb : entity work.tehb(arch) generic map (DATA_TYPE)
    port map(
      clk        => clk,
      rst        => rst,
      ins_valid  => tehb_pvalid,
      outs_ready => outs_ready,
      outs_valid => outs_valid,
      ins_ready  => tehb_ready,
      ins        => tehb_data_in,
      outs       => outs
    );
end architecture;
