library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

entity end_sync is
  generic (
    DATA_WIDTH   : integer;
    NUM_MEMORIES : integer
  );
  port (
    clk, rst : in std_logic;
    -- input channel
    ins       : in  data_array(0 downto 0)(DATA_WIDTH - 1 downto 0);
    ins_valid : in  std_logic_vector(0 downto 0);
    ins_ready : out std_logic_vector(0 downto 0);
    -- memory input channels
    memDone_valid : in  std_logic_vector(NUM_MEMORIES - 1 downto 0) := (others => '1');
    memDone_ready : out std_logic_vector(NUM_MEMORIES - 1 downto 0);
    -- output channel
    outs       : out data_array(0 downto 0)(DATA_WIDTH - 1 downto 0);
    outs_valid : out std_logic_vector(0 downto 0);
    outs_ready : in  std_logic_vector(0 downto 0)
  );
end entity;

architecture arch of end_sync is
begin
  control : entity work.end_sync_dataless(arch) generic map (NUM_MEMORIES)
    port map(
      clk           => clk,
      rst           => rst,
      ins_valid     => ins_valid,
      ins_ready     => ins_ready,
      memDone_valid => memDone_valid,
      memDone_ready => memDone_ready,
      outs_valid    => outs_valid,
      outs_ready    => outs_ready
    );

  outs(0) <= ins(0);
end architecture;
