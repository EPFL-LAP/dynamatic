library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

entity fork_dataless_with_tag is
  generic (
    SIZE : integer
  );
  port (
    clk, rst : in std_logic;
    -- input channel
    ins_valid : in  std_logic;
    ins_spec_tag : in std_logic;
    ins_ready : out std_logic;
    -- output channels
    outs_valid : out std_logic_vector(SIZE - 1 downto 0);
    outs_spec_tag : out std_logic_vector(SIZE - 1 downto 0);
    outs_ready : in  std_logic_vector(SIZE - 1 downto 0)
  );
end entity;

architecture arch of fork_dataless_with_tag is
  signal ins : std_logic_vector(0 downto 0);
  signal outs : data_array (SIZE - 1 downto 0)(0 downto 0);
begin
  -- This is actually fork with 1-bit data
  ins(0) <= ins_spec_tag;
  process(outs)
  begin
    for i in 0 to SIZE - 1 loop
      outs_spec_tag(i) <= outs(i)(0);
    end loop;
  end process;
  fork_inner : entity work.handshake_fork(arch)
    generic map(
      SIZE => SIZE,
      DATA_TYPE => 1
    )
    port map(
      clk        => clk,
      rst        => rst,
      ins        => ins,
      ins_valid  => ins_valid,
      ins_ready  => ins_ready,
      outs       => outs,
      outs_valid => outs_valid,
      outs_ready => outs_ready
    );
end architecture;