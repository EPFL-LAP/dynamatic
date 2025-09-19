library ieee;
use ieee.std_logic_1164.all;
use work.types.all;

entity blocker is
  generic (
    SIZE      : integer;
    DATA_TYPE : integer
  );
  port (
    clk, rst : in std_logic;
    -- input channels
    ins       : in data_array(SIZE - 1 downto 0)(31 downto 0);
    ins_valid : in  std_logic_vector(SIZE - 1 downto 0);
    ins_ready : out std_logic_vector(SIZE - 1 downto 0);
    -- output channel
    outs : out std_logic_vector(31 downto 0);
    outs_valid : out std_logic;
    outs_ready : in  std_logic

  );
end blocker;

architecture arch of blocker is
begin
  join_inner : entity work.join(arch) generic map(SIZE)
    port map(
      -- inputs
      ins_valid  => ins_valid,
      outs_ready => outs_ready,
      -- outputs
      outs_valid => outs_valid,
      ins_ready  => ins_ready
    );

    -- simple data pass-through
  outs <= ins(0);
end architecture;
