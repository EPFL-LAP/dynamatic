library ieee;
use ieee.std_logic_1164.all;
use work.types.all;

entity gate is
  generic (
    SIZE      : integer;
    DATA_TYPE : integer
  );
  port (
    clk, rst : in std_logic;
    -- data input channel
    ins : in data_array(0 downto 0)(DATA_TYPE - 1 downto 0);
    -- ins_data_valid : in std_logic;
    -- ins_data_ready : out std_logic;
    -- control input channels
    ins_valid : in  std_logic_vector(SIZE - 1 downto 0);
    ins_ready : out std_logic_vector(SIZE - 1 downto 0);
    -- output channel
    outs : out std_logic_vector(DATA_TYPE - 1 downto 0);
    outs_valid : out std_logic;
    outs_ready : in  std_logic

  );
end gate;

architecture arch of gate is
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
