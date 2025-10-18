library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

entity control_merge is
  generic (
    SIZE        : integer;
    DATA_TYPE  : integer;
    INDEX_TYPE : integer
  );
  port (
    clk : in  std_logic;
    rst : in std_logic;
    -- input channels
    ins       : in  data_array(SIZE - 1 downto 0)(DATA_TYPE - 1 downto 0);
    ins_valid : in  std_logic_vector(SIZE - 1 downto 0);
    ins_ready : out std_logic_vector(SIZE - 1 downto 0);
    -- data output channel
    outs       : out std_logic_vector(DATA_TYPE - 1 downto 0);
    outs_valid : out std_logic;
    outs_ready : in  std_logic;
    -- index output channel
    index       : out std_logic_vector(INDEX_TYPE - 1 downto 0);
    index_valid : out std_logic;
    index_ready : in  std_logic
  );
end entity;

architecture arch of control_merge is
begin
  assert false
  report "control_merge implementation with data signal has a bug. Use beta backend instead"
  severity failure;
end architecture;
