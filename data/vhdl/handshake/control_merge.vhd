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
    clk, rst : in std_logic;
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
  signal index_internal : std_logic_vector(DATA_TYPE - 1 downto 0);
begin
  control : entity work.control_merge_dataless
    port map(
      clk         => clk,
      rst         => rst,
      ins_valid   => ins_valid,
      ins_ready   => ins_ready,
      outs_valid  => outs_valid,
      outs_ready  => outs_ready,
      index       => index_internal,
      index_valid => index_valid,
      index_ready => index_ready
    );

  index <= index_internal;
  outs  <= ins(index_internal);
end architecture;
