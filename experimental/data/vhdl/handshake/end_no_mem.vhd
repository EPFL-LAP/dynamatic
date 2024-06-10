library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

entity end_sync_no_mem is
  generic (
    BITWIDTH   : integer;
    MEM_INPUTS : integer
  );
  port (
    -- inputs
    clk, rst   : in std_logic;
    ins        : in data_array(0 downto 0)(BITWIDTH - 1 downto 0);
    ins_valid  : in std_logic_vector(0 downto 0);
    outs_ready : in std_logic_vector(0 downto 0);
    -- outputs
    outs       : out data_array(0 downto 0)(BITWIDTH - 1 downto 0);
    outs_valid : out std_logic_vector(0 downto 0);
    ins_ready  : out std_logic_vector(0 downto 0)
  );
end entity;

architecture arch of end_sync_no_mem is
  signal valid : std_logic;
begin

  process (ins_valid, ins)
    variable tmp_data_out  : unsigned(BITWIDTH - 1 downto 0);
    variable tmp_valid_out : std_logic;
  begin
    tmp_data_out  := unsigned((ins(0)));
    tmp_valid_out := '0';
    if (ins_valid(0) = '1') then
      tmp_data_out  := unsigned(ins(0));
      tmp_valid_out := ins_valid(0);
    end if;
    outs(0) <= std_logic_vector(resize(tmp_data_out, BITWIDTH));
    valid   <= tmp_valid_out;
  end process;

  outs_valid(0) <= valid;
  ins_ready(0)  <= ins_valid(0) and outs_ready(0);
end architecture;
