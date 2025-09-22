-- handshake_mux_3 : mux({'size': 2, 'data_bitwidth': 12, 'index_bitwidth': 1, 'extra_signals': {}})


library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.math_real.all;
use work.types.all;

-- Entity of mux
entity handshake_mux_3 is
  port (
    clk : in std_logic;
    rst : in std_logic;
    -- data input channels
    ins       : in  data_array(2 - 1 downto 0)(12 - 1 downto 0);
    ins_valid : in  std_logic_vector(2 - 1 downto 0);
    ins_ready : out std_logic_vector(2 - 1 downto 0);
    -- index input channel
    index       : in  std_logic_vector(1 - 1 downto 0);
    index_valid : in  std_logic;
    index_ready : out std_logic;
    -- output channel
    outs       : out std_logic_vector(12 - 1 downto 0);
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;

-- Architecture of mux
architecture arch of handshake_mux_3 is
begin
  process (ins, ins_valid, outs_ready, index, index_valid)
    variable selectedData                   : std_logic_vector(12 - 1 downto 0);
    variable selectedData_valid, indexEqual : std_logic;
  begin
    selectedData       := ins(0);
    selectedData_valid := '0';

    for i in 2 - 1 downto 0 loop
      if unsigned(index) = to_unsigned(i, index'length) then
        indexEqual := '1';
      else
        indexEqual := '0';
      end if;
      if indexEqual and index_valid and ins_valid(i) then
        selectedData       := ins(i);
        selectedData_valid := '1';
      end if;
      ins_ready(i) <= (indexEqual and index_valid and ins_valid(i) and outs_ready) or (not ins_valid(i));
    end loop;

    index_ready <= (not index_valid) or (selectedData_valid and outs_ready);
    outs        <= selectedData;
    outs_valid  <= selectedData_valid;
  end process;
end architecture;

