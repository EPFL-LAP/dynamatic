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
begin
  process (ins_valid, ins, outs_ready)
    variable tmp_data_out  : unsigned(DATA_TYPE - 1 downto 0);
    variable tmp_valid_out : std_logic;
    variable tmp_ready_out : std_logic_vector(SIZE - 1 downto 0);
  begin
    tmp_data_out  := unsigned(ins(0));
    tmp_valid_out := '0';
    tmp_ready_out := (others => '0');

    for I in 0 to (SIZE - 1) loop
      if (ins_valid(I) = '1') then
        tmp_data_out  := unsigned(ins(I));
        tmp_valid_out := '1';
        tmp_ready_out(i) := outs_ready;
        exit;
      end if;
    end loop;
    
    -- The outs channel is not persistent, meaning the data payload 
    -- may change while valid remains high
    outs <= std_logic_vector(resize(tmp_data_out, DATA_TYPE));
    outs_valid  <= tmp_valid_out;
    ins_ready <= tmp_ready_out;
  end process;
  
end architecture;
