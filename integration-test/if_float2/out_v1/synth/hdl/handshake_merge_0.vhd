-- handshake_merge_0 : merge({'size': 2, 'bitwidth': 3, 'extra_signals': {}})


library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

-- Entity of merge
entity handshake_merge_0 is
  port (
    clk : in std_logic;
    rst : in std_logic;
    -- input channels
    ins       : in  data_array(2 - 1 downto 0)(3 - 1 downto 0);
    ins_valid : in  std_logic_vector(2 - 1 downto 0);
    ins_ready : out std_logic_vector(2 - 1 downto 0);
    -- output channel
    outs       : out std_logic_vector(3 - 1 downto 0);
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;

-- Architecture of merge
architecture arch of handshake_merge_0 is
begin
  process (ins_valid, ins, outs_ready)
    variable tmp_data_out  : unsigned(3 - 1 downto 0);
    variable tmp_valid_out : std_logic;
    variable tmp_ready_out : std_logic_vector(2 - 1 downto 0);
  begin
    tmp_data_out  := unsigned(ins(0));
    tmp_valid_out := '0';
    tmp_ready_out := (others => '0');

    for I in 0 to (2 - 1) loop
      if (ins_valid(I) = '1') then
        tmp_data_out  := unsigned(ins(I));
        tmp_valid_out := '1';
        tmp_ready_out(i) := outs_ready;
        exit;
      end if;
    end loop;

    outs <= std_logic_vector(resize(tmp_data_out, 3));
    outs_valid  <= tmp_valid_out;
    ins_ready <= tmp_ready_out;
  end process;

end architecture;

