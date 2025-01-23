library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity merge_notehb_dataless is
  generic (
    INPUTS : integer
  );
  port (
    clk, rst : in std_logic;
    -- input channels
    ins_valid : in  std_logic_vector(INPUTS - 1 downto 0);
    ins_ready : out std_logic_vector(INPUTS - 1 downto 0);
    -- output channel
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;

architecture arch of merge_notehb_dataless is
begin
  process (ins_valid, outs_ready)
    variable tmp_valid_out : std_logic;
    variable tmp_ready_out : std_logic_vector(INPUTS - 1 downto 0);
  begin
    tmp_valid_out := '0';
    tmp_ready_out := (others => '0');
    
    for i in 0 to (INPUTS - 1) loop
      if (ins_valid(i) = '1') then
        tmp_valid_out := '1';
        tmp_ready_out(i) := outs_ready;
        exit;
      end if;
    end loop;

    outs_valid <= tmp_valid_out;
    ins_ready <= tmp_ready_out;
  end process;

end architecture;