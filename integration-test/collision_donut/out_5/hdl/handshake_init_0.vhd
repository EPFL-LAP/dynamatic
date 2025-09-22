-- handshake_init_0 : init({'init_token': 0})


library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of init
entity handshake_init_0 is
  port (
    clk, rst : in std_logic;
    -- input channel
    ins       : in  std_logic_vector(0 downto 0);
    ins_valid : in  std_logic;
    ins_ready : out std_logic;
    -- output channel
    outs       : out std_logic_vector(0 downto 0);
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;

-- Architecture of init
architecture arch of handshake_init_0 is
  signal init : std_logic;
begin
  outs <= "0" when init else ins;
  outs_valid <= '1' when init else ins_valid;
  ins_ready <= '0' when init else outs_ready;

  init_proc : process (clk)
  begin
    if rising_edge(clk) then
      if rst = '1' then
        init <= '1';
      else
        if outs_ready then
          init <= '0';
        end if;
      end if;
    end if;
  end process;
end architecture;

