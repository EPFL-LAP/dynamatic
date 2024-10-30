library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity dvse is
  generic (
    DATA_TYPE   : integer;
    NUM_SLOTS   : integer
  );
  port (
    clk, rst : in std_logic;
    -- input channel
    ins       : in  std_logic_vector(DATA_TYPE - 1 downto 0);
    ins_valid : in  std_logic;
    ins_ready : out std_logic;
    -- output channel
    outs       : out std_logic_vector(DATA_TYPE - 1 downto 0);
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;

architecture arch of dvse is

  signal regEn, inputReady : std_logic;
  type DVSE_MEMORY is array (0 to NUM_SLOTS - 1) of std_logic_vector(DATA_TYPE - 1 downto 0);
  signal Memory  : DVSE_MEMORY;

begin

  control : entity work.dvse_dataless
    port map(
      clk        => clk,
      rst        => rst,
      ins_valid  => ins_valid,
      ins_ready  => inputReady,
      outs_valid => outs_valid,
      outs_ready => outs_ready
    );

  -- If not valid, the data in this slot is obsolete
  -- Hence, no need to reset
  -- If reset is necessary, then add the following lines:
  -- if (rst = '1') then
  --   for i in 0 to NUM_SLOTS - 1 loop
  --     Memory(i) <= (others => '0');
  --   end loop;
  process (clk) is
  begin
    if (rising_edge(clk)) then
      if (regEn) then
        for i in 1 to NUM_SLOTS - 1 loop
          Memory(i) <= Memory(i - 1);
        end loop;
        Memory(0) <= ins;
      end if;
    end if;
  end process;

  regEn <= inputReady;
  ins_ready <= inputReady;
  outs <= Memory(NUM_SLOTS - 1);

end architecture;