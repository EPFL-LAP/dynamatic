-- handshake_merge_0 : merge({'size': 2, 'port_types': {'ins_0': '!handshake.channel<i1>', 'ins_1': '!handshake.channel<i1>', 'outs': '!handshake.channel<i1>'}, 'bitwidth': 1, 'extra_signals': {}})


library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

-- Entity of merge_notehb
entity handshake_merge_0_inner is
  port (
    clk, rst : in std_logic;
    -- input channels
    ins       : in  data_array(2 - 1 downto 0)(1 - 1 downto 0);
    ins_valid : in  std_logic_vector(2 - 1 downto 0);
    ins_ready : out std_logic_vector(2 - 1 downto 0);
    -- output channel
    outs       : out std_logic_vector(1 - 1 downto 0);
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;

-- Architecture of merge_notehb
architecture arch of handshake_merge_0_inner is
begin
  process (ins_valid, ins, outs_ready)
    variable tmp_data_out  : unsigned(1 - 1 downto 0);
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

    outs <= std_logic_vector(resize(tmp_data_out, 1));
    outs_valid  <= tmp_valid_out;
    ins_ready <= tmp_ready_out;
  end process;

end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of tehb_dataless
entity handshake_merge_0_tehb_dataless is
  port (
    clk, rst : in std_logic;
    -- input channel
    ins_valid : in  std_logic;
    ins_ready : out std_logic;
    -- output channel
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;

-- Architecture of tehb_dataless
architecture arch of handshake_merge_0_tehb_dataless is
  signal fullReg, outputValid : std_logic;
begin
  outputValid <= ins_valid or fullReg;

  process (clk) is
  begin
    if (rising_edge(clk)) then
      if (rst = '1') then
        fullReg <= '0';
      else
        fullReg <= outputValid and not outs_ready;
      end if;
    end if;
  end process;

  ins_ready  <= not fullReg;
  outs_valid <= outputValid;
end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of tehb
entity handshake_merge_0_tehb is
  port (
    clk, rst : in std_logic;
    -- input channel
    ins       : in  std_logic_vector(1 - 1 downto 0);
    ins_valid : in  std_logic;
    ins_ready : out std_logic;
    -- output channel
    outs       : out std_logic_vector(1 - 1 downto 0);
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;

-- Architecture of tehb
architecture arch of handshake_merge_0_tehb is
  signal regEnable, regNotFull : std_logic;
  signal dataReg               : std_logic_vector(1 - 1 downto 0);
begin
  regEnable <= regNotFull and ins_valid and not outs_ready;

  control : entity work.handshake_merge_0_tehb_dataless
    port map(
      clk        => clk,
      rst        => rst,
      ins_valid  => ins_valid,
      ins_ready  => regNotFull,
      outs_valid => outs_valid,
      outs_ready => outs_ready
    );

  process (clk) is
  begin
    if (rising_edge(clk)) then
      if (rst = '1') then
        dataReg <= (others => '0');
      elsif (regEnable) then
        dataReg <= ins;
      end if;
    end if;
  end process;

  process (regNotFull, dataReg, ins) is
  begin
    if (regNotFull) then
      outs <= ins;
    else
      outs <= dataReg;
    end if;
  end process;

  ins_ready <= regNotFull;

end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

-- Entity of merge
entity handshake_merge_0 is
  port (
    clk, rst : in std_logic;
    -- input channels
    ins       : in  data_array(2 - 1 downto 0)(1 - 1 downto 0);
    ins_valid : in  std_logic_vector(2 - 1 downto 0);
    ins_ready : out std_logic_vector(2 - 1 downto 0);
    -- output channel
    outs       : out std_logic_vector(1 - 1 downto 0);
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;

-- Architecture of merge
architecture arch of handshake_merge_0 is
  signal tehb_data_in : std_logic_vector(1 - 1 downto 0);
  signal tehb_pvalid  : std_logic;
  signal tehb_ready   : std_logic;
begin

  merge_ins : entity work.handshake_merge_0_inner(arch)
    port map(
      clk        => clk,
      rst        => rst,
      ins        => ins,
      ins_valid  => ins_valid,
      outs_ready => tehb_ready,
      ins_ready  => ins_ready,
      outs       => tehb_data_in,
      outs_valid => tehb_pvalid
    );

  tehb : entity work.handshake_merge_0_tehb(arch)
    port map(
      clk        => clk,
      rst        => rst,
      ins_valid  => tehb_pvalid,
      outs_ready => outs_ready,
      outs_valid => outs_valid,
      ins_ready  => tehb_ready,
      ins        => tehb_data_in,
      outs       => outs
    );
end architecture;

