library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity oehb is
  generic (
    DATA_TYPE : integer
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

architecture arch of oehb is
  signal regEn, inputReady : std_logic;
begin

  control : entity work.oehb_dataless
    port map(
      clk        => clk,
      rst        => rst,
      ins_valid  => ins_valid,
      ins_ready  => inputReady,
      outs_valid => outs_valid,
      outs_ready => outs_ready
    );

  process (clk) is
  begin
    if (rising_edge(clk)) then
      if (rst = '1') then
        outs <= (others => '0');
      elsif (regEn) then
        outs <= ins;
      end if;
    end if;
  end process;

  ins_ready <= inputReady;
  regEn     <= inputReady and ins_valid;
end architecture;

entity oehb_with_tag is
  generic (
    DATA_TYPE : integer
  );
  port (
    clk, rst : in std_logic;
    -- input channel
    ins       : in  std_logic_vector(DATA_TYPE - 1 downto 0);
    ins_valid : in  std_logic;
    ins_spec_tag : in std_logic;
    ins_ready : out std_logic;
    -- output channel
    outs       : out std_logic_vector(DATA_TYPE - 1 downto 0);
    outs_valid : out std_logic;
    outs_spec_tag : out std_logic;
    outs_ready : in  std_logic
  );
end entity;

architecture arch of oehb_with_tag is
  signal ins_inner : std_logic_vector(DATA_TYPE downto 0);
  signal outs_inner : std_logic_vector(DATA_TYPE downto 0);
begin
  ins_inner <= ins_spec_tag & ins;
  outs_spec_tag <= outs_inner(DATA_TYPE);
  outs <= outs_inner(DATA_TYPE - 1 downto 0);
  oehb_inner : entity work.oehb(arch) generic map(DATA_TYPE + 1)
    port map(
      clk        => clk,
      rst        => rst,
      ins        => ins_inner,
      ins_valid  => ins_valid,
      ins_ready  => ins_ready,
      outs       => outs_inner,
      outs_valid => outs_valid,
      outs_ready => outs_ready
    );
end architecture;
