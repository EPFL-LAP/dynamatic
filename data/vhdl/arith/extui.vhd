library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity extui is
  generic (
    INPUT_TYPE  : integer;
    OUTPUT_TYPE : integer
  );
  port (
    clk, rst : in std_logic;
    -- input channel
    ins       : in  std_logic_vector(INPUT_TYPE - 1 downto 0);
    ins_valid : in  std_logic;
    ins_ready : out std_logic;
    -- output channel
    outs       : out std_logic_vector(OUTPUT_TYPE - 1 downto 0);
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;

architecture arch of extui is
begin
  outs(OUTPUT_TYPE - 1 downto INPUT_TYPE) <= (OUTPUT_TYPE - INPUT_TYPE - 1 downto 0 => '0');
  outs(INPUT_TYPE - 1 downto 0)            <= ins;
  outs_valid                                <= ins_valid;
  ins_ready                                 <= outs_ready;
end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity extui_with_tag is
  generic (
    INPUT_TYPE  : integer;
    OUTPUT_TYPE : integer
  );
  port (
    clk, rst : in std_logic;
    -- input channel
    ins       : in  std_logic_vector(INPUT_TYPE - 1 downto 0);
    ins_valid : in  std_logic;
    ins_spec_tag : in std_logic;
    ins_ready : out std_logic;
    -- output channel
    outs       : out std_logic_vector(OUTPUT_TYPE - 1 downto 0);
    outs_valid : out std_logic;
    outs_spec_tag : out std_logic;
    outs_ready : in  std_logic
  );
end entity;

architecture arch of extui_with_tag is
begin
  outs_spec_tag <= ins_spec_tag;
  extui_inner : entity work.extui(arch)
    generic map(
      INPUT_TYPE,
      OUTPUT_TYPE
    )
    port map(
      clk => clk,
      rst => rst,
      ins => ins,
      ins_valid => ins_valid,
      ins_ready => ins_ready,
      outs => outs,
      outs_valid => outs_valid,
      outs_ready => outs_ready
    );
end architecture;
