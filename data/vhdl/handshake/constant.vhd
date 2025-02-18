library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity ENTITY_NAME is
  generic (
    DATA_WIDTH : integer
  );
  port (
    clk, rst : in std_logic;
    -- input channel
    ctrl_valid : in  std_logic;
    ctrl_ready : out std_logic;
    -- output channel
    outs       : out std_logic_vector(DATA_WIDTH - 1 downto 0);
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;

architecture arch of ENTITY_NAME is
begin
  outs       <= "VALUE";
  outs_valid <= ctrl_valid;
  ctrl_ready <= outs_ready;
end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity ENTITY_NAME_with_tag is
  generic (
    DATA_WIDTH : integer
  );
  port (
    clk, rst : in std_logic;
    -- input channel
    ctrl_valid : in  std_logic;
    ctrl_spec_tag : in std_logic;
    ctrl_ready : out std_logic;
    -- output channel
    outs       : out std_logic_vector(DATA_WIDTH - 1 downto 0);
    outs_valid : out std_logic;
    outs_spec_tag : out std_logic;
    outs_ready : in  std_logic
  );
end entity;

architecture arch of ENTITY_NAME_with_tag is
begin
  outs_spec_tag <= ctrl_spec_tag;
  inner : entity work.ENTITY_NAME(arch)
    generic map(
      DATA_WIDTH
    )
    port map(
      clk => clk,
      rst => rst,
      ctrl_valid => ctrl_valid,
      ctrl_ready => ctrl_ready,
      outs => outs,
      outs_valid => outs_valid,
      outs_ready => outs_ready
    );
end architecture;
