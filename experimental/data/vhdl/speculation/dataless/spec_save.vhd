library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

entity spec_save_dataless_with_tag is
  port (
    clk, rst : in std_logic;
    -- input channel
    ins_valid : in  std_logic;
    ins_spec_tag : in std_logic; -- not used
    ctrl      : in std_logic_vector(0 downto 0);
    ctrl_valid : in std_logic;
    ctrl_spec_tag : in std_logic; -- not used
    outs_ready : in  std_logic;
    -- output channel
    outs_valid : out std_logic;
    outs_spec_tag : out std_logic;
    ins_ready : out std_logic;
    ctrl_ready : out std_logic
  );
end entity;

architecture arch of spec_save_dataless_with_tag is
  signal ins_inner : std_logic_vector(0 downto 0);
  signal outs_inner : std_logic_vector(0 downto 0);
begin
  ins_inner(0) <= '0';
  spec_save : entity work.spec_save_with_tag
    generic map(
      DATA_TYPE => 1
    )
    port map(
      clk => clk,
      rst => rst,
      ins => ins_inner,
      ins_valid => ins_valid,
      ins_spec_tag => ins_spec_tag,
      ctrl => ctrl,
      ctrl_valid => ctrl_valid,
      ctrl_spec_tag => ctrl_spec_tag,
      outs => outs_inner,
      outs_ready => outs_ready,
      outs_valid => outs_valid,
      outs_spec_tag => outs_spec_tag,
      ins_ready => ins_ready,
      ctrl_ready => ctrl_ready
    );
end architecture;
