-- handshake_constant_7 : constant({'value': '01100100', 'bitwidth': 8, 'extra_signals': {'spec': 1}})


library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of constant
entity handshake_constant_7_inner is
  port (
    clk : in std_logic;
    rst : in std_logic;
    -- input channel
    ctrl_valid : in  std_logic;
    ctrl_ready : out std_logic;
    -- output channel
    outs       : out std_logic_vector(8 - 1 downto 0);
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;

-- Architecture of constant
architecture arch of handshake_constant_7_inner is
begin
  outs       <= "01100100";
  outs_valid <= ctrl_valid;
  ctrl_ready <= outs_ready;
end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

-- Entity of signal manager
entity handshake_constant_7 is
  port(
    clk : in std_logic;
    rst : in std_logic;
    ctrl_valid : in std_logic;
    ctrl_ready : out std_logic;
    ctrl_spec : in std_logic_vector(1 - 1 downto 0);
    outs : out std_logic_vector(8 - 1 downto 0);
    outs_valid : out std_logic;
    outs_ready : in std_logic;
    outs_spec : out std_logic_vector(1 - 1 downto 0)
  );
end entity;

-- Architecture of signal manager (default)
architecture arch of handshake_constant_7 is
begin
  -- Forward extra signals to output channels
  outs_spec <= ctrl_spec;

  inner : entity work.handshake_constant_7_inner(arch)
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

