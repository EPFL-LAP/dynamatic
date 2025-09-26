-- handshake_extsi_2 : extsi({'input_bitwidth': 8, 'output_bitwidth': 9, 'extra_signals': {'spec': 1}})


library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of extsi
entity handshake_extsi_2_inner is
  port (
    clk : in std_logic;
    rst : in std_logic;
    -- input channel
    ins       : in  std_logic_vector(8 - 1 downto 0);
    ins_valid : in  std_logic;
    ins_ready : out std_logic;
    -- output channel
    outs       : out std_logic_vector(9 - 1 downto 0);
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;

-- Architecture of extsi
architecture arch of handshake_extsi_2_inner is
begin
  outs(9 - 1 downto 8) <= (9 - 8 - 1 downto 0 => ins(8 - 1));
  outs(8 - 1 downto 0)            <= ins;
  outs_valid                                <= ins_valid;
  ins_ready                                 <= outs_ready;
end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

-- Entity of signal manager
entity handshake_extsi_2 is
  port(
    clk : in std_logic;
    rst : in std_logic;
    ins : in std_logic_vector(8 - 1 downto 0);
    ins_valid : in std_logic;
    ins_ready : out std_logic;
    ins_spec : in std_logic_vector(1 - 1 downto 0);
    outs : out std_logic_vector(9 - 1 downto 0);
    outs_valid : out std_logic;
    outs_ready : in std_logic;
    outs_spec : out std_logic_vector(1 - 1 downto 0)
  );
end entity;

-- Architecture of signal manager (default)
architecture arch of handshake_extsi_2 is
begin
  -- Forward extra signals to output channels
  outs_spec <= ins_spec;

  inner : entity work.handshake_extsi_2_inner(arch)
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

