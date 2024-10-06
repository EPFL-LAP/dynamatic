library ieee;
use ieee.std_logic_1164.all;

entity join_handshake is
  generic (
    SIZE : integer
  );
  port (
    clk, rst : in std_logic;
    -- input channels
    ins_valid : in  std_logic_vector(SIZE - 1 downto 0);
    ins_ready : out std_logic_vector(SIZE - 1 downto 0);
    -- output channel
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end join_handshake;

architecture arch of join_handshake is
begin
  join_inner : entity work.join(arch) generic map(SIZE)
    port map(
      -- inputs
      ins_valid  => ins_valid,
      outs_ready => outs_ready,
      -- outputs
      outs_valid => outs_valid,
      ins_ready  => ins_ready
    );
end architecture;

library ieee;
use ieee.std_logic_1164.all;

entity join_handshake_with_tag is
  generic (
    SIZE : integer
  );
  port (
    clk, rst : in std_logic;
    -- input channels
    ins_valid : in  std_logic_vector(SIZE - 1 downto 0);
    ins_spec_tag : in  std_logic_vector(SIZE - 1 downto 0);
    ins_ready : out std_logic_vector(SIZE - 1 downto 0);
    -- output channel
    outs_valid : out std_logic;
    outs_spec_tag : out std_logic;
    outs_ready : in  std_logic
  );
end join_handshake_with_tag;

architecture arch of join_handshake_with_tag is
begin
  join_inner : entity work.join_with_tag(arch) generic map(SIZE)
    port map(
      -- inputs
      ins_valid  => ins_valid,
      ins_spec_tag => ins_spec_tag,
      outs_ready => outs_ready,
      -- outputs
      outs_valid => outs_valid,
      outs_spec_tag => outs_spec_tag,
      ins_ready  => ins_ready
    );
end architecture;
