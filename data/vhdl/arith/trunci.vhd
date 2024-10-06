library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity trunci is
  generic (
    INPUT_TYPE  : integer;
    OUTPUT_TYPE : integer
  );
  port (
    -- inputs
    clk        : in std_logic;
    rst        : in std_logic;
    ins        : in std_logic_vector(INPUT_TYPE - 1 downto 0);
    ins_valid  : in std_logic;
    outs_ready : in std_logic;
    -- outputs
    outs       : out std_logic_vector(OUTPUT_TYPE - 1 downto 0);
    outs_valid : out std_logic;
    ins_ready  : out std_logic
  );
end entity;

architecture arch of trunci is
begin
  outs       <= ins(OUTPUT_TYPE - 1 downto 0);
  outs_valid <= ins_valid;
  ins_ready  <= not ins_valid or (ins_valid and outs_ready);
end architecture;

entity trunci_with_tag is
  generic (
    INPUT_TYPE  : integer;
    OUTPUT_TYPE : integer
  );
  port (
    -- inputs
    clk        : in std_logic;
    rst        : in std_logic;
    ins        : in std_logic_vector(INPUT_TYPE - 1 downto 0);
    ins_valid  : in std_logic;
    ins_spec_tag : in std_logic;
    outs_ready : in std_logic;
    -- outputs
    outs       : out std_logic_vector(OUTPUT_TYPE - 1 downto 0);
    outs_valid : out std_logic;
    outs_spec_tag : out std_logic;
    ins_ready  : out std_logic
  );
end entity;

architecture arch of trunci_with_tag is
begin
  outs_spec_tag <= ins_spec_tag;
  trunci_inner : entity work.trunci(arch) generic map(INPUT_TYPE, OUTPUT_TYPE)
    port map(
      clk        => clk,
      rst        => rst,
      ins        => ins,
      ins_valid  => ins_valid,
      outs_ready => outs_ready,
      outs       => outs,
      outs_valid => outs_valid,
      ins_ready  => ins_ready
    );
end architecture
