library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity cond_br is
  generic (
    DATA_TYPE : integer
  );
  port (
    clk, rst : in std_logic;
    -- data input channel
    data       : in  std_logic_vector(DATA_TYPE - 1 downto 0);
    data_valid : in  std_logic;
    data_ready : out std_logic;
    -- condition input channel
    condition       : in  std_logic_vector(0 downto 0);
    condition_valid : in  std_logic;
    condition_ready : out std_logic;
    -- true output channel
    trueOut       : out std_logic_vector(DATA_TYPE - 1 downto 0);
    trueOut_valid : out std_logic;
    trueOut_ready : in  std_logic;
    -- false output channel
    falseOut       : out std_logic_vector(DATA_TYPE - 1 downto 0);
    falseOut_valid : out std_logic;
    falseOut_ready : in  std_logic
  );
end entity;

architecture arch of cond_br is
begin
  control : entity work.cond_br_dataless
    port map(
      clk             => clk,
      rst             => rst,
      data_valid      => data_valid,
      data_ready      => data_ready,
      condition       => condition,
      condition_valid => condition_valid,
      condition_ready => condition_ready,
      trueOut_valid   => trueOut_valid,
      trueOut_ready   => trueOut_ready,
      falseOut_valid  => falseOut_valid,
      falseOut_ready  => falseOut_ready
    );

  trueOut  <= data;
  falseOut <= data;
end architecture;
