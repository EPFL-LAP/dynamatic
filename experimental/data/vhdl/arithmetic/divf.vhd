library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity divf is
  generic (
    BITWIDTH : integer
  );
  port (
    clk         : in std_logic;
    rst         : in std_logic;
    pValidArray : in std_logic_vector(1 downto 0);
    nReady      : in std_logic;
    valid       : out std_logic;
    readyArray  : out std_logic_vector(1 downto 0);
    -- dataInArray 
    inToShift    : in std_logic_vector(BITWIDTH - 1 downto 0);
    inShiftBy    : in std_logic_vector(BITWIDTH - 1 downto 0);
    dataOutArray : out std_logic_vector(BITWIDTH - 1 downto 0));
end entity;

architecture arch of divf is

  -- Interface to Vivado component
  component array_RAM_fdiv_32ns_32ns_32_30_1 is
    generic (
      ID         : integer := 1;
      NUM_STAGE  : integer := 30;
      din0_WIDTH : integer := 32;
      din1_WIDTH : integer := 32;
      dout_WIDTH : integer := 32
    );
    port (
      clk   : in std_logic;
      reset : in std_logic;
      ce    : in std_logic;
      din0  : in std_logic_vector(din0_WIDTH - 1 downto 0);
      din1  : in std_logic_vector(din1_WIDTH - 1 downto 0);
      dout  : out std_logic_vector(dout_WIDTH - 1 downto 0)
    );
  end component;

  signal join_valid                         : std_logic;
  signal buff_valid, oehb_valid, oehb_ready : std_logic;
  signal oehb_dataOut, oehb_datain          : std_logic;

begin

  join_write_temp : entity work.join(arch) generic map(2)
    port map(
      pValidArray, --pValidArray
      oehb_ready,  --nready                    
      join_valid,  --valid          
      readyArray); --readyarray 
  buff : entity work.delay_buffer(arch)
    generic map(28)
    port map(
      clk,
      rst,
      join_valid,
      oehb_ready,
      buff_valid);
  oehb : entity work.OEHB(arch) generic map (1)
    port map(
      --inputspValidArray
      clk            => clk,
      rst            => rst,
      pValidArray(0) => buff_valid, -- real or speculatef condition (determined by merge1)
      nReady         => nReady,
      valid          => valid,
      --outputs
      readyArray(0) => oehb_ready,
      inToShift     => oehb_datain,
      dataOutArray  => oehb_dataOut
    );

  array_RAM_fdiv_32ns_32ns_32_30_1_U1 : component array_RAM_fdiv_32ns_32ns_32_30_1
    port map(
      clk   => clk,
      reset => rst,
      ce    => oehb_ready,
      din0  => inToShift,
      din1  => inShiftBy,
      dout  => dataOutArray);
  end architecture;
