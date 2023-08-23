library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;
-- #NAME# = oeq, ogt, oge, olt, ole, one, ord, ueq, ugt, uge, ult, ule, une, uno
-- #CONST# = 00001, 00010, 00011, 00100, 00101, 00110, 00111, 01000, 01001, 01010, 01011, 01100, 01101, 01110 
entity cmpf_#NAME# is
  generic (
    BITWIDTH : integer
  );
  port (
    clk, rst : in std_logic;
    -- dataInArray
    inToShift    : in std_logic_vector(BITWIDTH - 1 downto 0);
    inShiftBy    : in std_logic_vector(BITWIDTH - 1 downto 0);
    dataOutArray : out std_logic_vector(BITWIDTH - 1 downto 0);
    pValidArray  : in std_logic_vector(1 downto 0);
    nReady       : in std_logic;
    valid        : out std_logic;
    readyArray   : out std_logic_vector(1 downto 0));
end entity;

architecture arch of cmpf_#NAME# is

  --Interface to vivado component
  component array_RAM_cmpf_32cud is
    generic (
      ID         : integer := 1;
      NUM_STAGE  : integer := 2;
      din0_WIDTH : integer := 32;
      din1_WIDTH : integer := 32;
      dout_WIDTH : integer := 1
    );
    port (
      clk    : in std_logic;
      reset  : in std_logic;
      ce     : in std_logic;
      din0   : in std_logic_vector(din0_WIDTH - 1 downto 0);
      din1   : in std_logic_vector(din1_WIDTH - 1 downto 0);
      opcode : in std_logic_vector(4 downto 0);
      dout   : out std_logic_vector(dout_WIDTH - 1 downto 0)
    );
  end component;

  signal join_valid   : std_logic;
  constant alu_opcode : std_logic_vector(4 downto 0) := "#CONST#";

begin

  dataOutArray(BITWIDTH - 1 downto 1) <= (others => '0');

  array_RAM_cmpf_32ns_32ns_1_2_1_u1 : component array_RAM_cmpf_32cud
    generic map(
      ID         => 1,
      NUM_STAGE  => 2,
      din0_WIDTH => 32,
      din1_WIDTH => 32,
      dout_WIDTH => 1)
    port map(
      clk     => clk,
      reset   => rst,
      din0    => inToShift,
      din1    => inShiftBy,
      ce      => nReady,
      opcode  => alu_opcode,
      dout(0) => dataOutArray(0));

    join_write_temp : entity work.join(arch) generic map(2)
      port map(
        pValidArray, --pValidArray
        nReady,      --nready                    
        join_valid,  --valid          
        readyArray); --readyarray 

    buff : entity work.delay_buffer(arch)
      generic map(1)
      port map(
        clk,
        rst,
        join_valid,
        nReady,
        valid);
  end architecture;
