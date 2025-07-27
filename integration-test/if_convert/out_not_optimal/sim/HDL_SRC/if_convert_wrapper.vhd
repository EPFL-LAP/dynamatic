library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity if_convert_wrapper is
  port (
    a_din0 : in std_logic_vector(31 downto 0);
    a_din1 : in std_logic_vector(31 downto 0);
    b_din0 : in std_logic_vector(31 downto 0);
    b_din1 : in std_logic_vector(31 downto 0);
    a_start_valid : in std_logic;
    b_start_valid : in std_logic;
    start_valid : in std_logic;
    clk : in std_logic;
    rst : in std_logic;
    a_end_ready : in std_logic;
    b_end_ready : in std_logic;
    end_ready : in std_logic;
    a_start_ready : out std_logic;
    b_start_ready : out std_logic;
    start_ready : out std_logic;
    a_end_valid : out std_logic;
    b_end_valid : out std_logic;
    end_valid : out std_logic;
    a_ce0 : out std_logic;
    a_we0 : out std_logic;
    a_address0 : out std_logic_vector(7 downto 0);
    a_dout0 : out std_logic_vector(31 downto 0);
    a_ce1 : out std_logic;
    a_we1 : out std_logic;
    a_address1 : out std_logic_vector(7 downto 0);
    a_dout1 : out std_logic_vector(31 downto 0);
    b_ce0 : out std_logic;
    b_we0 : out std_logic;
    b_address0 : out std_logic_vector(7 downto 0);
    b_dout0 : out std_logic_vector(31 downto 0);
    b_ce1 : out std_logic;
    b_we1 : out std_logic;
    b_address1 : out std_logic_vector(7 downto 0);
    b_dout1 : out std_logic_vector(31 downto 0)
  );
end entity;

architecture behavioral of if_convert_wrapper is

  signal mem_to_bram_converter_b_ce0 : std_logic;
  signal mem_to_bram_converter_b_we0 : std_logic;
  signal mem_to_bram_converter_b_address0 : std_logic_vector(7 downto 0);
  signal mem_to_bram_converter_b_dout0 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_b_ce1 : std_logic;
  signal mem_to_bram_converter_b_we1 : std_logic;
  signal mem_to_bram_converter_b_address1 : std_logic_vector(7 downto 0);
  signal mem_to_bram_converter_b_dout1 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_b_loadData : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_a_ce0 : std_logic;
  signal mem_to_bram_converter_a_we0 : std_logic;
  signal mem_to_bram_converter_a_address0 : std_logic_vector(7 downto 0);
  signal mem_to_bram_converter_a_dout0 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_a_ce1 : std_logic;
  signal mem_to_bram_converter_a_we1 : std_logic;
  signal mem_to_bram_converter_a_address1 : std_logic_vector(7 downto 0);
  signal mem_to_bram_converter_a_dout1 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_a_loadData : std_logic_vector(31 downto 0);
  signal if_convert_wrapped_a_end_valid : std_logic;
  signal if_convert_wrapped_a_end_ready : std_logic;
  signal if_convert_wrapped_b_end_valid : std_logic;
  signal if_convert_wrapped_b_end_ready : std_logic;
  signal if_convert_wrapped_end_valid : std_logic;
  signal if_convert_wrapped_end_ready : std_logic;
  signal if_convert_wrapped_a_loadEn : std_logic;
  signal if_convert_wrapped_a_loadAddr : std_logic_vector(7 downto 0);
  signal if_convert_wrapped_a_storeEn : std_logic;
  signal if_convert_wrapped_a_storeAddr : std_logic_vector(7 downto 0);
  signal if_convert_wrapped_a_storeData : std_logic_vector(31 downto 0);
  signal if_convert_wrapped_b_loadEn : std_logic;
  signal if_convert_wrapped_b_loadAddr : std_logic_vector(7 downto 0);
  signal if_convert_wrapped_b_storeEn : std_logic;
  signal if_convert_wrapped_b_storeAddr : std_logic_vector(7 downto 0);
  signal if_convert_wrapped_b_storeData : std_logic_vector(31 downto 0);

begin

  a_end_valid <= if_convert_wrapped_a_end_valid;
  if_convert_wrapped_a_end_ready <= a_end_ready;
  b_end_valid <= if_convert_wrapped_b_end_valid;
  if_convert_wrapped_b_end_ready <= b_end_ready;
  end_valid <= if_convert_wrapped_end_valid;
  if_convert_wrapped_end_ready <= end_ready;
  a_ce0 <= mem_to_bram_converter_a_ce0;
  a_we0 <= mem_to_bram_converter_a_we0;
  a_address0 <= mem_to_bram_converter_a_address0;
  a_dout0 <= mem_to_bram_converter_a_dout0;
  a_ce1 <= mem_to_bram_converter_a_ce1;
  a_we1 <= mem_to_bram_converter_a_we1;
  a_address1 <= mem_to_bram_converter_a_address1;
  a_dout1 <= mem_to_bram_converter_a_dout1;
  b_ce0 <= mem_to_bram_converter_b_ce0;
  b_we0 <= mem_to_bram_converter_b_we0;
  b_address0 <= mem_to_bram_converter_b_address0;
  b_dout0 <= mem_to_bram_converter_b_dout0;
  b_ce1 <= mem_to_bram_converter_b_ce1;
  b_we1 <= mem_to_bram_converter_b_we1;
  b_address1 <= mem_to_bram_converter_b_address1;
  b_dout1 <= mem_to_bram_converter_b_dout1;

  mem_to_bram_converter_b : entity work.mem_to_bram_32_8(arch)
    port map(
      loadEn => if_convert_wrapped_b_loadEn,
      loadAddr => if_convert_wrapped_b_loadAddr,
      storeEn => if_convert_wrapped_b_storeEn,
      storeAddr => if_convert_wrapped_b_storeAddr,
      storeData => if_convert_wrapped_b_storeData,
      din0 => b_din0,
      din1 => b_din1,
      ce0 => mem_to_bram_converter_b_ce0,
      we0 => mem_to_bram_converter_b_we0,
      address0 => mem_to_bram_converter_b_address0,
      dout0 => mem_to_bram_converter_b_dout0,
      ce1 => mem_to_bram_converter_b_ce1,
      we1 => mem_to_bram_converter_b_we1,
      address1 => mem_to_bram_converter_b_address1,
      dout1 => mem_to_bram_converter_b_dout1,
      loadData => mem_to_bram_converter_b_loadData
    );

  mem_to_bram_converter_a : entity work.mem_to_bram_32_8(arch)
    port map(
      loadEn => if_convert_wrapped_a_loadEn,
      loadAddr => if_convert_wrapped_a_loadAddr,
      storeEn => if_convert_wrapped_a_storeEn,
      storeAddr => if_convert_wrapped_a_storeAddr,
      storeData => if_convert_wrapped_a_storeData,
      din0 => a_din0,
      din1 => a_din1,
      ce0 => mem_to_bram_converter_a_ce0,
      we0 => mem_to_bram_converter_a_we0,
      address0 => mem_to_bram_converter_a_address0,
      dout0 => mem_to_bram_converter_a_dout0,
      ce1 => mem_to_bram_converter_a_ce1,
      we1 => mem_to_bram_converter_a_we1,
      address1 => mem_to_bram_converter_a_address1,
      dout1 => mem_to_bram_converter_a_dout1,
      loadData => mem_to_bram_converter_a_loadData
    );

  if_convert_wrapped : entity work.if_convert(behavioral)
    port map(
      a_loadData => mem_to_bram_converter_a_loadData,
      b_loadData => mem_to_bram_converter_b_loadData,
      a_start_valid => a_start_valid,
      a_start_ready => a_start_ready,
      b_start_valid => b_start_valid,
      b_start_ready => b_start_ready,
      start_valid => start_valid,
      start_ready => start_ready,
      clk => clk,
      rst => rst,
      a_end_valid => if_convert_wrapped_a_end_valid,
      a_end_ready => if_convert_wrapped_a_end_ready,
      b_end_valid => if_convert_wrapped_b_end_valid,
      b_end_ready => if_convert_wrapped_b_end_ready,
      end_valid => if_convert_wrapped_end_valid,
      end_ready => if_convert_wrapped_end_ready,
      a_loadEn => if_convert_wrapped_a_loadEn,
      a_loadAddr => if_convert_wrapped_a_loadAddr,
      a_storeEn => if_convert_wrapped_a_storeEn,
      a_storeAddr => if_convert_wrapped_a_storeAddr,
      a_storeData => if_convert_wrapped_a_storeData,
      b_loadEn => if_convert_wrapped_b_loadEn,
      b_loadAddr => if_convert_wrapped_b_loadAddr,
      b_storeEn => if_convert_wrapped_b_storeEn,
      b_storeAddr => if_convert_wrapped_b_storeAddr,
      b_storeData => if_convert_wrapped_b_storeData
    );

end architecture;
