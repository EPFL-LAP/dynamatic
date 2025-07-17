library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity multiple_exit_wrapper is
  port (
    arr_din0 : in std_logic_vector(31 downto 0);
    arr_din1 : in std_logic_vector(31 downto 0);
    size : in std_logic_vector(31 downto 0);
    size_valid : in std_logic;
    arr_start_valid : in std_logic;
    start_valid : in std_logic;
    clk : in std_logic;
    rst : in std_logic;
    arr_end_ready : in std_logic;
    end_ready : in std_logic;
    size_ready : out std_logic;
    arr_start_ready : out std_logic;
    start_ready : out std_logic;
    arr_end_valid : out std_logic;
    end_valid : out std_logic;
    arr_ce0 : out std_logic;
    arr_we0 : out std_logic;
    arr_address0 : out std_logic_vector(3 downto 0);
    arr_dout0 : out std_logic_vector(31 downto 0);
    arr_ce1 : out std_logic;
    arr_we1 : out std_logic;
    arr_address1 : out std_logic_vector(3 downto 0);
    arr_dout1 : out std_logic_vector(31 downto 0)
  );
end entity;

architecture behavioral of multiple_exit_wrapper is

  signal mem_to_bram_converter_arr_ce0 : std_logic;
  signal mem_to_bram_converter_arr_we0 : std_logic;
  signal mem_to_bram_converter_arr_address0 : std_logic_vector(3 downto 0);
  signal mem_to_bram_converter_arr_dout0 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_arr_ce1 : std_logic;
  signal mem_to_bram_converter_arr_we1 : std_logic;
  signal mem_to_bram_converter_arr_address1 : std_logic_vector(3 downto 0);
  signal mem_to_bram_converter_arr_dout1 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_arr_loadData : std_logic_vector(31 downto 0);
  signal multiple_exit_wrapped_arr_end_valid : std_logic;
  signal multiple_exit_wrapped_arr_end_ready : std_logic;
  signal multiple_exit_wrapped_end_valid : std_logic;
  signal multiple_exit_wrapped_end_ready : std_logic;
  signal multiple_exit_wrapped_arr_loadEn : std_logic;
  signal multiple_exit_wrapped_arr_loadAddr : std_logic_vector(3 downto 0);
  signal multiple_exit_wrapped_arr_storeEn : std_logic;
  signal multiple_exit_wrapped_arr_storeAddr : std_logic_vector(3 downto 0);
  signal multiple_exit_wrapped_arr_storeData : std_logic_vector(31 downto 0);

begin

  arr_end_valid <= multiple_exit_wrapped_arr_end_valid;
  multiple_exit_wrapped_arr_end_ready <= arr_end_ready;
  end_valid <= multiple_exit_wrapped_end_valid;
  multiple_exit_wrapped_end_ready <= end_ready;
  arr_ce0 <= mem_to_bram_converter_arr_ce0;
  arr_we0 <= mem_to_bram_converter_arr_we0;
  arr_address0 <= mem_to_bram_converter_arr_address0;
  arr_dout0 <= mem_to_bram_converter_arr_dout0;
  arr_ce1 <= mem_to_bram_converter_arr_ce1;
  arr_we1 <= mem_to_bram_converter_arr_we1;
  arr_address1 <= mem_to_bram_converter_arr_address1;
  arr_dout1 <= mem_to_bram_converter_arr_dout1;

  mem_to_bram_converter_arr : entity work.mem_to_bram(arch) generic map(32, 4)
    port map(
      loadEn => multiple_exit_wrapped_arr_loadEn,
      loadAddr => multiple_exit_wrapped_arr_loadAddr,
      storeEn => multiple_exit_wrapped_arr_storeEn,
      storeAddr => multiple_exit_wrapped_arr_storeAddr,
      storeData => multiple_exit_wrapped_arr_storeData,
      din0 => arr_din0,
      din1 => arr_din1,
      ce0 => mem_to_bram_converter_arr_ce0,
      we0 => mem_to_bram_converter_arr_we0,
      address0 => mem_to_bram_converter_arr_address0,
      dout0 => mem_to_bram_converter_arr_dout0,
      ce1 => mem_to_bram_converter_arr_ce1,
      we1 => mem_to_bram_converter_arr_we1,
      address1 => mem_to_bram_converter_arr_address1,
      dout1 => mem_to_bram_converter_arr_dout1,
      loadData => mem_to_bram_converter_arr_loadData
    );

  multiple_exit_wrapped : entity work.multiple_exit(behavioral)
    port map(
      arr_loadData => mem_to_bram_converter_arr_loadData,
      size => size,
      size_valid => size_valid,
      size_ready => size_ready,
      arr_start_valid => arr_start_valid,
      arr_start_ready => arr_start_ready,
      start_valid => start_valid,
      start_ready => start_ready,
      clk => clk,
      rst => rst,
      arr_end_valid => multiple_exit_wrapped_arr_end_valid,
      arr_end_ready => multiple_exit_wrapped_arr_end_ready,
      end_valid => multiple_exit_wrapped_end_valid,
      end_ready => multiple_exit_wrapped_end_ready,
      arr_loadEn => multiple_exit_wrapped_arr_loadEn,
      arr_loadAddr => multiple_exit_wrapped_arr_loadAddr,
      arr_storeEn => multiple_exit_wrapped_arr_storeEn,
      arr_storeAddr => multiple_exit_wrapped_arr_storeAddr,
      arr_storeData => multiple_exit_wrapped_arr_storeData
    );

end architecture;
