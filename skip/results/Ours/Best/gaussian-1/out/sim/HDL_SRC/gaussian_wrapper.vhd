library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity gaussian_wrapper is
  port (
    c_din0 : in std_logic_vector(31 downto 0);
    c_din1 : in std_logic_vector(31 downto 0);
    a_din0 : in std_logic_vector(31 downto 0);
    a_din1 : in std_logic_vector(31 downto 0);
    c_start_valid : in std_logic;
    a_start_valid : in std_logic;
    start_valid : in std_logic;
    clk : in std_logic;
    rst : in std_logic;
    out0_ready : in std_logic;
    c_end_ready : in std_logic;
    a_end_ready : in std_logic;
    end_ready : in std_logic;
    c_start_ready : out std_logic;
    a_start_ready : out std_logic;
    start_ready : out std_logic;
    out0 : out std_logic_vector(31 downto 0);
    out0_valid : out std_logic;
    c_end_valid : out std_logic;
    a_end_valid : out std_logic;
    end_valid : out std_logic;
    c_ce0 : out std_logic;
    c_we0 : out std_logic;
    c_address0 : out std_logic_vector(4 downto 0);
    c_dout0 : out std_logic_vector(31 downto 0);
    c_ce1 : out std_logic;
    c_we1 : out std_logic;
    c_address1 : out std_logic_vector(4 downto 0);
    c_dout1 : out std_logic_vector(31 downto 0);
    a_ce0 : out std_logic;
    a_we0 : out std_logic;
    a_address0 : out std_logic_vector(8 downto 0);
    a_dout0 : out std_logic_vector(31 downto 0);
    a_ce1 : out std_logic;
    a_we1 : out std_logic;
    a_address1 : out std_logic_vector(8 downto 0);
    a_dout1 : out std_logic_vector(31 downto 0)
  );
end entity;

architecture behavioral of gaussian_wrapper is

  signal mem_to_bram_converter_c_ce0 : std_logic;
  signal mem_to_bram_converter_c_we0 : std_logic;
  signal mem_to_bram_converter_c_address0 : std_logic_vector(4 downto 0);
  signal mem_to_bram_converter_c_dout0 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_c_ce1 : std_logic;
  signal mem_to_bram_converter_c_we1 : std_logic;
  signal mem_to_bram_converter_c_address1 : std_logic_vector(4 downto 0);
  signal mem_to_bram_converter_c_dout1 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_c_loadData : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_a_ce0 : std_logic;
  signal mem_to_bram_converter_a_we0 : std_logic;
  signal mem_to_bram_converter_a_address0 : std_logic_vector(8 downto 0);
  signal mem_to_bram_converter_a_dout0 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_a_ce1 : std_logic;
  signal mem_to_bram_converter_a_we1 : std_logic;
  signal mem_to_bram_converter_a_address1 : std_logic_vector(8 downto 0);
  signal mem_to_bram_converter_a_dout1 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_a_loadData : std_logic_vector(31 downto 0);
  signal gaussian_wrapped_out0 : std_logic_vector(31 downto 0);
  signal gaussian_wrapped_out0_valid : std_logic;
  signal gaussian_wrapped_out0_ready : std_logic;
  signal gaussian_wrapped_c_end_valid : std_logic;
  signal gaussian_wrapped_c_end_ready : std_logic;
  signal gaussian_wrapped_a_end_valid : std_logic;
  signal gaussian_wrapped_a_end_ready : std_logic;
  signal gaussian_wrapped_end_valid : std_logic;
  signal gaussian_wrapped_end_ready : std_logic;
  signal gaussian_wrapped_c_loadEn : std_logic;
  signal gaussian_wrapped_c_loadAddr : std_logic_vector(4 downto 0);
  signal gaussian_wrapped_c_storeEn : std_logic;
  signal gaussian_wrapped_c_storeAddr : std_logic_vector(4 downto 0);
  signal gaussian_wrapped_c_storeData : std_logic_vector(31 downto 0);
  signal gaussian_wrapped_a_loadEn : std_logic;
  signal gaussian_wrapped_a_loadAddr : std_logic_vector(8 downto 0);
  signal gaussian_wrapped_a_storeEn : std_logic;
  signal gaussian_wrapped_a_storeAddr : std_logic_vector(8 downto 0);
  signal gaussian_wrapped_a_storeData : std_logic_vector(31 downto 0);

begin

  out0 <= gaussian_wrapped_out0;
  out0_valid <= gaussian_wrapped_out0_valid;
  gaussian_wrapped_out0_ready <= out0_ready;
  c_end_valid <= gaussian_wrapped_c_end_valid;
  gaussian_wrapped_c_end_ready <= c_end_ready;
  a_end_valid <= gaussian_wrapped_a_end_valid;
  gaussian_wrapped_a_end_ready <= a_end_ready;
  end_valid <= gaussian_wrapped_end_valid;
  gaussian_wrapped_end_ready <= end_ready;
  c_ce0 <= mem_to_bram_converter_c_ce0;
  c_we0 <= mem_to_bram_converter_c_we0;
  c_address0 <= mem_to_bram_converter_c_address0;
  c_dout0 <= mem_to_bram_converter_c_dout0;
  c_ce1 <= mem_to_bram_converter_c_ce1;
  c_we1 <= mem_to_bram_converter_c_we1;
  c_address1 <= mem_to_bram_converter_c_address1;
  c_dout1 <= mem_to_bram_converter_c_dout1;
  a_ce0 <= mem_to_bram_converter_a_ce0;
  a_we0 <= mem_to_bram_converter_a_we0;
  a_address0 <= mem_to_bram_converter_a_address0;
  a_dout0 <= mem_to_bram_converter_a_dout0;
  a_ce1 <= mem_to_bram_converter_a_ce1;
  a_we1 <= mem_to_bram_converter_a_we1;
  a_address1 <= mem_to_bram_converter_a_address1;
  a_dout1 <= mem_to_bram_converter_a_dout1;

  mem_to_bram_converter_c : entity work.mem_to_bram(arch) generic map(32, 5)
    port map(
      loadEn => gaussian_wrapped_c_loadEn,
      loadAddr => gaussian_wrapped_c_loadAddr,
      storeEn => gaussian_wrapped_c_storeEn,
      storeAddr => gaussian_wrapped_c_storeAddr,
      storeData => gaussian_wrapped_c_storeData,
      din0 => c_din0,
      din1 => c_din1,
      ce0 => mem_to_bram_converter_c_ce0,
      we0 => mem_to_bram_converter_c_we0,
      address0 => mem_to_bram_converter_c_address0,
      dout0 => mem_to_bram_converter_c_dout0,
      ce1 => mem_to_bram_converter_c_ce1,
      we1 => mem_to_bram_converter_c_we1,
      address1 => mem_to_bram_converter_c_address1,
      dout1 => mem_to_bram_converter_c_dout1,
      loadData => mem_to_bram_converter_c_loadData
    );

  mem_to_bram_converter_a : entity work.mem_to_bram(arch) generic map(32, 9)
    port map(
      loadEn => gaussian_wrapped_a_loadEn,
      loadAddr => gaussian_wrapped_a_loadAddr,
      storeEn => gaussian_wrapped_a_storeEn,
      storeAddr => gaussian_wrapped_a_storeAddr,
      storeData => gaussian_wrapped_a_storeData,
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

  gaussian_wrapped : entity work.gaussian(behavioral)
    port map(
      c_loadData => mem_to_bram_converter_c_loadData,
      a_loadData => mem_to_bram_converter_a_loadData,
      c_start_valid => c_start_valid,
      c_start_ready => c_start_ready,
      a_start_valid => a_start_valid,
      a_start_ready => a_start_ready,
      start_valid => start_valid,
      start_ready => start_ready,
      clk => clk,
      rst => rst,
      out0 => gaussian_wrapped_out0,
      out0_valid => gaussian_wrapped_out0_valid,
      out0_ready => gaussian_wrapped_out0_ready,
      c_end_valid => gaussian_wrapped_c_end_valid,
      c_end_ready => gaussian_wrapped_c_end_ready,
      a_end_valid => gaussian_wrapped_a_end_valid,
      a_end_ready => gaussian_wrapped_a_end_ready,
      end_valid => gaussian_wrapped_end_valid,
      end_ready => gaussian_wrapped_end_ready,
      c_loadEn => gaussian_wrapped_c_loadEn,
      c_loadAddr => gaussian_wrapped_c_loadAddr,
      c_storeEn => gaussian_wrapped_c_storeEn,
      c_storeAddr => gaussian_wrapped_c_storeAddr,
      c_storeData => gaussian_wrapped_c_storeData,
      a_loadEn => gaussian_wrapped_a_loadEn,
      a_loadAddr => gaussian_wrapped_a_loadAddr,
      a_storeEn => gaussian_wrapped_a_storeEn,
      a_storeAddr => gaussian_wrapped_a_storeAddr,
      a_storeData => gaussian_wrapped_a_storeData
    );

end architecture;
