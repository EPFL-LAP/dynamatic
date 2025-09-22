library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity subdiag_fast_wrapper is
  port (
    d1_din0 : in std_logic_vector(31 downto 0);
    d1_din1 : in std_logic_vector(31 downto 0);
    d2_din0 : in std_logic_vector(31 downto 0);
    d2_din1 : in std_logic_vector(31 downto 0);
    e_din0 : in std_logic_vector(31 downto 0);
    e_din1 : in std_logic_vector(31 downto 0);
    d1_start_valid : in std_logic;
    d2_start_valid : in std_logic;
    e_start_valid : in std_logic;
    start_valid : in std_logic;
    clk : in std_logic;
    rst : in std_logic;
    out0_ready : in std_logic;
    d1_end_ready : in std_logic;
    d2_end_ready : in std_logic;
    e_end_ready : in std_logic;
    end_ready : in std_logic;
    d1_start_ready : out std_logic;
    d2_start_ready : out std_logic;
    e_start_ready : out std_logic;
    start_ready : out std_logic;
    out0 : out std_logic_vector(31 downto 0);
    out0_valid : out std_logic;
    d1_end_valid : out std_logic;
    d2_end_valid : out std_logic;
    e_end_valid : out std_logic;
    end_valid : out std_logic;
    d1_ce0 : out std_logic;
    d1_we0 : out std_logic;
    d1_address0 : out std_logic_vector(9 downto 0);
    d1_dout0 : out std_logic_vector(31 downto 0);
    d1_ce1 : out std_logic;
    d1_we1 : out std_logic;
    d1_address1 : out std_logic_vector(9 downto 0);
    d1_dout1 : out std_logic_vector(31 downto 0);
    d2_ce0 : out std_logic;
    d2_we0 : out std_logic;
    d2_address0 : out std_logic_vector(9 downto 0);
    d2_dout0 : out std_logic_vector(31 downto 0);
    d2_ce1 : out std_logic;
    d2_we1 : out std_logic;
    d2_address1 : out std_logic_vector(9 downto 0);
    d2_dout1 : out std_logic_vector(31 downto 0);
    e_ce0 : out std_logic;
    e_we0 : out std_logic;
    e_address0 : out std_logic_vector(9 downto 0);
    e_dout0 : out std_logic_vector(31 downto 0);
    e_ce1 : out std_logic;
    e_we1 : out std_logic;
    e_address1 : out std_logic_vector(9 downto 0);
    e_dout1 : out std_logic_vector(31 downto 0)
  );
end entity;

architecture behavioral of subdiag_fast_wrapper is

  signal mem_to_bram_converter_d2_ce0 : std_logic;
  signal mem_to_bram_converter_d2_we0 : std_logic;
  signal mem_to_bram_converter_d2_address0 : std_logic_vector(9 downto 0);
  signal mem_to_bram_converter_d2_dout0 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_d2_ce1 : std_logic;
  signal mem_to_bram_converter_d2_we1 : std_logic;
  signal mem_to_bram_converter_d2_address1 : std_logic_vector(9 downto 0);
  signal mem_to_bram_converter_d2_dout1 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_d2_loadData : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_e_ce0 : std_logic;
  signal mem_to_bram_converter_e_we0 : std_logic;
  signal mem_to_bram_converter_e_address0 : std_logic_vector(9 downto 0);
  signal mem_to_bram_converter_e_dout0 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_e_ce1 : std_logic;
  signal mem_to_bram_converter_e_we1 : std_logic;
  signal mem_to_bram_converter_e_address1 : std_logic_vector(9 downto 0);
  signal mem_to_bram_converter_e_dout1 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_e_loadData : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_d1_ce0 : std_logic;
  signal mem_to_bram_converter_d1_we0 : std_logic;
  signal mem_to_bram_converter_d1_address0 : std_logic_vector(9 downto 0);
  signal mem_to_bram_converter_d1_dout0 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_d1_ce1 : std_logic;
  signal mem_to_bram_converter_d1_we1 : std_logic;
  signal mem_to_bram_converter_d1_address1 : std_logic_vector(9 downto 0);
  signal mem_to_bram_converter_d1_dout1 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_d1_loadData : std_logic_vector(31 downto 0);
  signal subdiag_fast_wrapped_out0 : std_logic_vector(31 downto 0);
  signal subdiag_fast_wrapped_out0_valid : std_logic;
  signal subdiag_fast_wrapped_out0_ready : std_logic;
  signal subdiag_fast_wrapped_d1_end_valid : std_logic;
  signal subdiag_fast_wrapped_d1_end_ready : std_logic;
  signal subdiag_fast_wrapped_d2_end_valid : std_logic;
  signal subdiag_fast_wrapped_d2_end_ready : std_logic;
  signal subdiag_fast_wrapped_e_end_valid : std_logic;
  signal subdiag_fast_wrapped_e_end_ready : std_logic;
  signal subdiag_fast_wrapped_end_valid : std_logic;
  signal subdiag_fast_wrapped_end_ready : std_logic;
  signal subdiag_fast_wrapped_d1_loadEn : std_logic;
  signal subdiag_fast_wrapped_d1_loadAddr : std_logic_vector(9 downto 0);
  signal subdiag_fast_wrapped_d1_storeEn : std_logic;
  signal subdiag_fast_wrapped_d1_storeAddr : std_logic_vector(9 downto 0);
  signal subdiag_fast_wrapped_d1_storeData : std_logic_vector(31 downto 0);
  signal subdiag_fast_wrapped_d2_loadEn : std_logic;
  signal subdiag_fast_wrapped_d2_loadAddr : std_logic_vector(9 downto 0);
  signal subdiag_fast_wrapped_d2_storeEn : std_logic;
  signal subdiag_fast_wrapped_d2_storeAddr : std_logic_vector(9 downto 0);
  signal subdiag_fast_wrapped_d2_storeData : std_logic_vector(31 downto 0);
  signal subdiag_fast_wrapped_e_loadEn : std_logic;
  signal subdiag_fast_wrapped_e_loadAddr : std_logic_vector(9 downto 0);
  signal subdiag_fast_wrapped_e_storeEn : std_logic;
  signal subdiag_fast_wrapped_e_storeAddr : std_logic_vector(9 downto 0);
  signal subdiag_fast_wrapped_e_storeData : std_logic_vector(31 downto 0);

begin

  out0 <= subdiag_fast_wrapped_out0;
  out0_valid <= subdiag_fast_wrapped_out0_valid;
  subdiag_fast_wrapped_out0_ready <= out0_ready;
  d1_end_valid <= subdiag_fast_wrapped_d1_end_valid;
  subdiag_fast_wrapped_d1_end_ready <= d1_end_ready;
  d2_end_valid <= subdiag_fast_wrapped_d2_end_valid;
  subdiag_fast_wrapped_d2_end_ready <= d2_end_ready;
  e_end_valid <= subdiag_fast_wrapped_e_end_valid;
  subdiag_fast_wrapped_e_end_ready <= e_end_ready;
  end_valid <= subdiag_fast_wrapped_end_valid;
  subdiag_fast_wrapped_end_ready <= end_ready;
  d1_ce0 <= mem_to_bram_converter_d1_ce0;
  d1_we0 <= mem_to_bram_converter_d1_we0;
  d1_address0 <= mem_to_bram_converter_d1_address0;
  d1_dout0 <= mem_to_bram_converter_d1_dout0;
  d1_ce1 <= mem_to_bram_converter_d1_ce1;
  d1_we1 <= mem_to_bram_converter_d1_we1;
  d1_address1 <= mem_to_bram_converter_d1_address1;
  d1_dout1 <= mem_to_bram_converter_d1_dout1;
  d2_ce0 <= mem_to_bram_converter_d2_ce0;
  d2_we0 <= mem_to_bram_converter_d2_we0;
  d2_address0 <= mem_to_bram_converter_d2_address0;
  d2_dout0 <= mem_to_bram_converter_d2_dout0;
  d2_ce1 <= mem_to_bram_converter_d2_ce1;
  d2_we1 <= mem_to_bram_converter_d2_we1;
  d2_address1 <= mem_to_bram_converter_d2_address1;
  d2_dout1 <= mem_to_bram_converter_d2_dout1;
  e_ce0 <= mem_to_bram_converter_e_ce0;
  e_we0 <= mem_to_bram_converter_e_we0;
  e_address0 <= mem_to_bram_converter_e_address0;
  e_dout0 <= mem_to_bram_converter_e_dout0;
  e_ce1 <= mem_to_bram_converter_e_ce1;
  e_we1 <= mem_to_bram_converter_e_we1;
  e_address1 <= mem_to_bram_converter_e_address1;
  e_dout1 <= mem_to_bram_converter_e_dout1;

  mem_to_bram_converter_d2 : entity work.mem_to_bram_32_10(arch)
    port map(
      loadEn => subdiag_fast_wrapped_d2_loadEn,
      loadAddr => subdiag_fast_wrapped_d2_loadAddr,
      storeEn => subdiag_fast_wrapped_d2_storeEn,
      storeAddr => subdiag_fast_wrapped_d2_storeAddr,
      storeData => subdiag_fast_wrapped_d2_storeData,
      din0 => d2_din0,
      din1 => d2_din1,
      ce0 => mem_to_bram_converter_d2_ce0,
      we0 => mem_to_bram_converter_d2_we0,
      address0 => mem_to_bram_converter_d2_address0,
      dout0 => mem_to_bram_converter_d2_dout0,
      ce1 => mem_to_bram_converter_d2_ce1,
      we1 => mem_to_bram_converter_d2_we1,
      address1 => mem_to_bram_converter_d2_address1,
      dout1 => mem_to_bram_converter_d2_dout1,
      loadData => mem_to_bram_converter_d2_loadData
    );

  mem_to_bram_converter_e : entity work.mem_to_bram_32_10(arch)
    port map(
      loadEn => subdiag_fast_wrapped_e_loadEn,
      loadAddr => subdiag_fast_wrapped_e_loadAddr,
      storeEn => subdiag_fast_wrapped_e_storeEn,
      storeAddr => subdiag_fast_wrapped_e_storeAddr,
      storeData => subdiag_fast_wrapped_e_storeData,
      din0 => e_din0,
      din1 => e_din1,
      ce0 => mem_to_bram_converter_e_ce0,
      we0 => mem_to_bram_converter_e_we0,
      address0 => mem_to_bram_converter_e_address0,
      dout0 => mem_to_bram_converter_e_dout0,
      ce1 => mem_to_bram_converter_e_ce1,
      we1 => mem_to_bram_converter_e_we1,
      address1 => mem_to_bram_converter_e_address1,
      dout1 => mem_to_bram_converter_e_dout1,
      loadData => mem_to_bram_converter_e_loadData
    );

  mem_to_bram_converter_d1 : entity work.mem_to_bram_32_10(arch)
    port map(
      loadEn => subdiag_fast_wrapped_d1_loadEn,
      loadAddr => subdiag_fast_wrapped_d1_loadAddr,
      storeEn => subdiag_fast_wrapped_d1_storeEn,
      storeAddr => subdiag_fast_wrapped_d1_storeAddr,
      storeData => subdiag_fast_wrapped_d1_storeData,
      din0 => d1_din0,
      din1 => d1_din1,
      ce0 => mem_to_bram_converter_d1_ce0,
      we0 => mem_to_bram_converter_d1_we0,
      address0 => mem_to_bram_converter_d1_address0,
      dout0 => mem_to_bram_converter_d1_dout0,
      ce1 => mem_to_bram_converter_d1_ce1,
      we1 => mem_to_bram_converter_d1_we1,
      address1 => mem_to_bram_converter_d1_address1,
      dout1 => mem_to_bram_converter_d1_dout1,
      loadData => mem_to_bram_converter_d1_loadData
    );

  subdiag_fast_wrapped : entity work.subdiag_fast(behavioral)
    port map(
      d1_loadData => mem_to_bram_converter_d1_loadData,
      d2_loadData => mem_to_bram_converter_d2_loadData,
      e_loadData => mem_to_bram_converter_e_loadData,
      d1_start_valid => d1_start_valid,
      d1_start_ready => d1_start_ready,
      d2_start_valid => d2_start_valid,
      d2_start_ready => d2_start_ready,
      e_start_valid => e_start_valid,
      e_start_ready => e_start_ready,
      start_valid => start_valid,
      start_ready => start_ready,
      clk => clk,
      rst => rst,
      out0 => subdiag_fast_wrapped_out0,
      out0_valid => subdiag_fast_wrapped_out0_valid,
      out0_ready => subdiag_fast_wrapped_out0_ready,
      d1_end_valid => subdiag_fast_wrapped_d1_end_valid,
      d1_end_ready => subdiag_fast_wrapped_d1_end_ready,
      d2_end_valid => subdiag_fast_wrapped_d2_end_valid,
      d2_end_ready => subdiag_fast_wrapped_d2_end_ready,
      e_end_valid => subdiag_fast_wrapped_e_end_valid,
      e_end_ready => subdiag_fast_wrapped_e_end_ready,
      end_valid => subdiag_fast_wrapped_end_valid,
      end_ready => subdiag_fast_wrapped_end_ready,
      d1_loadEn => subdiag_fast_wrapped_d1_loadEn,
      d1_loadAddr => subdiag_fast_wrapped_d1_loadAddr,
      d1_storeEn => subdiag_fast_wrapped_d1_storeEn,
      d1_storeAddr => subdiag_fast_wrapped_d1_storeAddr,
      d1_storeData => subdiag_fast_wrapped_d1_storeData,
      d2_loadEn => subdiag_fast_wrapped_d2_loadEn,
      d2_loadAddr => subdiag_fast_wrapped_d2_loadAddr,
      d2_storeEn => subdiag_fast_wrapped_d2_storeEn,
      d2_storeAddr => subdiag_fast_wrapped_d2_storeAddr,
      d2_storeData => subdiag_fast_wrapped_d2_storeData,
      e_loadEn => subdiag_fast_wrapped_e_loadEn,
      e_loadAddr => subdiag_fast_wrapped_e_loadAddr,
      e_storeEn => subdiag_fast_wrapped_e_storeEn,
      e_storeAddr => subdiag_fast_wrapped_e_storeAddr,
      e_storeData => subdiag_fast_wrapped_e_storeData
    );

end architecture;
