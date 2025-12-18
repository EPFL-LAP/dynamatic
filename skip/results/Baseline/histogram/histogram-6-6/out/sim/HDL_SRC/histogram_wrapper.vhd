library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity histogram_wrapper is
  port (
    feature_din0 : in std_logic_vector(31 downto 0);
    feature_din1 : in std_logic_vector(31 downto 0);
    weight_din0 : in std_logic_vector(31 downto 0);
    weight_din1 : in std_logic_vector(31 downto 0);
    hist_din0 : in std_logic_vector(31 downto 0);
    hist_din1 : in std_logic_vector(31 downto 0);
    n : in std_logic_vector(31 downto 0);
    n_valid : in std_logic;
    feature_start_valid : in std_logic;
    weight_start_valid : in std_logic;
    hist_start_valid : in std_logic;
    start_valid : in std_logic;
    clk : in std_logic;
    rst : in std_logic;
    feature_end_ready : in std_logic;
    weight_end_ready : in std_logic;
    hist_end_ready : in std_logic;
    end_ready : in std_logic;
    n_ready : out std_logic;
    feature_start_ready : out std_logic;
    weight_start_ready : out std_logic;
    hist_start_ready : out std_logic;
    start_ready : out std_logic;
    feature_end_valid : out std_logic;
    weight_end_valid : out std_logic;
    hist_end_valid : out std_logic;
    end_valid : out std_logic;
    feature_ce0 : out std_logic;
    feature_we0 : out std_logic;
    feature_address0 : out std_logic_vector(9 downto 0);
    feature_dout0 : out std_logic_vector(31 downto 0);
    feature_ce1 : out std_logic;
    feature_we1 : out std_logic;
    feature_address1 : out std_logic_vector(9 downto 0);
    feature_dout1 : out std_logic_vector(31 downto 0);
    weight_ce0 : out std_logic;
    weight_we0 : out std_logic;
    weight_address0 : out std_logic_vector(9 downto 0);
    weight_dout0 : out std_logic_vector(31 downto 0);
    weight_ce1 : out std_logic;
    weight_we1 : out std_logic;
    weight_address1 : out std_logic_vector(9 downto 0);
    weight_dout1 : out std_logic_vector(31 downto 0);
    hist_ce0 : out std_logic;
    hist_we0 : out std_logic;
    hist_address0 : out std_logic_vector(9 downto 0);
    hist_dout0 : out std_logic_vector(31 downto 0);
    hist_ce1 : out std_logic;
    hist_we1 : out std_logic;
    hist_address1 : out std_logic_vector(9 downto 0);
    hist_dout1 : out std_logic_vector(31 downto 0)
  );
end entity;

architecture behavioral of histogram_wrapper is

  signal mem_to_bram_converter_weight_ce0 : std_logic;
  signal mem_to_bram_converter_weight_we0 : std_logic;
  signal mem_to_bram_converter_weight_address0 : std_logic_vector(9 downto 0);
  signal mem_to_bram_converter_weight_dout0 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_weight_ce1 : std_logic;
  signal mem_to_bram_converter_weight_we1 : std_logic;
  signal mem_to_bram_converter_weight_address1 : std_logic_vector(9 downto 0);
  signal mem_to_bram_converter_weight_dout1 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_weight_loadData : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_hist_ce0 : std_logic;
  signal mem_to_bram_converter_hist_we0 : std_logic;
  signal mem_to_bram_converter_hist_address0 : std_logic_vector(9 downto 0);
  signal mem_to_bram_converter_hist_dout0 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_hist_ce1 : std_logic;
  signal mem_to_bram_converter_hist_we1 : std_logic;
  signal mem_to_bram_converter_hist_address1 : std_logic_vector(9 downto 0);
  signal mem_to_bram_converter_hist_dout1 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_hist_loadData : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_feature_ce0 : std_logic;
  signal mem_to_bram_converter_feature_we0 : std_logic;
  signal mem_to_bram_converter_feature_address0 : std_logic_vector(9 downto 0);
  signal mem_to_bram_converter_feature_dout0 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_feature_ce1 : std_logic;
  signal mem_to_bram_converter_feature_we1 : std_logic;
  signal mem_to_bram_converter_feature_address1 : std_logic_vector(9 downto 0);
  signal mem_to_bram_converter_feature_dout1 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_feature_loadData : std_logic_vector(31 downto 0);
  signal histogram_wrapped_feature_end_valid : std_logic;
  signal histogram_wrapped_feature_end_ready : std_logic;
  signal histogram_wrapped_weight_end_valid : std_logic;
  signal histogram_wrapped_weight_end_ready : std_logic;
  signal histogram_wrapped_hist_end_valid : std_logic;
  signal histogram_wrapped_hist_end_ready : std_logic;
  signal histogram_wrapped_end_valid : std_logic;
  signal histogram_wrapped_end_ready : std_logic;
  signal histogram_wrapped_feature_loadEn : std_logic;
  signal histogram_wrapped_feature_loadAddr : std_logic_vector(9 downto 0);
  signal histogram_wrapped_feature_storeEn : std_logic;
  signal histogram_wrapped_feature_storeAddr : std_logic_vector(9 downto 0);
  signal histogram_wrapped_feature_storeData : std_logic_vector(31 downto 0);
  signal histogram_wrapped_weight_loadEn : std_logic;
  signal histogram_wrapped_weight_loadAddr : std_logic_vector(9 downto 0);
  signal histogram_wrapped_weight_storeEn : std_logic;
  signal histogram_wrapped_weight_storeAddr : std_logic_vector(9 downto 0);
  signal histogram_wrapped_weight_storeData : std_logic_vector(31 downto 0);
  signal histogram_wrapped_hist_loadEn : std_logic;
  signal histogram_wrapped_hist_loadAddr : std_logic_vector(9 downto 0);
  signal histogram_wrapped_hist_storeEn : std_logic;
  signal histogram_wrapped_hist_storeAddr : std_logic_vector(9 downto 0);
  signal histogram_wrapped_hist_storeData : std_logic_vector(31 downto 0);

begin

  feature_end_valid <= histogram_wrapped_feature_end_valid;
  histogram_wrapped_feature_end_ready <= feature_end_ready;
  weight_end_valid <= histogram_wrapped_weight_end_valid;
  histogram_wrapped_weight_end_ready <= weight_end_ready;
  hist_end_valid <= histogram_wrapped_hist_end_valid;
  histogram_wrapped_hist_end_ready <= hist_end_ready;
  end_valid <= histogram_wrapped_end_valid;
  histogram_wrapped_end_ready <= end_ready;
  feature_ce0 <= mem_to_bram_converter_feature_ce0;
  feature_we0 <= mem_to_bram_converter_feature_we0;
  feature_address0 <= mem_to_bram_converter_feature_address0;
  feature_dout0 <= mem_to_bram_converter_feature_dout0;
  feature_ce1 <= mem_to_bram_converter_feature_ce1;
  feature_we1 <= mem_to_bram_converter_feature_we1;
  feature_address1 <= mem_to_bram_converter_feature_address1;
  feature_dout1 <= mem_to_bram_converter_feature_dout1;
  weight_ce0 <= mem_to_bram_converter_weight_ce0;
  weight_we0 <= mem_to_bram_converter_weight_we0;
  weight_address0 <= mem_to_bram_converter_weight_address0;
  weight_dout0 <= mem_to_bram_converter_weight_dout0;
  weight_ce1 <= mem_to_bram_converter_weight_ce1;
  weight_we1 <= mem_to_bram_converter_weight_we1;
  weight_address1 <= mem_to_bram_converter_weight_address1;
  weight_dout1 <= mem_to_bram_converter_weight_dout1;
  hist_ce0 <= mem_to_bram_converter_hist_ce0;
  hist_we0 <= mem_to_bram_converter_hist_we0;
  hist_address0 <= mem_to_bram_converter_hist_address0;
  hist_dout0 <= mem_to_bram_converter_hist_dout0;
  hist_ce1 <= mem_to_bram_converter_hist_ce1;
  hist_we1 <= mem_to_bram_converter_hist_we1;
  hist_address1 <= mem_to_bram_converter_hist_address1;
  hist_dout1 <= mem_to_bram_converter_hist_dout1;

  mem_to_bram_converter_weight : entity work.mem_to_bram(arch) generic map(32, 10)
    port map(
      loadEn => histogram_wrapped_weight_loadEn,
      loadAddr => histogram_wrapped_weight_loadAddr,
      storeEn => histogram_wrapped_weight_storeEn,
      storeAddr => histogram_wrapped_weight_storeAddr,
      storeData => histogram_wrapped_weight_storeData,
      din0 => weight_din0,
      din1 => weight_din1,
      ce0 => mem_to_bram_converter_weight_ce0,
      we0 => mem_to_bram_converter_weight_we0,
      address0 => mem_to_bram_converter_weight_address0,
      dout0 => mem_to_bram_converter_weight_dout0,
      ce1 => mem_to_bram_converter_weight_ce1,
      we1 => mem_to_bram_converter_weight_we1,
      address1 => mem_to_bram_converter_weight_address1,
      dout1 => mem_to_bram_converter_weight_dout1,
      loadData => mem_to_bram_converter_weight_loadData
    );

  mem_to_bram_converter_hist : entity work.mem_to_bram(arch) generic map(32, 10)
    port map(
      loadEn => histogram_wrapped_hist_loadEn,
      loadAddr => histogram_wrapped_hist_loadAddr,
      storeEn => histogram_wrapped_hist_storeEn,
      storeAddr => histogram_wrapped_hist_storeAddr,
      storeData => histogram_wrapped_hist_storeData,
      din0 => hist_din0,
      din1 => hist_din1,
      ce0 => mem_to_bram_converter_hist_ce0,
      we0 => mem_to_bram_converter_hist_we0,
      address0 => mem_to_bram_converter_hist_address0,
      dout0 => mem_to_bram_converter_hist_dout0,
      ce1 => mem_to_bram_converter_hist_ce1,
      we1 => mem_to_bram_converter_hist_we1,
      address1 => mem_to_bram_converter_hist_address1,
      dout1 => mem_to_bram_converter_hist_dout1,
      loadData => mem_to_bram_converter_hist_loadData
    );

  mem_to_bram_converter_feature : entity work.mem_to_bram(arch) generic map(32, 10)
    port map(
      loadEn => histogram_wrapped_feature_loadEn,
      loadAddr => histogram_wrapped_feature_loadAddr,
      storeEn => histogram_wrapped_feature_storeEn,
      storeAddr => histogram_wrapped_feature_storeAddr,
      storeData => histogram_wrapped_feature_storeData,
      din0 => feature_din0,
      din1 => feature_din1,
      ce0 => mem_to_bram_converter_feature_ce0,
      we0 => mem_to_bram_converter_feature_we0,
      address0 => mem_to_bram_converter_feature_address0,
      dout0 => mem_to_bram_converter_feature_dout0,
      ce1 => mem_to_bram_converter_feature_ce1,
      we1 => mem_to_bram_converter_feature_we1,
      address1 => mem_to_bram_converter_feature_address1,
      dout1 => mem_to_bram_converter_feature_dout1,
      loadData => mem_to_bram_converter_feature_loadData
    );

  histogram_wrapped : entity work.histogram(behavioral)
    port map(
      feature_loadData => mem_to_bram_converter_feature_loadData,
      weight_loadData => mem_to_bram_converter_weight_loadData,
      hist_loadData => mem_to_bram_converter_hist_loadData,
      n => n,
      n_valid => n_valid,
      n_ready => n_ready,
      feature_start_valid => feature_start_valid,
      feature_start_ready => feature_start_ready,
      weight_start_valid => weight_start_valid,
      weight_start_ready => weight_start_ready,
      hist_start_valid => hist_start_valid,
      hist_start_ready => hist_start_ready,
      start_valid => start_valid,
      start_ready => start_ready,
      clk => clk,
      rst => rst,
      feature_end_valid => histogram_wrapped_feature_end_valid,
      feature_end_ready => histogram_wrapped_feature_end_ready,
      weight_end_valid => histogram_wrapped_weight_end_valid,
      weight_end_ready => histogram_wrapped_weight_end_ready,
      hist_end_valid => histogram_wrapped_hist_end_valid,
      hist_end_ready => histogram_wrapped_hist_end_ready,
      end_valid => histogram_wrapped_end_valid,
      end_ready => histogram_wrapped_end_ready,
      feature_loadEn => histogram_wrapped_feature_loadEn,
      feature_loadAddr => histogram_wrapped_feature_loadAddr,
      feature_storeEn => histogram_wrapped_feature_storeEn,
      feature_storeAddr => histogram_wrapped_feature_storeAddr,
      feature_storeData => histogram_wrapped_feature_storeData,
      weight_loadEn => histogram_wrapped_weight_loadEn,
      weight_loadAddr => histogram_wrapped_weight_loadAddr,
      weight_storeEn => histogram_wrapped_weight_storeEn,
      weight_storeAddr => histogram_wrapped_weight_storeAddr,
      weight_storeData => histogram_wrapped_weight_storeData,
      hist_loadEn => histogram_wrapped_hist_loadEn,
      hist_loadAddr => histogram_wrapped_hist_loadAddr,
      hist_storeEn => histogram_wrapped_hist_storeEn,
      hist_storeAddr => histogram_wrapped_hist_storeAddr,
      hist_storeData => histogram_wrapped_hist_storeData
    );

end architecture;
