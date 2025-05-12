library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity bicg_wrapper is
  port (
    q_din0 : in std_logic_vector(31 downto 0);
    q_din1 : in std_logic_vector(31 downto 0);
    q_start_valid : in std_logic;
    start_valid : in std_logic;
    clk : in std_logic;
    rst : in std_logic;
    out0_ready : in std_logic;
    q_end_ready : in std_logic;
    end_ready : in std_logic;
    q_start_ready : out std_logic;
    start_ready : out std_logic;
    out0 : out std_logic_vector(31 downto 0);
    out0_valid : out std_logic;
    q_end_valid : out std_logic;
    end_valid : out std_logic;
    q_ce0 : out std_logic;
    q_we0 : out std_logic;
    q_address0 : out std_logic_vector(4 downto 0);
    q_dout0 : out std_logic_vector(31 downto 0);
    q_ce1 : out std_logic;
    q_we1 : out std_logic;
    q_address1 : out std_logic_vector(4 downto 0);
    q_dout1 : out std_logic_vector(31 downto 0)
  );
end entity;

architecture behavioral of bicg_wrapper is

  signal mem_to_bram_converter_q_ce0 : std_logic;
  signal mem_to_bram_converter_q_we0 : std_logic;
  signal mem_to_bram_converter_q_address0 : std_logic_vector(4 downto 0);
  signal mem_to_bram_converter_q_dout0 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_q_ce1 : std_logic;
  signal mem_to_bram_converter_q_we1 : std_logic;
  signal mem_to_bram_converter_q_address1 : std_logic_vector(4 downto 0);
  signal mem_to_bram_converter_q_dout1 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_q_loadData : std_logic_vector(31 downto 0);
  signal bicg_wrapped_out0 : std_logic_vector(31 downto 0);
  signal bicg_wrapped_out0_valid : std_logic;
  signal bicg_wrapped_out0_ready : std_logic;
  signal bicg_wrapped_q_end_valid : std_logic;
  signal bicg_wrapped_q_end_ready : std_logic;
  signal bicg_wrapped_end_valid : std_logic;
  signal bicg_wrapped_end_ready : std_logic;
  signal bicg_wrapped_q_loadEn : std_logic;
  signal bicg_wrapped_q_loadAddr : std_logic_vector(4 downto 0);
  signal bicg_wrapped_q_storeEn : std_logic;
  signal bicg_wrapped_q_storeAddr : std_logic_vector(4 downto 0);
  signal bicg_wrapped_q_storeData : std_logic_vector(31 downto 0);

begin

  out0 <= bicg_wrapped_out0;
  out0_valid <= bicg_wrapped_out0_valid;
  bicg_wrapped_out0_ready <= out0_ready;
  q_end_valid <= bicg_wrapped_q_end_valid;
  bicg_wrapped_q_end_ready <= q_end_ready;
  end_valid <= bicg_wrapped_end_valid;
  bicg_wrapped_end_ready <= end_ready;
  q_ce0 <= mem_to_bram_converter_q_ce0;
  q_we0 <= mem_to_bram_converter_q_we0;
  q_address0 <= mem_to_bram_converter_q_address0;
  q_dout0 <= mem_to_bram_converter_q_dout0;
  q_ce1 <= mem_to_bram_converter_q_ce1;
  q_we1 <= mem_to_bram_converter_q_we1;
  q_address1 <= mem_to_bram_converter_q_address1;
  q_dout1 <= mem_to_bram_converter_q_dout1;

  mem_to_bram_converter_q : entity work.mem_to_bram_32_5(arch)
    port map(
      loadEn => bicg_wrapped_q_loadEn,
      loadAddr => bicg_wrapped_q_loadAddr,
      storeEn => bicg_wrapped_q_storeEn,
      storeAddr => bicg_wrapped_q_storeAddr,
      storeData => bicg_wrapped_q_storeData,
      din0 => q_din0,
      din1 => q_din1,
      ce0 => mem_to_bram_converter_q_ce0,
      we0 => mem_to_bram_converter_q_we0,
      address0 => mem_to_bram_converter_q_address0,
      dout0 => mem_to_bram_converter_q_dout0,
      ce1 => mem_to_bram_converter_q_ce1,
      we1 => mem_to_bram_converter_q_we1,
      address1 => mem_to_bram_converter_q_address1,
      dout1 => mem_to_bram_converter_q_dout1,
      loadData => mem_to_bram_converter_q_loadData
    );

  bicg_wrapped : entity work.bicg(behavioral)
    port map(
      q_loadData => mem_to_bram_converter_q_loadData,
      q_start_valid => q_start_valid,
      q_start_ready => q_start_ready,
      start_valid => start_valid,
      start_ready => start_ready,
      clk => clk,
      rst => rst,
      out0 => bicg_wrapped_out0,
      out0_valid => bicg_wrapped_out0_valid,
      out0_ready => bicg_wrapped_out0_ready,
      q_end_valid => bicg_wrapped_q_end_valid,
      q_end_ready => bicg_wrapped_q_end_ready,
      end_valid => bicg_wrapped_end_valid,
      end_ready => bicg_wrapped_end_ready,
      q_loadEn => bicg_wrapped_q_loadEn,
      q_loadAddr => bicg_wrapped_q_loadAddr,
      q_storeEn => bicg_wrapped_q_storeEn,
      q_storeAddr => bicg_wrapped_q_storeAddr,
      q_storeData => bicg_wrapped_q_storeData
    );

end architecture;
