library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity if_float2_wrapper is
  port (
    x0 : in std_logic_vector(31 downto 0);
    x0_valid : in std_logic;
    a_din0 : in std_logic_vector(31 downto 0);
    a_din1 : in std_logic_vector(31 downto 0);
    minus_trace_din0 : in std_logic_vector(31 downto 0);
    minus_trace_din1 : in std_logic_vector(31 downto 0);
    a_start_valid : in std_logic;
    minus_trace_start_valid : in std_logic;
    start_valid : in std_logic;
    clk : in std_logic;
    rst : in std_logic;
    out0_ready : in std_logic;
    a_end_ready : in std_logic;
    minus_trace_end_ready : in std_logic;
    end_ready : in std_logic;
    x0_ready : out std_logic;
    a_start_ready : out std_logic;
    minus_trace_start_ready : out std_logic;
    start_ready : out std_logic;
    out0 : out std_logic_vector(31 downto 0);
    out0_valid : out std_logic;
    a_end_valid : out std_logic;
    minus_trace_end_valid : out std_logic;
    end_valid : out std_logic;
    a_ce0 : out std_logic;
    a_we0 : out std_logic;
    a_address0 : out std_logic_vector(6 downto 0);
    a_dout0 : out std_logic_vector(31 downto 0);
    a_ce1 : out std_logic;
    a_we1 : out std_logic;
    a_address1 : out std_logic_vector(6 downto 0);
    a_dout1 : out std_logic_vector(31 downto 0);
    minus_trace_ce0 : out std_logic;
    minus_trace_we0 : out std_logic;
    minus_trace_address0 : out std_logic_vector(6 downto 0);
    minus_trace_dout0 : out std_logic_vector(31 downto 0);
    minus_trace_ce1 : out std_logic;
    minus_trace_we1 : out std_logic;
    minus_trace_address1 : out std_logic_vector(6 downto 0);
    minus_trace_dout1 : out std_logic_vector(31 downto 0)
  );
end entity;

architecture behavioral of if_float2_wrapper is

  signal mem_to_bram_converter_minus_trace_ce0 : std_logic;
  signal mem_to_bram_converter_minus_trace_we0 : std_logic;
  signal mem_to_bram_converter_minus_trace_address0 : std_logic_vector(6 downto 0);
  signal mem_to_bram_converter_minus_trace_dout0 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_minus_trace_ce1 : std_logic;
  signal mem_to_bram_converter_minus_trace_we1 : std_logic;
  signal mem_to_bram_converter_minus_trace_address1 : std_logic_vector(6 downto 0);
  signal mem_to_bram_converter_minus_trace_dout1 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_minus_trace_loadData : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_a_ce0 : std_logic;
  signal mem_to_bram_converter_a_we0 : std_logic;
  signal mem_to_bram_converter_a_address0 : std_logic_vector(6 downto 0);
  signal mem_to_bram_converter_a_dout0 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_a_ce1 : std_logic;
  signal mem_to_bram_converter_a_we1 : std_logic;
  signal mem_to_bram_converter_a_address1 : std_logic_vector(6 downto 0);
  signal mem_to_bram_converter_a_dout1 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_a_loadData : std_logic_vector(31 downto 0);
  signal if_float2_wrapped_out0 : std_logic_vector(31 downto 0);
  signal if_float2_wrapped_out0_valid : std_logic;
  signal if_float2_wrapped_out0_ready : std_logic;
  signal if_float2_wrapped_a_end_valid : std_logic;
  signal if_float2_wrapped_a_end_ready : std_logic;
  signal if_float2_wrapped_minus_trace_end_valid : std_logic;
  signal if_float2_wrapped_minus_trace_end_ready : std_logic;
  signal if_float2_wrapped_end_valid : std_logic;
  signal if_float2_wrapped_end_ready : std_logic;
  signal if_float2_wrapped_a_loadEn : std_logic;
  signal if_float2_wrapped_a_loadAddr : std_logic_vector(6 downto 0);
  signal if_float2_wrapped_a_storeEn : std_logic;
  signal if_float2_wrapped_a_storeAddr : std_logic_vector(6 downto 0);
  signal if_float2_wrapped_a_storeData : std_logic_vector(31 downto 0);
  signal if_float2_wrapped_minus_trace_loadEn : std_logic;
  signal if_float2_wrapped_minus_trace_loadAddr : std_logic_vector(6 downto 0);
  signal if_float2_wrapped_minus_trace_storeEn : std_logic;
  signal if_float2_wrapped_minus_trace_storeAddr : std_logic_vector(6 downto 0);
  signal if_float2_wrapped_minus_trace_storeData : std_logic_vector(31 downto 0);

begin

  out0 <= if_float2_wrapped_out0;
  out0_valid <= if_float2_wrapped_out0_valid;
  if_float2_wrapped_out0_ready <= out0_ready;
  a_end_valid <= if_float2_wrapped_a_end_valid;
  if_float2_wrapped_a_end_ready <= a_end_ready;
  minus_trace_end_valid <= if_float2_wrapped_minus_trace_end_valid;
  if_float2_wrapped_minus_trace_end_ready <= minus_trace_end_ready;
  end_valid <= if_float2_wrapped_end_valid;
  if_float2_wrapped_end_ready <= end_ready;
  a_ce0 <= mem_to_bram_converter_a_ce0;
  a_we0 <= mem_to_bram_converter_a_we0;
  a_address0 <= mem_to_bram_converter_a_address0;
  a_dout0 <= mem_to_bram_converter_a_dout0;
  a_ce1 <= mem_to_bram_converter_a_ce1;
  a_we1 <= mem_to_bram_converter_a_we1;
  a_address1 <= mem_to_bram_converter_a_address1;
  a_dout1 <= mem_to_bram_converter_a_dout1;
  minus_trace_ce0 <= mem_to_bram_converter_minus_trace_ce0;
  minus_trace_we0 <= mem_to_bram_converter_minus_trace_we0;
  minus_trace_address0 <= mem_to_bram_converter_minus_trace_address0;
  minus_trace_dout0 <= mem_to_bram_converter_minus_trace_dout0;
  minus_trace_ce1 <= mem_to_bram_converter_minus_trace_ce1;
  minus_trace_we1 <= mem_to_bram_converter_minus_trace_we1;
  minus_trace_address1 <= mem_to_bram_converter_minus_trace_address1;
  minus_trace_dout1 <= mem_to_bram_converter_minus_trace_dout1;

  mem_to_bram_converter_minus_trace : entity work.mem_to_bram_32_7(arch)
    port map(
      loadEn => if_float2_wrapped_minus_trace_loadEn,
      loadAddr => if_float2_wrapped_minus_trace_loadAddr,
      storeEn => if_float2_wrapped_minus_trace_storeEn,
      storeAddr => if_float2_wrapped_minus_trace_storeAddr,
      storeData => if_float2_wrapped_minus_trace_storeData,
      din0 => minus_trace_din0,
      din1 => minus_trace_din1,
      ce0 => mem_to_bram_converter_minus_trace_ce0,
      we0 => mem_to_bram_converter_minus_trace_we0,
      address0 => mem_to_bram_converter_minus_trace_address0,
      dout0 => mem_to_bram_converter_minus_trace_dout0,
      ce1 => mem_to_bram_converter_minus_trace_ce1,
      we1 => mem_to_bram_converter_minus_trace_we1,
      address1 => mem_to_bram_converter_minus_trace_address1,
      dout1 => mem_to_bram_converter_minus_trace_dout1,
      loadData => mem_to_bram_converter_minus_trace_loadData
    );

  mem_to_bram_converter_a : entity work.mem_to_bram_32_7(arch)
    port map(
      loadEn => if_float2_wrapped_a_loadEn,
      loadAddr => if_float2_wrapped_a_loadAddr,
      storeEn => if_float2_wrapped_a_storeEn,
      storeAddr => if_float2_wrapped_a_storeAddr,
      storeData => if_float2_wrapped_a_storeData,
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

  if_float2_wrapped : entity work.if_float2(behavioral)
    port map(
      x0 => x0,
      x0_valid => x0_valid,
      x0_ready => x0_ready,
      a_loadData => mem_to_bram_converter_a_loadData,
      minus_trace_loadData => mem_to_bram_converter_minus_trace_loadData,
      a_start_valid => a_start_valid,
      a_start_ready => a_start_ready,
      minus_trace_start_valid => minus_trace_start_valid,
      minus_trace_start_ready => minus_trace_start_ready,
      start_valid => start_valid,
      start_ready => start_ready,
      clk => clk,
      rst => rst,
      out0 => if_float2_wrapped_out0,
      out0_valid => if_float2_wrapped_out0_valid,
      out0_ready => if_float2_wrapped_out0_ready,
      a_end_valid => if_float2_wrapped_a_end_valid,
      a_end_ready => if_float2_wrapped_a_end_ready,
      minus_trace_end_valid => if_float2_wrapped_minus_trace_end_valid,
      minus_trace_end_ready => if_float2_wrapped_minus_trace_end_ready,
      end_valid => if_float2_wrapped_end_valid,
      end_ready => if_float2_wrapped_end_ready,
      a_loadEn => if_float2_wrapped_a_loadEn,
      a_loadAddr => if_float2_wrapped_a_loadAddr,
      a_storeEn => if_float2_wrapped_a_storeEn,
      a_storeAddr => if_float2_wrapped_a_storeAddr,
      a_storeData => if_float2_wrapped_a_storeData,
      minus_trace_loadEn => if_float2_wrapped_minus_trace_loadEn,
      minus_trace_loadAddr => if_float2_wrapped_minus_trace_loadAddr,
      minus_trace_storeEn => if_float2_wrapped_minus_trace_storeEn,
      minus_trace_storeAddr => if_float2_wrapped_minus_trace_storeAddr,
      minus_trace_storeData => if_float2_wrapped_minus_trace_storeData
    );

end architecture;
