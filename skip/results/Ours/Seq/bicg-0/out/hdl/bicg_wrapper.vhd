library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity bicg_wrapper is
  port (
    a_din0 : in std_logic_vector(31 downto 0);
    a_din1 : in std_logic_vector(31 downto 0);
    s_din0 : in std_logic_vector(31 downto 0);
    s_din1 : in std_logic_vector(31 downto 0);
    q_din0 : in std_logic_vector(31 downto 0);
    q_din1 : in std_logic_vector(31 downto 0);
    p_din0 : in std_logic_vector(31 downto 0);
    p_din1 : in std_logic_vector(31 downto 0);
    r_din0 : in std_logic_vector(31 downto 0);
    r_din1 : in std_logic_vector(31 downto 0);
    a_start_valid : in std_logic;
    s_start_valid : in std_logic;
    q_start_valid : in std_logic;
    p_start_valid : in std_logic;
    r_start_valid : in std_logic;
    start_valid : in std_logic;
    clk : in std_logic;
    rst : in std_logic;
    out0_ready : in std_logic;
    a_end_ready : in std_logic;
    s_end_ready : in std_logic;
    q_end_ready : in std_logic;
    p_end_ready : in std_logic;
    r_end_ready : in std_logic;
    end_ready : in std_logic;
    a_start_ready : out std_logic;
    s_start_ready : out std_logic;
    q_start_ready : out std_logic;
    p_start_ready : out std_logic;
    r_start_ready : out std_logic;
    start_ready : out std_logic;
    out0 : out std_logic_vector(31 downto 0);
    out0_valid : out std_logic;
    a_end_valid : out std_logic;
    s_end_valid : out std_logic;
    q_end_valid : out std_logic;
    p_end_valid : out std_logic;
    r_end_valid : out std_logic;
    end_valid : out std_logic;
    a_ce0 : out std_logic;
    a_we0 : out std_logic;
    a_address0 : out std_logic_vector(9 downto 0);
    a_dout0 : out std_logic_vector(31 downto 0);
    a_ce1 : out std_logic;
    a_we1 : out std_logic;
    a_address1 : out std_logic_vector(9 downto 0);
    a_dout1 : out std_logic_vector(31 downto 0);
    s_ce0 : out std_logic;
    s_we0 : out std_logic;
    s_address0 : out std_logic_vector(4 downto 0);
    s_dout0 : out std_logic_vector(31 downto 0);
    s_ce1 : out std_logic;
    s_we1 : out std_logic;
    s_address1 : out std_logic_vector(4 downto 0);
    s_dout1 : out std_logic_vector(31 downto 0);
    q_ce0 : out std_logic;
    q_we0 : out std_logic;
    q_address0 : out std_logic_vector(4 downto 0);
    q_dout0 : out std_logic_vector(31 downto 0);
    q_ce1 : out std_logic;
    q_we1 : out std_logic;
    q_address1 : out std_logic_vector(4 downto 0);
    q_dout1 : out std_logic_vector(31 downto 0);
    p_ce0 : out std_logic;
    p_we0 : out std_logic;
    p_address0 : out std_logic_vector(4 downto 0);
    p_dout0 : out std_logic_vector(31 downto 0);
    p_ce1 : out std_logic;
    p_we1 : out std_logic;
    p_address1 : out std_logic_vector(4 downto 0);
    p_dout1 : out std_logic_vector(31 downto 0);
    r_ce0 : out std_logic;
    r_we0 : out std_logic;
    r_address0 : out std_logic_vector(4 downto 0);
    r_dout0 : out std_logic_vector(31 downto 0);
    r_ce1 : out std_logic;
    r_we1 : out std_logic;
    r_address1 : out std_logic_vector(4 downto 0);
    r_dout1 : out std_logic_vector(31 downto 0)
  );
end entity;

architecture behavioral of bicg_wrapper is

  signal mem_to_bram_converter_s_ce0 : std_logic;
  signal mem_to_bram_converter_s_we0 : std_logic;
  signal mem_to_bram_converter_s_address0 : std_logic_vector(4 downto 0);
  signal mem_to_bram_converter_s_dout0 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_s_ce1 : std_logic;
  signal mem_to_bram_converter_s_we1 : std_logic;
  signal mem_to_bram_converter_s_address1 : std_logic_vector(4 downto 0);
  signal mem_to_bram_converter_s_dout1 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_s_loadData : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_p_ce0 : std_logic;
  signal mem_to_bram_converter_p_we0 : std_logic;
  signal mem_to_bram_converter_p_address0 : std_logic_vector(4 downto 0);
  signal mem_to_bram_converter_p_dout0 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_p_ce1 : std_logic;
  signal mem_to_bram_converter_p_we1 : std_logic;
  signal mem_to_bram_converter_p_address1 : std_logic_vector(4 downto 0);
  signal mem_to_bram_converter_p_dout1 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_p_loadData : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_q_ce0 : std_logic;
  signal mem_to_bram_converter_q_we0 : std_logic;
  signal mem_to_bram_converter_q_address0 : std_logic_vector(4 downto 0);
  signal mem_to_bram_converter_q_dout0 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_q_ce1 : std_logic;
  signal mem_to_bram_converter_q_we1 : std_logic;
  signal mem_to_bram_converter_q_address1 : std_logic_vector(4 downto 0);
  signal mem_to_bram_converter_q_dout1 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_q_loadData : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_a_ce0 : std_logic;
  signal mem_to_bram_converter_a_we0 : std_logic;
  signal mem_to_bram_converter_a_address0 : std_logic_vector(9 downto 0);
  signal mem_to_bram_converter_a_dout0 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_a_ce1 : std_logic;
  signal mem_to_bram_converter_a_we1 : std_logic;
  signal mem_to_bram_converter_a_address1 : std_logic_vector(9 downto 0);
  signal mem_to_bram_converter_a_dout1 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_a_loadData : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_r_ce0 : std_logic;
  signal mem_to_bram_converter_r_we0 : std_logic;
  signal mem_to_bram_converter_r_address0 : std_logic_vector(4 downto 0);
  signal mem_to_bram_converter_r_dout0 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_r_ce1 : std_logic;
  signal mem_to_bram_converter_r_we1 : std_logic;
  signal mem_to_bram_converter_r_address1 : std_logic_vector(4 downto 0);
  signal mem_to_bram_converter_r_dout1 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_r_loadData : std_logic_vector(31 downto 0);
  signal bicg_wrapped_out0 : std_logic_vector(31 downto 0);
  signal bicg_wrapped_out0_valid : std_logic;
  signal bicg_wrapped_out0_ready : std_logic;
  signal bicg_wrapped_a_end_valid : std_logic;
  signal bicg_wrapped_a_end_ready : std_logic;
  signal bicg_wrapped_s_end_valid : std_logic;
  signal bicg_wrapped_s_end_ready : std_logic;
  signal bicg_wrapped_q_end_valid : std_logic;
  signal bicg_wrapped_q_end_ready : std_logic;
  signal bicg_wrapped_p_end_valid : std_logic;
  signal bicg_wrapped_p_end_ready : std_logic;
  signal bicg_wrapped_r_end_valid : std_logic;
  signal bicg_wrapped_r_end_ready : std_logic;
  signal bicg_wrapped_end_valid : std_logic;
  signal bicg_wrapped_end_ready : std_logic;
  signal bicg_wrapped_a_loadEn : std_logic;
  signal bicg_wrapped_a_loadAddr : std_logic_vector(9 downto 0);
  signal bicg_wrapped_a_storeEn : std_logic;
  signal bicg_wrapped_a_storeAddr : std_logic_vector(9 downto 0);
  signal bicg_wrapped_a_storeData : std_logic_vector(31 downto 0);
  signal bicg_wrapped_s_loadEn : std_logic;
  signal bicg_wrapped_s_loadAddr : std_logic_vector(4 downto 0);
  signal bicg_wrapped_s_storeEn : std_logic;
  signal bicg_wrapped_s_storeAddr : std_logic_vector(4 downto 0);
  signal bicg_wrapped_s_storeData : std_logic_vector(31 downto 0);
  signal bicg_wrapped_q_loadEn : std_logic;
  signal bicg_wrapped_q_loadAddr : std_logic_vector(4 downto 0);
  signal bicg_wrapped_q_storeEn : std_logic;
  signal bicg_wrapped_q_storeAddr : std_logic_vector(4 downto 0);
  signal bicg_wrapped_q_storeData : std_logic_vector(31 downto 0);
  signal bicg_wrapped_p_loadEn : std_logic;
  signal bicg_wrapped_p_loadAddr : std_logic_vector(4 downto 0);
  signal bicg_wrapped_p_storeEn : std_logic;
  signal bicg_wrapped_p_storeAddr : std_logic_vector(4 downto 0);
  signal bicg_wrapped_p_storeData : std_logic_vector(31 downto 0);
  signal bicg_wrapped_r_loadEn : std_logic;
  signal bicg_wrapped_r_loadAddr : std_logic_vector(4 downto 0);
  signal bicg_wrapped_r_storeEn : std_logic;
  signal bicg_wrapped_r_storeAddr : std_logic_vector(4 downto 0);
  signal bicg_wrapped_r_storeData : std_logic_vector(31 downto 0);

begin

  out0 <= bicg_wrapped_out0;
  out0_valid <= bicg_wrapped_out0_valid;
  bicg_wrapped_out0_ready <= out0_ready;
  a_end_valid <= bicg_wrapped_a_end_valid;
  bicg_wrapped_a_end_ready <= a_end_ready;
  s_end_valid <= bicg_wrapped_s_end_valid;
  bicg_wrapped_s_end_ready <= s_end_ready;
  q_end_valid <= bicg_wrapped_q_end_valid;
  bicg_wrapped_q_end_ready <= q_end_ready;
  p_end_valid <= bicg_wrapped_p_end_valid;
  bicg_wrapped_p_end_ready <= p_end_ready;
  r_end_valid <= bicg_wrapped_r_end_valid;
  bicg_wrapped_r_end_ready <= r_end_ready;
  end_valid <= bicg_wrapped_end_valid;
  bicg_wrapped_end_ready <= end_ready;
  a_ce0 <= mem_to_bram_converter_a_ce0;
  a_we0 <= mem_to_bram_converter_a_we0;
  a_address0 <= mem_to_bram_converter_a_address0;
  a_dout0 <= mem_to_bram_converter_a_dout0;
  a_ce1 <= mem_to_bram_converter_a_ce1;
  a_we1 <= mem_to_bram_converter_a_we1;
  a_address1 <= mem_to_bram_converter_a_address1;
  a_dout1 <= mem_to_bram_converter_a_dout1;
  s_ce0 <= mem_to_bram_converter_s_ce0;
  s_we0 <= mem_to_bram_converter_s_we0;
  s_address0 <= mem_to_bram_converter_s_address0;
  s_dout0 <= mem_to_bram_converter_s_dout0;
  s_ce1 <= mem_to_bram_converter_s_ce1;
  s_we1 <= mem_to_bram_converter_s_we1;
  s_address1 <= mem_to_bram_converter_s_address1;
  s_dout1 <= mem_to_bram_converter_s_dout1;
  q_ce0 <= mem_to_bram_converter_q_ce0;
  q_we0 <= mem_to_bram_converter_q_we0;
  q_address0 <= mem_to_bram_converter_q_address0;
  q_dout0 <= mem_to_bram_converter_q_dout0;
  q_ce1 <= mem_to_bram_converter_q_ce1;
  q_we1 <= mem_to_bram_converter_q_we1;
  q_address1 <= mem_to_bram_converter_q_address1;
  q_dout1 <= mem_to_bram_converter_q_dout1;
  p_ce0 <= mem_to_bram_converter_p_ce0;
  p_we0 <= mem_to_bram_converter_p_we0;
  p_address0 <= mem_to_bram_converter_p_address0;
  p_dout0 <= mem_to_bram_converter_p_dout0;
  p_ce1 <= mem_to_bram_converter_p_ce1;
  p_we1 <= mem_to_bram_converter_p_we1;
  p_address1 <= mem_to_bram_converter_p_address1;
  p_dout1 <= mem_to_bram_converter_p_dout1;
  r_ce0 <= mem_to_bram_converter_r_ce0;
  r_we0 <= mem_to_bram_converter_r_we0;
  r_address0 <= mem_to_bram_converter_r_address0;
  r_dout0 <= mem_to_bram_converter_r_dout0;
  r_ce1 <= mem_to_bram_converter_r_ce1;
  r_we1 <= mem_to_bram_converter_r_we1;
  r_address1 <= mem_to_bram_converter_r_address1;
  r_dout1 <= mem_to_bram_converter_r_dout1;

  mem_to_bram_converter_s : entity work.mem_to_bram(arch) generic map(32, 5)
    port map(
      loadEn => bicg_wrapped_s_loadEn,
      loadAddr => bicg_wrapped_s_loadAddr,
      storeEn => bicg_wrapped_s_storeEn,
      storeAddr => bicg_wrapped_s_storeAddr,
      storeData => bicg_wrapped_s_storeData,
      din0 => s_din0,
      din1 => s_din1,
      ce0 => mem_to_bram_converter_s_ce0,
      we0 => mem_to_bram_converter_s_we0,
      address0 => mem_to_bram_converter_s_address0,
      dout0 => mem_to_bram_converter_s_dout0,
      ce1 => mem_to_bram_converter_s_ce1,
      we1 => mem_to_bram_converter_s_we1,
      address1 => mem_to_bram_converter_s_address1,
      dout1 => mem_to_bram_converter_s_dout1,
      loadData => mem_to_bram_converter_s_loadData
    );

  mem_to_bram_converter_p : entity work.mem_to_bram(arch) generic map(32, 5)
    port map(
      loadEn => bicg_wrapped_p_loadEn,
      loadAddr => bicg_wrapped_p_loadAddr,
      storeEn => bicg_wrapped_p_storeEn,
      storeAddr => bicg_wrapped_p_storeAddr,
      storeData => bicg_wrapped_p_storeData,
      din0 => p_din0,
      din1 => p_din1,
      ce0 => mem_to_bram_converter_p_ce0,
      we0 => mem_to_bram_converter_p_we0,
      address0 => mem_to_bram_converter_p_address0,
      dout0 => mem_to_bram_converter_p_dout0,
      ce1 => mem_to_bram_converter_p_ce1,
      we1 => mem_to_bram_converter_p_we1,
      address1 => mem_to_bram_converter_p_address1,
      dout1 => mem_to_bram_converter_p_dout1,
      loadData => mem_to_bram_converter_p_loadData
    );

  mem_to_bram_converter_q : entity work.mem_to_bram(arch) generic map(32, 5)
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

  mem_to_bram_converter_a : entity work.mem_to_bram(arch) generic map(32, 10)
    port map(
      loadEn => bicg_wrapped_a_loadEn,
      loadAddr => bicg_wrapped_a_loadAddr,
      storeEn => bicg_wrapped_a_storeEn,
      storeAddr => bicg_wrapped_a_storeAddr,
      storeData => bicg_wrapped_a_storeData,
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

  mem_to_bram_converter_r : entity work.mem_to_bram(arch) generic map(32, 5)
    port map(
      loadEn => bicg_wrapped_r_loadEn,
      loadAddr => bicg_wrapped_r_loadAddr,
      storeEn => bicg_wrapped_r_storeEn,
      storeAddr => bicg_wrapped_r_storeAddr,
      storeData => bicg_wrapped_r_storeData,
      din0 => r_din0,
      din1 => r_din1,
      ce0 => mem_to_bram_converter_r_ce0,
      we0 => mem_to_bram_converter_r_we0,
      address0 => mem_to_bram_converter_r_address0,
      dout0 => mem_to_bram_converter_r_dout0,
      ce1 => mem_to_bram_converter_r_ce1,
      we1 => mem_to_bram_converter_r_we1,
      address1 => mem_to_bram_converter_r_address1,
      dout1 => mem_to_bram_converter_r_dout1,
      loadData => mem_to_bram_converter_r_loadData
    );

  bicg_wrapped : entity work.bicg(behavioral)
    port map(
      a_loadData => mem_to_bram_converter_a_loadData,
      s_loadData => mem_to_bram_converter_s_loadData,
      q_loadData => mem_to_bram_converter_q_loadData,
      p_loadData => mem_to_bram_converter_p_loadData,
      r_loadData => mem_to_bram_converter_r_loadData,
      a_start_valid => a_start_valid,
      a_start_ready => a_start_ready,
      s_start_valid => s_start_valid,
      s_start_ready => s_start_ready,
      q_start_valid => q_start_valid,
      q_start_ready => q_start_ready,
      p_start_valid => p_start_valid,
      p_start_ready => p_start_ready,
      r_start_valid => r_start_valid,
      r_start_ready => r_start_ready,
      start_valid => start_valid,
      start_ready => start_ready,
      clk => clk,
      rst => rst,
      out0 => bicg_wrapped_out0,
      out0_valid => bicg_wrapped_out0_valid,
      out0_ready => bicg_wrapped_out0_ready,
      a_end_valid => bicg_wrapped_a_end_valid,
      a_end_ready => bicg_wrapped_a_end_ready,
      s_end_valid => bicg_wrapped_s_end_valid,
      s_end_ready => bicg_wrapped_s_end_ready,
      q_end_valid => bicg_wrapped_q_end_valid,
      q_end_ready => bicg_wrapped_q_end_ready,
      p_end_valid => bicg_wrapped_p_end_valid,
      p_end_ready => bicg_wrapped_p_end_ready,
      r_end_valid => bicg_wrapped_r_end_valid,
      r_end_ready => bicg_wrapped_r_end_ready,
      end_valid => bicg_wrapped_end_valid,
      end_ready => bicg_wrapped_end_ready,
      a_loadEn => bicg_wrapped_a_loadEn,
      a_loadAddr => bicg_wrapped_a_loadAddr,
      a_storeEn => bicg_wrapped_a_storeEn,
      a_storeAddr => bicg_wrapped_a_storeAddr,
      a_storeData => bicg_wrapped_a_storeData,
      s_loadEn => bicg_wrapped_s_loadEn,
      s_loadAddr => bicg_wrapped_s_loadAddr,
      s_storeEn => bicg_wrapped_s_storeEn,
      s_storeAddr => bicg_wrapped_s_storeAddr,
      s_storeData => bicg_wrapped_s_storeData,
      q_loadEn => bicg_wrapped_q_loadEn,
      q_loadAddr => bicg_wrapped_q_loadAddr,
      q_storeEn => bicg_wrapped_q_storeEn,
      q_storeAddr => bicg_wrapped_q_storeAddr,
      q_storeData => bicg_wrapped_q_storeData,
      p_loadEn => bicg_wrapped_p_loadEn,
      p_loadAddr => bicg_wrapped_p_loadAddr,
      p_storeEn => bicg_wrapped_p_storeEn,
      p_storeAddr => bicg_wrapped_p_storeAddr,
      p_storeData => bicg_wrapped_p_storeData,
      r_loadEn => bicg_wrapped_r_loadEn,
      r_loadAddr => bicg_wrapped_r_loadAddr,
      r_storeEn => bicg_wrapped_r_storeEn,
      r_storeAddr => bicg_wrapped_r_storeAddr,
      r_storeData => bicg_wrapped_r_storeData
    );

end architecture;
