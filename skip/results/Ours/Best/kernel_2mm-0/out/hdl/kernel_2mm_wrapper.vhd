library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity kernel_2mm_wrapper is
  port (
    alpha : in std_logic_vector(31 downto 0);
    alpha_valid : in std_logic;
    beta : in std_logic_vector(31 downto 0);
    beta_valid : in std_logic;
    tmp_din0 : in std_logic_vector(31 downto 0);
    tmp_din1 : in std_logic_vector(31 downto 0);
    A_din0 : in std_logic_vector(31 downto 0);
    A_din1 : in std_logic_vector(31 downto 0);
    B_din0 : in std_logic_vector(31 downto 0);
    B_din1 : in std_logic_vector(31 downto 0);
    C_din0 : in std_logic_vector(31 downto 0);
    C_din1 : in std_logic_vector(31 downto 0);
    D_din0 : in std_logic_vector(31 downto 0);
    D_din1 : in std_logic_vector(31 downto 0);
    tmp_start_valid : in std_logic;
    A_start_valid : in std_logic;
    B_start_valid : in std_logic;
    C_start_valid : in std_logic;
    D_start_valid : in std_logic;
    start_valid : in std_logic;
    clk : in std_logic;
    rst : in std_logic;
    tmp_end_ready : in std_logic;
    A_end_ready : in std_logic;
    B_end_ready : in std_logic;
    C_end_ready : in std_logic;
    D_end_ready : in std_logic;
    end_ready : in std_logic;
    alpha_ready : out std_logic;
    beta_ready : out std_logic;
    tmp_start_ready : out std_logic;
    A_start_ready : out std_logic;
    B_start_ready : out std_logic;
    C_start_ready : out std_logic;
    D_start_ready : out std_logic;
    start_ready : out std_logic;
    tmp_end_valid : out std_logic;
    A_end_valid : out std_logic;
    B_end_valid : out std_logic;
    C_end_valid : out std_logic;
    D_end_valid : out std_logic;
    end_valid : out std_logic;
    tmp_ce0 : out std_logic;
    tmp_we0 : out std_logic;
    tmp_address0 : out std_logic_vector(6 downto 0);
    tmp_dout0 : out std_logic_vector(31 downto 0);
    tmp_ce1 : out std_logic;
    tmp_we1 : out std_logic;
    tmp_address1 : out std_logic_vector(6 downto 0);
    tmp_dout1 : out std_logic_vector(31 downto 0);
    A_ce0 : out std_logic;
    A_we0 : out std_logic;
    A_address0 : out std_logic_vector(6 downto 0);
    A_dout0 : out std_logic_vector(31 downto 0);
    A_ce1 : out std_logic;
    A_we1 : out std_logic;
    A_address1 : out std_logic_vector(6 downto 0);
    A_dout1 : out std_logic_vector(31 downto 0);
    B_ce0 : out std_logic;
    B_we0 : out std_logic;
    B_address0 : out std_logic_vector(6 downto 0);
    B_dout0 : out std_logic_vector(31 downto 0);
    B_ce1 : out std_logic;
    B_we1 : out std_logic;
    B_address1 : out std_logic_vector(6 downto 0);
    B_dout1 : out std_logic_vector(31 downto 0);
    C_ce0 : out std_logic;
    C_we0 : out std_logic;
    C_address0 : out std_logic_vector(6 downto 0);
    C_dout0 : out std_logic_vector(31 downto 0);
    C_ce1 : out std_logic;
    C_we1 : out std_logic;
    C_address1 : out std_logic_vector(6 downto 0);
    C_dout1 : out std_logic_vector(31 downto 0);
    D_ce0 : out std_logic;
    D_we0 : out std_logic;
    D_address0 : out std_logic_vector(6 downto 0);
    D_dout0 : out std_logic_vector(31 downto 0);
    D_ce1 : out std_logic;
    D_we1 : out std_logic;
    D_address1 : out std_logic_vector(6 downto 0);
    D_dout1 : out std_logic_vector(31 downto 0)
  );
end entity;

architecture behavioral of kernel_2mm_wrapper is

  signal mem_to_bram_converter_C_ce0 : std_logic;
  signal mem_to_bram_converter_C_we0 : std_logic;
  signal mem_to_bram_converter_C_address0 : std_logic_vector(6 downto 0);
  signal mem_to_bram_converter_C_dout0 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_C_ce1 : std_logic;
  signal mem_to_bram_converter_C_we1 : std_logic;
  signal mem_to_bram_converter_C_address1 : std_logic_vector(6 downto 0);
  signal mem_to_bram_converter_C_dout1 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_C_loadData : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_tmp_ce0 : std_logic;
  signal mem_to_bram_converter_tmp_we0 : std_logic;
  signal mem_to_bram_converter_tmp_address0 : std_logic_vector(6 downto 0);
  signal mem_to_bram_converter_tmp_dout0 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_tmp_ce1 : std_logic;
  signal mem_to_bram_converter_tmp_we1 : std_logic;
  signal mem_to_bram_converter_tmp_address1 : std_logic_vector(6 downto 0);
  signal mem_to_bram_converter_tmp_dout1 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_tmp_loadData : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_D_ce0 : std_logic;
  signal mem_to_bram_converter_D_we0 : std_logic;
  signal mem_to_bram_converter_D_address0 : std_logic_vector(6 downto 0);
  signal mem_to_bram_converter_D_dout0 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_D_ce1 : std_logic;
  signal mem_to_bram_converter_D_we1 : std_logic;
  signal mem_to_bram_converter_D_address1 : std_logic_vector(6 downto 0);
  signal mem_to_bram_converter_D_dout1 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_D_loadData : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_B_ce0 : std_logic;
  signal mem_to_bram_converter_B_we0 : std_logic;
  signal mem_to_bram_converter_B_address0 : std_logic_vector(6 downto 0);
  signal mem_to_bram_converter_B_dout0 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_B_ce1 : std_logic;
  signal mem_to_bram_converter_B_we1 : std_logic;
  signal mem_to_bram_converter_B_address1 : std_logic_vector(6 downto 0);
  signal mem_to_bram_converter_B_dout1 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_B_loadData : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_A_ce0 : std_logic;
  signal mem_to_bram_converter_A_we0 : std_logic;
  signal mem_to_bram_converter_A_address0 : std_logic_vector(6 downto 0);
  signal mem_to_bram_converter_A_dout0 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_A_ce1 : std_logic;
  signal mem_to_bram_converter_A_we1 : std_logic;
  signal mem_to_bram_converter_A_address1 : std_logic_vector(6 downto 0);
  signal mem_to_bram_converter_A_dout1 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_A_loadData : std_logic_vector(31 downto 0);
  signal kernel_2mm_wrapped_tmp_end_valid : std_logic;
  signal kernel_2mm_wrapped_tmp_end_ready : std_logic;
  signal kernel_2mm_wrapped_A_end_valid : std_logic;
  signal kernel_2mm_wrapped_A_end_ready : std_logic;
  signal kernel_2mm_wrapped_B_end_valid : std_logic;
  signal kernel_2mm_wrapped_B_end_ready : std_logic;
  signal kernel_2mm_wrapped_C_end_valid : std_logic;
  signal kernel_2mm_wrapped_C_end_ready : std_logic;
  signal kernel_2mm_wrapped_D_end_valid : std_logic;
  signal kernel_2mm_wrapped_D_end_ready : std_logic;
  signal kernel_2mm_wrapped_end_valid : std_logic;
  signal kernel_2mm_wrapped_end_ready : std_logic;
  signal kernel_2mm_wrapped_tmp_loadEn : std_logic;
  signal kernel_2mm_wrapped_tmp_loadAddr : std_logic_vector(6 downto 0);
  signal kernel_2mm_wrapped_tmp_storeEn : std_logic;
  signal kernel_2mm_wrapped_tmp_storeAddr : std_logic_vector(6 downto 0);
  signal kernel_2mm_wrapped_tmp_storeData : std_logic_vector(31 downto 0);
  signal kernel_2mm_wrapped_A_loadEn : std_logic;
  signal kernel_2mm_wrapped_A_loadAddr : std_logic_vector(6 downto 0);
  signal kernel_2mm_wrapped_A_storeEn : std_logic;
  signal kernel_2mm_wrapped_A_storeAddr : std_logic_vector(6 downto 0);
  signal kernel_2mm_wrapped_A_storeData : std_logic_vector(31 downto 0);
  signal kernel_2mm_wrapped_B_loadEn : std_logic;
  signal kernel_2mm_wrapped_B_loadAddr : std_logic_vector(6 downto 0);
  signal kernel_2mm_wrapped_B_storeEn : std_logic;
  signal kernel_2mm_wrapped_B_storeAddr : std_logic_vector(6 downto 0);
  signal kernel_2mm_wrapped_B_storeData : std_logic_vector(31 downto 0);
  signal kernel_2mm_wrapped_C_loadEn : std_logic;
  signal kernel_2mm_wrapped_C_loadAddr : std_logic_vector(6 downto 0);
  signal kernel_2mm_wrapped_C_storeEn : std_logic;
  signal kernel_2mm_wrapped_C_storeAddr : std_logic_vector(6 downto 0);
  signal kernel_2mm_wrapped_C_storeData : std_logic_vector(31 downto 0);
  signal kernel_2mm_wrapped_D_loadEn : std_logic;
  signal kernel_2mm_wrapped_D_loadAddr : std_logic_vector(6 downto 0);
  signal kernel_2mm_wrapped_D_storeEn : std_logic;
  signal kernel_2mm_wrapped_D_storeAddr : std_logic_vector(6 downto 0);
  signal kernel_2mm_wrapped_D_storeData : std_logic_vector(31 downto 0);

begin

  tmp_end_valid <= kernel_2mm_wrapped_tmp_end_valid;
  kernel_2mm_wrapped_tmp_end_ready <= tmp_end_ready;
  A_end_valid <= kernel_2mm_wrapped_A_end_valid;
  kernel_2mm_wrapped_A_end_ready <= A_end_ready;
  B_end_valid <= kernel_2mm_wrapped_B_end_valid;
  kernel_2mm_wrapped_B_end_ready <= B_end_ready;
  C_end_valid <= kernel_2mm_wrapped_C_end_valid;
  kernel_2mm_wrapped_C_end_ready <= C_end_ready;
  D_end_valid <= kernel_2mm_wrapped_D_end_valid;
  kernel_2mm_wrapped_D_end_ready <= D_end_ready;
  end_valid <= kernel_2mm_wrapped_end_valid;
  kernel_2mm_wrapped_end_ready <= end_ready;
  tmp_ce0 <= mem_to_bram_converter_tmp_ce0;
  tmp_we0 <= mem_to_bram_converter_tmp_we0;
  tmp_address0 <= mem_to_bram_converter_tmp_address0;
  tmp_dout0 <= mem_to_bram_converter_tmp_dout0;
  tmp_ce1 <= mem_to_bram_converter_tmp_ce1;
  tmp_we1 <= mem_to_bram_converter_tmp_we1;
  tmp_address1 <= mem_to_bram_converter_tmp_address1;
  tmp_dout1 <= mem_to_bram_converter_tmp_dout1;
  A_ce0 <= mem_to_bram_converter_A_ce0;
  A_we0 <= mem_to_bram_converter_A_we0;
  A_address0 <= mem_to_bram_converter_A_address0;
  A_dout0 <= mem_to_bram_converter_A_dout0;
  A_ce1 <= mem_to_bram_converter_A_ce1;
  A_we1 <= mem_to_bram_converter_A_we1;
  A_address1 <= mem_to_bram_converter_A_address1;
  A_dout1 <= mem_to_bram_converter_A_dout1;
  B_ce0 <= mem_to_bram_converter_B_ce0;
  B_we0 <= mem_to_bram_converter_B_we0;
  B_address0 <= mem_to_bram_converter_B_address0;
  B_dout0 <= mem_to_bram_converter_B_dout0;
  B_ce1 <= mem_to_bram_converter_B_ce1;
  B_we1 <= mem_to_bram_converter_B_we1;
  B_address1 <= mem_to_bram_converter_B_address1;
  B_dout1 <= mem_to_bram_converter_B_dout1;
  C_ce0 <= mem_to_bram_converter_C_ce0;
  C_we0 <= mem_to_bram_converter_C_we0;
  C_address0 <= mem_to_bram_converter_C_address0;
  C_dout0 <= mem_to_bram_converter_C_dout0;
  C_ce1 <= mem_to_bram_converter_C_ce1;
  C_we1 <= mem_to_bram_converter_C_we1;
  C_address1 <= mem_to_bram_converter_C_address1;
  C_dout1 <= mem_to_bram_converter_C_dout1;
  D_ce0 <= mem_to_bram_converter_D_ce0;
  D_we0 <= mem_to_bram_converter_D_we0;
  D_address0 <= mem_to_bram_converter_D_address0;
  D_dout0 <= mem_to_bram_converter_D_dout0;
  D_ce1 <= mem_to_bram_converter_D_ce1;
  D_we1 <= mem_to_bram_converter_D_we1;
  D_address1 <= mem_to_bram_converter_D_address1;
  D_dout1 <= mem_to_bram_converter_D_dout1;

  mem_to_bram_converter_C : entity work.mem_to_bram(arch) generic map(32, 7)
    port map(
      loadEn => kernel_2mm_wrapped_C_loadEn,
      loadAddr => kernel_2mm_wrapped_C_loadAddr,
      storeEn => kernel_2mm_wrapped_C_storeEn,
      storeAddr => kernel_2mm_wrapped_C_storeAddr,
      storeData => kernel_2mm_wrapped_C_storeData,
      din0 => C_din0,
      din1 => C_din1,
      ce0 => mem_to_bram_converter_C_ce0,
      we0 => mem_to_bram_converter_C_we0,
      address0 => mem_to_bram_converter_C_address0,
      dout0 => mem_to_bram_converter_C_dout0,
      ce1 => mem_to_bram_converter_C_ce1,
      we1 => mem_to_bram_converter_C_we1,
      address1 => mem_to_bram_converter_C_address1,
      dout1 => mem_to_bram_converter_C_dout1,
      loadData => mem_to_bram_converter_C_loadData
    );

  mem_to_bram_converter_tmp : entity work.mem_to_bram(arch) generic map(32, 7)
    port map(
      loadEn => kernel_2mm_wrapped_tmp_loadEn,
      loadAddr => kernel_2mm_wrapped_tmp_loadAddr,
      storeEn => kernel_2mm_wrapped_tmp_storeEn,
      storeAddr => kernel_2mm_wrapped_tmp_storeAddr,
      storeData => kernel_2mm_wrapped_tmp_storeData,
      din0 => tmp_din0,
      din1 => tmp_din1,
      ce0 => mem_to_bram_converter_tmp_ce0,
      we0 => mem_to_bram_converter_tmp_we0,
      address0 => mem_to_bram_converter_tmp_address0,
      dout0 => mem_to_bram_converter_tmp_dout0,
      ce1 => mem_to_bram_converter_tmp_ce1,
      we1 => mem_to_bram_converter_tmp_we1,
      address1 => mem_to_bram_converter_tmp_address1,
      dout1 => mem_to_bram_converter_tmp_dout1,
      loadData => mem_to_bram_converter_tmp_loadData
    );

  mem_to_bram_converter_D : entity work.mem_to_bram(arch) generic map(32, 7)
    port map(
      loadEn => kernel_2mm_wrapped_D_loadEn,
      loadAddr => kernel_2mm_wrapped_D_loadAddr,
      storeEn => kernel_2mm_wrapped_D_storeEn,
      storeAddr => kernel_2mm_wrapped_D_storeAddr,
      storeData => kernel_2mm_wrapped_D_storeData,
      din0 => D_din0,
      din1 => D_din1,
      ce0 => mem_to_bram_converter_D_ce0,
      we0 => mem_to_bram_converter_D_we0,
      address0 => mem_to_bram_converter_D_address0,
      dout0 => mem_to_bram_converter_D_dout0,
      ce1 => mem_to_bram_converter_D_ce1,
      we1 => mem_to_bram_converter_D_we1,
      address1 => mem_to_bram_converter_D_address1,
      dout1 => mem_to_bram_converter_D_dout1,
      loadData => mem_to_bram_converter_D_loadData
    );

  mem_to_bram_converter_B : entity work.mem_to_bram(arch) generic map(32, 7)
    port map(
      loadEn => kernel_2mm_wrapped_B_loadEn,
      loadAddr => kernel_2mm_wrapped_B_loadAddr,
      storeEn => kernel_2mm_wrapped_B_storeEn,
      storeAddr => kernel_2mm_wrapped_B_storeAddr,
      storeData => kernel_2mm_wrapped_B_storeData,
      din0 => B_din0,
      din1 => B_din1,
      ce0 => mem_to_bram_converter_B_ce0,
      we0 => mem_to_bram_converter_B_we0,
      address0 => mem_to_bram_converter_B_address0,
      dout0 => mem_to_bram_converter_B_dout0,
      ce1 => mem_to_bram_converter_B_ce1,
      we1 => mem_to_bram_converter_B_we1,
      address1 => mem_to_bram_converter_B_address1,
      dout1 => mem_to_bram_converter_B_dout1,
      loadData => mem_to_bram_converter_B_loadData
    );

  mem_to_bram_converter_A : entity work.mem_to_bram(arch) generic map(32, 7)
    port map(
      loadEn => kernel_2mm_wrapped_A_loadEn,
      loadAddr => kernel_2mm_wrapped_A_loadAddr,
      storeEn => kernel_2mm_wrapped_A_storeEn,
      storeAddr => kernel_2mm_wrapped_A_storeAddr,
      storeData => kernel_2mm_wrapped_A_storeData,
      din0 => A_din0,
      din1 => A_din1,
      ce0 => mem_to_bram_converter_A_ce0,
      we0 => mem_to_bram_converter_A_we0,
      address0 => mem_to_bram_converter_A_address0,
      dout0 => mem_to_bram_converter_A_dout0,
      ce1 => mem_to_bram_converter_A_ce1,
      we1 => mem_to_bram_converter_A_we1,
      address1 => mem_to_bram_converter_A_address1,
      dout1 => mem_to_bram_converter_A_dout1,
      loadData => mem_to_bram_converter_A_loadData
    );

  kernel_2mm_wrapped : entity work.kernel_2mm(behavioral)
    port map(
      alpha => alpha,
      alpha_valid => alpha_valid,
      alpha_ready => alpha_ready,
      beta => beta,
      beta_valid => beta_valid,
      beta_ready => beta_ready,
      tmp_loadData => mem_to_bram_converter_tmp_loadData,
      A_loadData => mem_to_bram_converter_A_loadData,
      B_loadData => mem_to_bram_converter_B_loadData,
      C_loadData => mem_to_bram_converter_C_loadData,
      D_loadData => mem_to_bram_converter_D_loadData,
      tmp_start_valid => tmp_start_valid,
      tmp_start_ready => tmp_start_ready,
      A_start_valid => A_start_valid,
      A_start_ready => A_start_ready,
      B_start_valid => B_start_valid,
      B_start_ready => B_start_ready,
      C_start_valid => C_start_valid,
      C_start_ready => C_start_ready,
      D_start_valid => D_start_valid,
      D_start_ready => D_start_ready,
      start_valid => start_valid,
      start_ready => start_ready,
      clk => clk,
      rst => rst,
      tmp_end_valid => kernel_2mm_wrapped_tmp_end_valid,
      tmp_end_ready => kernel_2mm_wrapped_tmp_end_ready,
      A_end_valid => kernel_2mm_wrapped_A_end_valid,
      A_end_ready => kernel_2mm_wrapped_A_end_ready,
      B_end_valid => kernel_2mm_wrapped_B_end_valid,
      B_end_ready => kernel_2mm_wrapped_B_end_ready,
      C_end_valid => kernel_2mm_wrapped_C_end_valid,
      C_end_ready => kernel_2mm_wrapped_C_end_ready,
      D_end_valid => kernel_2mm_wrapped_D_end_valid,
      D_end_ready => kernel_2mm_wrapped_D_end_ready,
      end_valid => kernel_2mm_wrapped_end_valid,
      end_ready => kernel_2mm_wrapped_end_ready,
      tmp_loadEn => kernel_2mm_wrapped_tmp_loadEn,
      tmp_loadAddr => kernel_2mm_wrapped_tmp_loadAddr,
      tmp_storeEn => kernel_2mm_wrapped_tmp_storeEn,
      tmp_storeAddr => kernel_2mm_wrapped_tmp_storeAddr,
      tmp_storeData => kernel_2mm_wrapped_tmp_storeData,
      A_loadEn => kernel_2mm_wrapped_A_loadEn,
      A_loadAddr => kernel_2mm_wrapped_A_loadAddr,
      A_storeEn => kernel_2mm_wrapped_A_storeEn,
      A_storeAddr => kernel_2mm_wrapped_A_storeAddr,
      A_storeData => kernel_2mm_wrapped_A_storeData,
      B_loadEn => kernel_2mm_wrapped_B_loadEn,
      B_loadAddr => kernel_2mm_wrapped_B_loadAddr,
      B_storeEn => kernel_2mm_wrapped_B_storeEn,
      B_storeAddr => kernel_2mm_wrapped_B_storeAddr,
      B_storeData => kernel_2mm_wrapped_B_storeData,
      C_loadEn => kernel_2mm_wrapped_C_loadEn,
      C_loadAddr => kernel_2mm_wrapped_C_loadAddr,
      C_storeEn => kernel_2mm_wrapped_C_storeEn,
      C_storeAddr => kernel_2mm_wrapped_C_storeAddr,
      C_storeData => kernel_2mm_wrapped_C_storeData,
      D_loadEn => kernel_2mm_wrapped_D_loadEn,
      D_loadAddr => kernel_2mm_wrapped_D_loadAddr,
      D_storeEn => kernel_2mm_wrapped_D_storeEn,
      D_storeAddr => kernel_2mm_wrapped_D_storeAddr,
      D_storeData => kernel_2mm_wrapped_D_storeData
    );

end architecture;
