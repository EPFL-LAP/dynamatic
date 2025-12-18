
library IEEE;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
use ieee.std_logic_textio.all;
use ieee.numeric_std.all;
use std.textio.all;
use work.sim_package.all;
entity tb is
end entity tb;

architecture behavior of tb is

  constant INPUT_n : STRING := "/beta/rouzbeh/hope/dynamatic/skip/runs/run_2025-12-16_11-47-44/histogram-8/out/sim/INPUT_VECTORS/input_n.dat";
  constant OUTPUT_n : STRING := "/beta/rouzbeh/hope/dynamatic/skip/runs/run_2025-12-16_11-47-44/histogram-8/out/sim/HDL_OUT/output_n.dat";
  constant DATA_WIDTH_n : INTEGER := 32;
  constant INPUT_feature : STRING := "/beta/rouzbeh/hope/dynamatic/skip/runs/run_2025-12-16_11-47-44/histogram-8/out/sim/INPUT_VECTORS/input_feature.dat";
  constant OUTPUT_feature : STRING := "/beta/rouzbeh/hope/dynamatic/skip/runs/run_2025-12-16_11-47-44/histogram-8/out/sim/HDL_OUT/output_feature.dat";
  constant DATA_WIDTH_feature : INTEGER := 32;
  constant ADDR_WIDTH_feature : INTEGER := 10;
  constant DATA_DEPTH_feature : INTEGER := 1000;
  constant INPUT_weight : STRING := "/beta/rouzbeh/hope/dynamatic/skip/runs/run_2025-12-16_11-47-44/histogram-8/out/sim/INPUT_VECTORS/input_weight.dat";
  constant OUTPUT_weight : STRING := "/beta/rouzbeh/hope/dynamatic/skip/runs/run_2025-12-16_11-47-44/histogram-8/out/sim/HDL_OUT/output_weight.dat";
  constant DATA_WIDTH_weight : INTEGER := 32;
  constant ADDR_WIDTH_weight : INTEGER := 10;
  constant DATA_DEPTH_weight : INTEGER := 1000;
  constant INPUT_hist : STRING := "/beta/rouzbeh/hope/dynamatic/skip/runs/run_2025-12-16_11-47-44/histogram-8/out/sim/INPUT_VECTORS/input_hist.dat";
  constant OUTPUT_hist : STRING := "/beta/rouzbeh/hope/dynamatic/skip/runs/run_2025-12-16_11-47-44/histogram-8/out/sim/HDL_OUT/output_hist.dat";
  constant DATA_WIDTH_hist : INTEGER := 32;
  constant ADDR_WIDTH_hist : INTEGER := 10;
  constant DATA_DEPTH_hist : INTEGER := 1000;
  constant HALF_CLK_PERIOD : TIME := 2.00 ns;
  constant RESET_LATENCY : TIME := 10.00 ns;
  constant TRANSACTION_NUM : INTEGER := 1;
  signal tb_clk : std_logic := '0';
  signal tb_rst : std_logic := '0';
  signal tb_start_valid : std_logic := '0';
  signal tb_start_ready : std_logic := '0';
  signal tb_started : std_logic;
  signal tb_global_valid : std_logic;
  signal tb_global_ready : std_logic;
  signal tb_stop : std_logic;
  signal n_ce0 : std_logic;
  signal n_we0 : std_logic;
  signal n_din0 : std_logic_vector(32 - 1 downto 0);
  signal n_dout0 : std_logic_vector(32 - 1 downto 0);
  signal n_dout0_valid : std_logic;
  signal n_dout0_ready : std_logic;
  signal feature_we0 : std_logic;
  signal feature_din0 : std_logic_vector(32 - 1 downto 0);
  signal feature_address0 : std_logic_vector(10 - 1 downto 0);
  signal feature_ce1 : std_logic;
  signal feature_dout1 : std_logic_vector(32 - 1 downto 0);
  signal feature_address1 : std_logic_vector(10 - 1 downto 0);
  signal feature_dout0 : std_logic_vector(32 - 1 downto 0);
  signal feature_din1 : std_logic_vector(32 - 1 downto 0) := (others => '0');
  signal feature_we1 : std_logic := '0';
  signal feature_ce0 : std_logic := '1';
  signal weight_we0 : std_logic;
  signal weight_din0 : std_logic_vector(32 - 1 downto 0);
  signal weight_address0 : std_logic_vector(10 - 1 downto 0);
  signal weight_ce1 : std_logic;
  signal weight_dout1 : std_logic_vector(32 - 1 downto 0);
  signal weight_address1 : std_logic_vector(10 - 1 downto 0);
  signal weight_dout0 : std_logic_vector(32 - 1 downto 0);
  signal weight_din1 : std_logic_vector(32 - 1 downto 0) := (others => '0');
  signal weight_we1 : std_logic := '0';
  signal weight_ce0 : std_logic := '1';
  signal hist_we0 : std_logic;
  signal hist_din0 : std_logic_vector(32 - 1 downto 0);
  signal hist_address0 : std_logic_vector(10 - 1 downto 0);
  signal hist_ce1 : std_logic;
  signal hist_dout1 : std_logic_vector(32 - 1 downto 0);
  signal hist_address1 : std_logic_vector(10 - 1 downto 0);
  signal hist_dout0 : std_logic_vector(32 - 1 downto 0);
  signal hist_din1 : std_logic_vector(32 - 1 downto 0) := (others => '0');
  signal hist_we1 : std_logic := '0';
  signal hist_ce0 : std_logic := '1';
  signal feature_end_valid : std_logic;
  signal feature_end_ready : std_logic;
  signal weight_end_valid : std_logic;
  signal weight_end_ready : std_logic;
  signal hist_end_valid : std_logic;
  signal hist_end_ready : std_logic;
  signal end_valid : std_logic;
  signal end_ready : std_logic;

  signal tb_temp_idle : std_logic := '1';
  shared variable transaction_idx : INTEGER := 0;
begin

  duv_inst: entity work.histogram
  port map(
    clk => tb_clk,
    end_ready => end_ready,
    end_valid => end_valid,
    feature_end_ready => feature_end_ready,
    feature_end_valid => feature_end_valid,
    feature_loadAddr => feature_address1,
    feature_loadData => feature_dout1,
    feature_loadEn => feature_ce1,
    feature_start_ready => open,
    feature_start_valid => '1',
    feature_storeAddr => feature_address0,
    feature_storeData => feature_din0,
    feature_storeEn => feature_we0,
    hist_end_ready => hist_end_ready,
    hist_end_valid => hist_end_valid,
    hist_loadAddr => hist_address1,
    hist_loadData => hist_dout1,
    hist_loadEn => hist_ce1,
    hist_start_ready => open,
    hist_start_valid => '1',
    hist_storeAddr => hist_address0,
    hist_storeData => hist_din0,
    hist_storeEn => hist_we0,
    n => n_dout0,
    n_ready => n_dout0_ready,
    n_valid => n_dout0_valid,
    rst => tb_rst,
    start_ready => open,
    start_valid => '1',
    weight_end_ready => weight_end_ready,
    weight_end_valid => weight_end_valid,
    weight_loadAddr => weight_address1,
    weight_loadData => weight_dout1,
    weight_loadEn => weight_ce1,
    weight_start_ready => open,
    weight_start_valid => '1',
    weight_storeAddr => weight_address0,
    weight_storeData => weight_din0,
    weight_storeEn => weight_we0
  );

  arg_inst_n: entity work.single_argument
  generic map(
    TV_IN => INPUT_n,
    TV_OUT => OUTPUT_n,
    DATA_WIDTH => DATA_WIDTH_n
  )
  port map(
    ce0 => '1',
    clk => tb_clk,
    din0 => (others => '0'),
    done => tb_temp_idle,
    dout0 => n_dout0,
    dout0_ready => n_dout0_ready,
    dout0_valid => n_dout0_valid,
    rst => tb_rst,
    we0 => '0'
  );

  mem_inst_feature: entity work.two_port_RAM
  generic map(
    TV_IN => INPUT_feature,
    TV_OUT => OUTPUT_feature,
    DATA_WIDTH => DATA_WIDTH_feature,
    ADDR_WIDTH => ADDR_WIDTH_feature,
    DEPTH => DATA_DEPTH_feature
  )
  port map(
    address0 => feature_address0,
    address1 => feature_address1,
    ce0 => '1',
    ce1 => feature_ce1,
    clk => tb_clk,
    din0 => feature_din0,
    din1 => feature_din1,
    done => tb_stop,
    dout0 => feature_dout0,
    dout1 => feature_dout1,
    rst => tb_rst,
    we0 => feature_we0,
    we1 => feature_we1
  );

  mem_inst_weight: entity work.two_port_RAM
  generic map(
    TV_IN => INPUT_weight,
    TV_OUT => OUTPUT_weight,
    DATA_WIDTH => DATA_WIDTH_weight,
    ADDR_WIDTH => ADDR_WIDTH_weight,
    DEPTH => DATA_DEPTH_weight
  )
  port map(
    address0 => weight_address0,
    address1 => weight_address1,
    ce0 => '1',
    ce1 => weight_ce1,
    clk => tb_clk,
    din0 => weight_din0,
    din1 => weight_din1,
    done => tb_stop,
    dout0 => weight_dout0,
    dout1 => weight_dout1,
    rst => tb_rst,
    we0 => weight_we0,
    we1 => weight_we1
  );

  mem_inst_hist: entity work.two_port_RAM
  generic map(
    TV_IN => INPUT_hist,
    TV_OUT => OUTPUT_hist,
    DATA_WIDTH => DATA_WIDTH_hist,
    ADDR_WIDTH => ADDR_WIDTH_hist,
    DEPTH => DATA_DEPTH_hist
  )
  port map(
    address0 => hist_address0,
    address1 => hist_address1,
    ce0 => '1',
    ce1 => hist_ce1,
    clk => tb_clk,
    din0 => hist_din0,
    din1 => hist_din1,
    done => tb_stop,
    dout0 => hist_dout0,
    dout1 => hist_dout1,
    rst => tb_rst,
    we0 => hist_we0,
    we1 => hist_we1
  );

  join_valids: entity work.tb_join
  generic map(
    SIZE => 4
  )
  port map(
    ins_ready(0) => feature_end_ready,
    ins_ready(1) => weight_end_ready,
    ins_ready(2) => hist_end_ready,
    ins_ready(3) => end_ready,
    ins_valid(0) => feature_end_valid,
    ins_valid(1) => weight_end_valid,
    ins_valid(2) => hist_end_valid,
    ins_valid(3) => end_valid,
    outs_ready => tb_global_ready,
    outs_valid => tb_global_valid
  );


  write_output_transactor_feature_runtime_proc : process
    file fp             : TEXT;
    variable fstatus    : FILE_OPEN_STATUS;
    variable token_line : LINE;
    variable token      : STRING(1 to 1024);
  begin
    file_open(fstatus, fp, OUTPUT_feature , WRITE_MODE);
    if (fstatus /= OPEN_OK) then
      assert false report "Open file " & OUTPUT_feature & " failed!!!" severity note;
      assert false report "ERROR: Simulation using HLS TB failed." severity failure;
    end if;
    write(token_line, string'("[[[runtime]]]"));
    writeline(fp, token_line);
    file_close(fp);
    while transaction_idx /= TRANSACTION_NUM loop
      wait until tb_clk'event and tb_clk = '1';
    end loop;
    wait until tb_clk'event and tb_clk = '1';
    wait until tb_clk'event and tb_clk = '1';
    file_open(fstatus, fp, OUTPUT_feature, APPEND_MODE);
    if (fstatus /= OPEN_OK) then
      assert false report "Open file " & OUTPUT_feature & " failed!!!" severity note;
      assert false report "ERROR: Simulation using HLS TB failed." severity failure;
    end if;
    write(token_line, string'("[[[/runtime]]]"));
    writeline(fp, token_line);
    file_close(fp);
    wait;
  end process;

  write_output_transactor_weight_runtime_proc : process
    file fp             : TEXT;
    variable fstatus    : FILE_OPEN_STATUS;
    variable token_line : LINE;
    variable token      : STRING(1 to 1024);
  begin
    file_open(fstatus, fp, OUTPUT_weight , WRITE_MODE);
    if (fstatus /= OPEN_OK) then
      assert false report "Open file " & OUTPUT_weight & " failed!!!" severity note;
      assert false report "ERROR: Simulation using HLS TB failed." severity failure;
    end if;
    write(token_line, string'("[[[runtime]]]"));
    writeline(fp, token_line);
    file_close(fp);
    while transaction_idx /= TRANSACTION_NUM loop
      wait until tb_clk'event and tb_clk = '1';
    end loop;
    wait until tb_clk'event and tb_clk = '1';
    wait until tb_clk'event and tb_clk = '1';
    file_open(fstatus, fp, OUTPUT_weight, APPEND_MODE);
    if (fstatus /= OPEN_OK) then
      assert false report "Open file " & OUTPUT_weight & " failed!!!" severity note;
      assert false report "ERROR: Simulation using HLS TB failed." severity failure;
    end if;
    write(token_line, string'("[[[/runtime]]]"));
    writeline(fp, token_line);
    file_close(fp);
    wait;
  end process;

  write_output_transactor_hist_runtime_proc : process
    file fp             : TEXT;
    variable fstatus    : FILE_OPEN_STATUS;
    variable token_line : LINE;
    variable token      : STRING(1 to 1024);
  begin
    file_open(fstatus, fp, OUTPUT_hist , WRITE_MODE);
    if (fstatus /= OPEN_OK) then
      assert false report "Open file " & OUTPUT_hist & " failed!!!" severity note;
      assert false report "ERROR: Simulation using HLS TB failed." severity failure;
    end if;
    write(token_line, string'("[[[runtime]]]"));
    writeline(fp, token_line);
    file_close(fp);
    while transaction_idx /= TRANSACTION_NUM loop
      wait until tb_clk'event and tb_clk = '1';
    end loop;
    wait until tb_clk'event and tb_clk = '1';
    wait until tb_clk'event and tb_clk = '1';
    file_open(fstatus, fp, OUTPUT_hist, APPEND_MODE);
    if (fstatus /= OPEN_OK) then
      assert false report "Open file " & OUTPUT_hist & " failed!!!" severity note;
      assert false report "ERROR: Simulation using HLS TB failed." severity failure;
    end if;
    write(token_line, string'("[[[/runtime]]]"));
    writeline(fp, token_line);
    file_close(fp);
    wait;
  end process;

  write_output_transactor_n_runtime_proc : process
    file fp             : TEXT;
    variable fstatus    : FILE_OPEN_STATUS;
    variable token_line : LINE;
    variable token      : STRING(1 to 1024);
  begin
    file_open(fstatus, fp, OUTPUT_n , WRITE_MODE);
    if (fstatus /= OPEN_OK) then
      assert false report "Open file " & OUTPUT_n & " failed!!!" severity note;
      assert false report "ERROR: Simulation using HLS TB failed." severity failure;
    end if;
    write(token_line, string'("[[[runtime]]]"));
    writeline(fp, token_line);
    file_close(fp);
    while transaction_idx /= TRANSACTION_NUM loop
      wait until tb_clk'event and tb_clk = '1';
    end loop;
    wait until tb_clk'event and tb_clk = '1';
    wait until tb_clk'event and tb_clk = '1';
    file_open(fstatus, fp, OUTPUT_n, APPEND_MODE);
    if (fstatus /= OPEN_OK) then
      assert false report "Open file " & OUTPUT_n & " failed!!!" severity note;
      assert false report "ERROR: Simulation using HLS TB failed." severity failure;
    end if;
    write(token_line, string'("[[[/runtime]]]"));
    writeline(fp, token_line);
    file_close(fp);
    wait;
  end process;


  generate_sim_done_proc : process
  begin
    while (transaction_idx /= TRANSACTION_NUM) loop
      wait until tb_clk'event and tb_clk = '1';
    end loop;
    wait until tb_clk'event and tb_clk = '1';
    wait until tb_clk'event and tb_clk = '1';
    wait until tb_clk'event and tb_clk = '1';
    assert false
    report "Simulation done! Latency = " & integer'image((now - RESET_LATENCY) / (2 * HALF_CLK_PERIOD)) & " cycles"
    severity note;
    assert false
    report "NORMAL EXIT (note: failure is to force the simulator to stop)"
    severity failure;
    wait;
  end process;

  gen_clock_proc : process
  begin
    tb_clk <= '0';
    while (true) loop
      wait for HALF_CLK_PERIOD;
      tb_clk <= not tb_clk;
    end loop;
    wait;
  end process;

  gen_reset_proc : process
  begin
    tb_rst <= '1';
    wait for RESET_LATENCY;
    tb_rst <= '0';
    wait;
  end process;

  acknowledge_tb_end: process(tb_clk,tb_rst)
  begin
    if (tb_rst = '1') then
      tb_global_ready <= '1';
      tb_stop <= '0';
    elsif rising_edge(tb_clk) then
      if (tb_global_valid = '1') then
        tb_global_ready <= '0';
        tb_stop <= '1';
      end if;
    end if;
  end process;

  generate_idle_signal: process(tb_clk,tb_rst)
  begin
    if (tb_rst = '1') then
      tb_temp_idle <= '1';
    elsif rising_edge(tb_clk) then
      tb_temp_idle <= tb_temp_idle;
      if (tb_start_valid = '1') then
        tb_temp_idle <= '0';
      end if;
      if(tb_stop = '1') then
        tb_temp_idle <= '1';
      end if;
    end if;
  end process generate_idle_signal;

  generate_start_signal : process(tb_clk, tb_rst)
  begin
    if (tb_rst = '1') then
      tb_start_valid <= '0';
      tb_started <= '0';
    elsif rising_edge(tb_clk) then
      if (tb_started = '0') then
        tb_start_valid <= '1';
        tb_started <= '1';
      else
        tb_start_valid <= tb_start_valid and (not tb_start_ready);
      end if;
    end if;
  end process generate_start_signal;

  transaction_increment : process
  begin
    wait until tb_rst = '0';
    while (tb_temp_idle /= '1') loop
      wait until tb_clk'event and tb_clk = '1';
    end loop;
    wait until tb_temp_idle = '0';
    while (true) loop
      while (tb_temp_idle /= '1') loop
        wait until tb_clk'event and tb_clk = '1';
      end loop;
      transaction_idx := transaction_idx + 1;
      wait until tb_temp_idle = '0';
    end loop;
  end process;
end architecture behavior;
