
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

  constant INPUT_a : STRING := "/beta/rouzbeh/clean/dynamatic/skip2/runs/run_2025-12-16_15-06-50/bicg-5-6/out/sim/INPUT_VECTORS/input_a.dat";
  constant OUTPUT_a : STRING := "/beta/rouzbeh/clean/dynamatic/skip2/runs/run_2025-12-16_15-06-50/bicg-5-6/out/sim/HDL_OUT/output_a.dat";
  constant DATA_WIDTH_a : INTEGER := 32;
  constant ADDR_WIDTH_a : INTEGER := 10;
  constant DATA_DEPTH_a : INTEGER := 900;
  constant INPUT_s : STRING := "/beta/rouzbeh/clean/dynamatic/skip2/runs/run_2025-12-16_15-06-50/bicg-5-6/out/sim/INPUT_VECTORS/input_s.dat";
  constant OUTPUT_s : STRING := "/beta/rouzbeh/clean/dynamatic/skip2/runs/run_2025-12-16_15-06-50/bicg-5-6/out/sim/HDL_OUT/output_s.dat";
  constant DATA_WIDTH_s : INTEGER := 32;
  constant ADDR_WIDTH_s : INTEGER := 5;
  constant DATA_DEPTH_s : INTEGER := 30;
  constant INPUT_q : STRING := "/beta/rouzbeh/clean/dynamatic/skip2/runs/run_2025-12-16_15-06-50/bicg-5-6/out/sim/INPUT_VECTORS/input_q.dat";
  constant OUTPUT_q : STRING := "/beta/rouzbeh/clean/dynamatic/skip2/runs/run_2025-12-16_15-06-50/bicg-5-6/out/sim/HDL_OUT/output_q.dat";
  constant DATA_WIDTH_q : INTEGER := 32;
  constant ADDR_WIDTH_q : INTEGER := 5;
  constant DATA_DEPTH_q : INTEGER := 30;
  constant INPUT_p : STRING := "/beta/rouzbeh/clean/dynamatic/skip2/runs/run_2025-12-16_15-06-50/bicg-5-6/out/sim/INPUT_VECTORS/input_p.dat";
  constant OUTPUT_p : STRING := "/beta/rouzbeh/clean/dynamatic/skip2/runs/run_2025-12-16_15-06-50/bicg-5-6/out/sim/HDL_OUT/output_p.dat";
  constant DATA_WIDTH_p : INTEGER := 32;
  constant ADDR_WIDTH_p : INTEGER := 5;
  constant DATA_DEPTH_p : INTEGER := 30;
  constant INPUT_r : STRING := "/beta/rouzbeh/clean/dynamatic/skip2/runs/run_2025-12-16_15-06-50/bicg-5-6/out/sim/INPUT_VECTORS/input_r.dat";
  constant OUTPUT_r : STRING := "/beta/rouzbeh/clean/dynamatic/skip2/runs/run_2025-12-16_15-06-50/bicg-5-6/out/sim/HDL_OUT/output_r.dat";
  constant DATA_WIDTH_r : INTEGER := 32;
  constant ADDR_WIDTH_r : INTEGER := 5;
  constant DATA_DEPTH_r : INTEGER := 30;
  constant INPUT_out0 : STRING := "";
  constant OUTPUT_out0 : STRING := "/beta/rouzbeh/clean/dynamatic/skip2/runs/run_2025-12-16_15-06-50/bicg-5-6/out/sim/HDL_OUT/output_out0.dat";
  constant DATA_WIDTH_out0 : INTEGER := 32;
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
  signal a_we0 : std_logic;
  signal a_din0 : std_logic_vector(32 - 1 downto 0);
  signal a_address0 : std_logic_vector(10 - 1 downto 0);
  signal a_ce1 : std_logic;
  signal a_dout1 : std_logic_vector(32 - 1 downto 0);
  signal a_address1 : std_logic_vector(10 - 1 downto 0);
  signal a_dout0 : std_logic_vector(32 - 1 downto 0);
  signal a_din1 : std_logic_vector(32 - 1 downto 0) := (others => '0');
  signal a_we1 : std_logic := '0';
  signal a_ce0 : std_logic := '1';
  signal s_we0 : std_logic;
  signal s_din0 : std_logic_vector(32 - 1 downto 0);
  signal s_address0 : std_logic_vector(5 - 1 downto 0);
  signal s_ce1 : std_logic;
  signal s_dout1 : std_logic_vector(32 - 1 downto 0);
  signal s_address1 : std_logic_vector(5 - 1 downto 0);
  signal s_dout0 : std_logic_vector(32 - 1 downto 0);
  signal s_din1 : std_logic_vector(32 - 1 downto 0) := (others => '0');
  signal s_we1 : std_logic := '0';
  signal s_ce0 : std_logic := '1';
  signal q_we0 : std_logic;
  signal q_din0 : std_logic_vector(32 - 1 downto 0);
  signal q_address0 : std_logic_vector(5 - 1 downto 0);
  signal q_ce1 : std_logic;
  signal q_dout1 : std_logic_vector(32 - 1 downto 0);
  signal q_address1 : std_logic_vector(5 - 1 downto 0);
  signal q_dout0 : std_logic_vector(32 - 1 downto 0);
  signal q_din1 : std_logic_vector(32 - 1 downto 0) := (others => '0');
  signal q_we1 : std_logic := '0';
  signal q_ce0 : std_logic := '1';
  signal p_we0 : std_logic;
  signal p_din0 : std_logic_vector(32 - 1 downto 0);
  signal p_address0 : std_logic_vector(5 - 1 downto 0);
  signal p_ce1 : std_logic;
  signal p_dout1 : std_logic_vector(32 - 1 downto 0);
  signal p_address1 : std_logic_vector(5 - 1 downto 0);
  signal p_dout0 : std_logic_vector(32 - 1 downto 0);
  signal p_din1 : std_logic_vector(32 - 1 downto 0) := (others => '0');
  signal p_we1 : std_logic := '0';
  signal p_ce0 : std_logic := '1';
  signal r_we0 : std_logic;
  signal r_din0 : std_logic_vector(32 - 1 downto 0);
  signal r_address0 : std_logic_vector(5 - 1 downto 0);
  signal r_ce1 : std_logic;
  signal r_dout1 : std_logic_vector(32 - 1 downto 0);
  signal r_address1 : std_logic_vector(5 - 1 downto 0);
  signal r_dout0 : std_logic_vector(32 - 1 downto 0);
  signal r_din1 : std_logic_vector(32 - 1 downto 0) := (others => '0');
  signal r_we1 : std_logic := '0';
  signal r_ce0 : std_logic := '1';
  signal out0_valid : std_logic;
  signal out0_ready : std_logic;
  signal out0_ce0 : std_logic;
  signal out0_we0 : std_logic;
  signal out0_din0 : std_logic_vector(32 - 1 downto 0);
  signal out0_dout0 : std_logic_vector(32 - 1 downto 0);
  signal out0_dout0_valid : std_logic;
  signal out0_dout0_ready : std_logic;
  signal a_end_valid : std_logic;
  signal a_end_ready : std_logic;
  signal s_end_valid : std_logic;
  signal s_end_ready : std_logic;
  signal q_end_valid : std_logic;
  signal q_end_ready : std_logic;
  signal p_end_valid : std_logic;
  signal p_end_ready : std_logic;
  signal r_end_valid : std_logic;
  signal r_end_ready : std_logic;
  signal end_valid : std_logic;
  signal end_ready : std_logic;

  signal tb_temp_idle : std_logic := '1';
  shared variable transaction_idx : INTEGER := 0;
begin

  duv_inst: entity work.bicg
  port map(
    a_end_ready => a_end_ready,
    a_end_valid => a_end_valid,
    a_loadAddr => a_address1,
    a_loadData => a_dout1,
    a_loadEn => a_ce1,
    a_start_ready => open,
    a_start_valid => '1',
    a_storeAddr => a_address0,
    a_storeData => a_din0,
    a_storeEn => a_we0,
    clk => tb_clk,
    end_ready => end_ready,
    end_valid => end_valid,
    out0 => out0_din0,
    out0_ready => out0_ready,
    out0_valid => out0_valid,
    p_end_ready => p_end_ready,
    p_end_valid => p_end_valid,
    p_loadAddr => p_address1,
    p_loadData => p_dout1,
    p_loadEn => p_ce1,
    p_start_ready => open,
    p_start_valid => '1',
    p_storeAddr => p_address0,
    p_storeData => p_din0,
    p_storeEn => p_we0,
    q_end_ready => q_end_ready,
    q_end_valid => q_end_valid,
    q_loadAddr => q_address1,
    q_loadData => q_dout1,
    q_loadEn => q_ce1,
    q_start_ready => open,
    q_start_valid => '1',
    q_storeAddr => q_address0,
    q_storeData => q_din0,
    q_storeEn => q_we0,
    r_end_ready => r_end_ready,
    r_end_valid => r_end_valid,
    r_loadAddr => r_address1,
    r_loadData => r_dout1,
    r_loadEn => r_ce1,
    r_start_ready => open,
    r_start_valid => '1',
    r_storeAddr => r_address0,
    r_storeData => r_din0,
    r_storeEn => r_we0,
    rst => tb_rst,
    s_end_ready => s_end_ready,
    s_end_valid => s_end_valid,
    s_loadAddr => s_address1,
    s_loadData => s_dout1,
    s_loadEn => s_ce1,
    s_start_ready => open,
    s_start_valid => '1',
    s_storeAddr => s_address0,
    s_storeData => s_din0,
    s_storeEn => s_we0,
    start_ready => open,
    start_valid => '1'
  );

  mem_inst_a: entity work.two_port_RAM
  generic map(
    TV_IN => INPUT_a,
    TV_OUT => OUTPUT_a,
    DATA_WIDTH => DATA_WIDTH_a,
    ADDR_WIDTH => ADDR_WIDTH_a,
    DEPTH => DATA_DEPTH_a
  )
  port map(
    address0 => a_address0,
    address1 => a_address1,
    ce0 => '1',
    ce1 => a_ce1,
    clk => tb_clk,
    din0 => a_din0,
    din1 => a_din1,
    done => tb_stop,
    dout0 => a_dout0,
    dout1 => a_dout1,
    rst => tb_rst,
    we0 => a_we0,
    we1 => a_we1
  );

  mem_inst_s: entity work.two_port_RAM
  generic map(
    TV_IN => INPUT_s,
    TV_OUT => OUTPUT_s,
    DATA_WIDTH => DATA_WIDTH_s,
    ADDR_WIDTH => ADDR_WIDTH_s,
    DEPTH => DATA_DEPTH_s
  )
  port map(
    address0 => s_address0,
    address1 => s_address1,
    ce0 => '1',
    ce1 => s_ce1,
    clk => tb_clk,
    din0 => s_din0,
    din1 => s_din1,
    done => tb_stop,
    dout0 => s_dout0,
    dout1 => s_dout1,
    rst => tb_rst,
    we0 => s_we0,
    we1 => s_we1
  );

  mem_inst_q: entity work.two_port_RAM
  generic map(
    TV_IN => INPUT_q,
    TV_OUT => OUTPUT_q,
    DATA_WIDTH => DATA_WIDTH_q,
    ADDR_WIDTH => ADDR_WIDTH_q,
    DEPTH => DATA_DEPTH_q
  )
  port map(
    address0 => q_address0,
    address1 => q_address1,
    ce0 => '1',
    ce1 => q_ce1,
    clk => tb_clk,
    din0 => q_din0,
    din1 => q_din1,
    done => tb_stop,
    dout0 => q_dout0,
    dout1 => q_dout1,
    rst => tb_rst,
    we0 => q_we0,
    we1 => q_we1
  );

  mem_inst_p: entity work.two_port_RAM
  generic map(
    TV_IN => INPUT_p,
    TV_OUT => OUTPUT_p,
    DATA_WIDTH => DATA_WIDTH_p,
    ADDR_WIDTH => ADDR_WIDTH_p,
    DEPTH => DATA_DEPTH_p
  )
  port map(
    address0 => p_address0,
    address1 => p_address1,
    ce0 => '1',
    ce1 => p_ce1,
    clk => tb_clk,
    din0 => p_din0,
    din1 => p_din1,
    done => tb_stop,
    dout0 => p_dout0,
    dout1 => p_dout1,
    rst => tb_rst,
    we0 => p_we0,
    we1 => p_we1
  );

  mem_inst_r: entity work.two_port_RAM
  generic map(
    TV_IN => INPUT_r,
    TV_OUT => OUTPUT_r,
    DATA_WIDTH => DATA_WIDTH_r,
    ADDR_WIDTH => ADDR_WIDTH_r,
    DEPTH => DATA_DEPTH_r
  )
  port map(
    address0 => r_address0,
    address1 => r_address1,
    ce0 => '1',
    ce1 => r_ce1,
    clk => tb_clk,
    din0 => r_din0,
    din1 => r_din1,
    done => tb_stop,
    dout0 => r_dout0,
    dout1 => r_dout1,
    rst => tb_rst,
    we0 => r_we0,
    we1 => r_we1
  );

  arg_inst_out0: entity work.single_argument
  generic map(
    TV_IN => INPUT_out0,
    TV_OUT => OUTPUT_out0,
    DATA_WIDTH => DATA_WIDTH_out0
  )
  port map(
    ce0 => '1',
    clk => tb_clk,
    din0 => out0_din0,
    done => tb_temp_idle,
    dout0 => out0_dout0,
    dout0_ready => out0_dout0_ready,
    dout0_valid => out0_dout0_valid,
    rst => tb_rst,
    we0 => out0_valid
  );

  join_valids: entity work.tb_join
  generic map(
    SIZE => 7
  )
  port map(
    ins_ready(0) => out0_ready,
    ins_ready(1) => a_end_ready,
    ins_ready(2) => s_end_ready,
    ins_ready(3) => q_end_ready,
    ins_ready(4) => p_end_ready,
    ins_ready(5) => r_end_ready,
    ins_ready(6) => end_ready,
    ins_valid(0) => out0_valid,
    ins_valid(1) => a_end_valid,
    ins_valid(2) => s_end_valid,
    ins_valid(3) => q_end_valid,
    ins_valid(4) => p_end_valid,
    ins_valid(5) => r_end_valid,
    ins_valid(6) => end_valid,
    outs_ready => tb_global_ready,
    outs_valid => tb_global_valid
  );


  write_output_transactor_a_runtime_proc : process
    file fp             : TEXT;
    variable fstatus    : FILE_OPEN_STATUS;
    variable token_line : LINE;
    variable token      : STRING(1 to 1024);
  begin
    file_open(fstatus, fp, OUTPUT_a , WRITE_MODE);
    if (fstatus /= OPEN_OK) then
      assert false report "Open file " & OUTPUT_a & " failed!!!" severity note;
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
    file_open(fstatus, fp, OUTPUT_a, APPEND_MODE);
    if (fstatus /= OPEN_OK) then
      assert false report "Open file " & OUTPUT_a & " failed!!!" severity note;
      assert false report "ERROR: Simulation using HLS TB failed." severity failure;
    end if;
    write(token_line, string'("[[[/runtime]]]"));
    writeline(fp, token_line);
    file_close(fp);
    wait;
  end process;

  write_output_transactor_s_runtime_proc : process
    file fp             : TEXT;
    variable fstatus    : FILE_OPEN_STATUS;
    variable token_line : LINE;
    variable token      : STRING(1 to 1024);
  begin
    file_open(fstatus, fp, OUTPUT_s , WRITE_MODE);
    if (fstatus /= OPEN_OK) then
      assert false report "Open file " & OUTPUT_s & " failed!!!" severity note;
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
    file_open(fstatus, fp, OUTPUT_s, APPEND_MODE);
    if (fstatus /= OPEN_OK) then
      assert false report "Open file " & OUTPUT_s & " failed!!!" severity note;
      assert false report "ERROR: Simulation using HLS TB failed." severity failure;
    end if;
    write(token_line, string'("[[[/runtime]]]"));
    writeline(fp, token_line);
    file_close(fp);
    wait;
  end process;

  write_output_transactor_q_runtime_proc : process
    file fp             : TEXT;
    variable fstatus    : FILE_OPEN_STATUS;
    variable token_line : LINE;
    variable token      : STRING(1 to 1024);
  begin
    file_open(fstatus, fp, OUTPUT_q , WRITE_MODE);
    if (fstatus /= OPEN_OK) then
      assert false report "Open file " & OUTPUT_q & " failed!!!" severity note;
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
    file_open(fstatus, fp, OUTPUT_q, APPEND_MODE);
    if (fstatus /= OPEN_OK) then
      assert false report "Open file " & OUTPUT_q & " failed!!!" severity note;
      assert false report "ERROR: Simulation using HLS TB failed." severity failure;
    end if;
    write(token_line, string'("[[[/runtime]]]"));
    writeline(fp, token_line);
    file_close(fp);
    wait;
  end process;

  write_output_transactor_p_runtime_proc : process
    file fp             : TEXT;
    variable fstatus    : FILE_OPEN_STATUS;
    variable token_line : LINE;
    variable token      : STRING(1 to 1024);
  begin
    file_open(fstatus, fp, OUTPUT_p , WRITE_MODE);
    if (fstatus /= OPEN_OK) then
      assert false report "Open file " & OUTPUT_p & " failed!!!" severity note;
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
    file_open(fstatus, fp, OUTPUT_p, APPEND_MODE);
    if (fstatus /= OPEN_OK) then
      assert false report "Open file " & OUTPUT_p & " failed!!!" severity note;
      assert false report "ERROR: Simulation using HLS TB failed." severity failure;
    end if;
    write(token_line, string'("[[[/runtime]]]"));
    writeline(fp, token_line);
    file_close(fp);
    wait;
  end process;

  write_output_transactor_r_runtime_proc : process
    file fp             : TEXT;
    variable fstatus    : FILE_OPEN_STATUS;
    variable token_line : LINE;
    variable token      : STRING(1 to 1024);
  begin
    file_open(fstatus, fp, OUTPUT_r , WRITE_MODE);
    if (fstatus /= OPEN_OK) then
      assert false report "Open file " & OUTPUT_r & " failed!!!" severity note;
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
    file_open(fstatus, fp, OUTPUT_r, APPEND_MODE);
    if (fstatus /= OPEN_OK) then
      assert false report "Open file " & OUTPUT_r & " failed!!!" severity note;
      assert false report "ERROR: Simulation using HLS TB failed." severity failure;
    end if;
    write(token_line, string'("[[[/runtime]]]"));
    writeline(fp, token_line);
    file_close(fp);
    wait;
  end process;

  write_output_transactor_out0_runtime_proc : process
    file fp             : TEXT;
    variable fstatus    : FILE_OPEN_STATUS;
    variable token_line : LINE;
    variable token      : STRING(1 to 1024);
  begin
    file_open(fstatus, fp, OUTPUT_out0 , WRITE_MODE);
    if (fstatus /= OPEN_OK) then
      assert false report "Open file " & OUTPUT_out0 & " failed!!!" severity note;
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
    file_open(fstatus, fp, OUTPUT_out0, APPEND_MODE);
    if (fstatus /= OPEN_OK) then
      assert false report "Open file " & OUTPUT_out0 & " failed!!!" severity note;
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
