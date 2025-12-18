
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

  constant INPUT_mat : STRING := "/beta/rouzbeh/hope/dynamatic/skip/runs/run_2025-12-16_00-21-42/matrix_power-0/out/sim/INPUT_VECTORS/input_mat.dat";
  constant OUTPUT_mat : STRING := "/beta/rouzbeh/hope/dynamatic/skip/runs/run_2025-12-16_00-21-42/matrix_power-0/out/sim/HDL_OUT/output_mat.dat";
  constant DATA_WIDTH_mat : INTEGER := 32;
  constant ADDR_WIDTH_mat : INTEGER := 9;
  constant DATA_DEPTH_mat : INTEGER := 400;
  constant INPUT_row : STRING := "/beta/rouzbeh/hope/dynamatic/skip/runs/run_2025-12-16_00-21-42/matrix_power-0/out/sim/INPUT_VECTORS/input_row.dat";
  constant OUTPUT_row : STRING := "/beta/rouzbeh/hope/dynamatic/skip/runs/run_2025-12-16_00-21-42/matrix_power-0/out/sim/HDL_OUT/output_row.dat";
  constant DATA_WIDTH_row : INTEGER := 32;
  constant ADDR_WIDTH_row : INTEGER := 5;
  constant DATA_DEPTH_row : INTEGER := 20;
  constant INPUT_col : STRING := "/beta/rouzbeh/hope/dynamatic/skip/runs/run_2025-12-16_00-21-42/matrix_power-0/out/sim/INPUT_VECTORS/input_col.dat";
  constant OUTPUT_col : STRING := "/beta/rouzbeh/hope/dynamatic/skip/runs/run_2025-12-16_00-21-42/matrix_power-0/out/sim/HDL_OUT/output_col.dat";
  constant DATA_WIDTH_col : INTEGER := 32;
  constant ADDR_WIDTH_col : INTEGER := 5;
  constant DATA_DEPTH_col : INTEGER := 20;
  constant INPUT_a : STRING := "/beta/rouzbeh/hope/dynamatic/skip/runs/run_2025-12-16_00-21-42/matrix_power-0/out/sim/INPUT_VECTORS/input_a.dat";
  constant OUTPUT_a : STRING := "/beta/rouzbeh/hope/dynamatic/skip/runs/run_2025-12-16_00-21-42/matrix_power-0/out/sim/HDL_OUT/output_a.dat";
  constant DATA_WIDTH_a : INTEGER := 32;
  constant ADDR_WIDTH_a : INTEGER := 5;
  constant DATA_DEPTH_a : INTEGER := 20;
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
  signal mat_we0 : std_logic;
  signal mat_din0 : std_logic_vector(32 - 1 downto 0);
  signal mat_address0 : std_logic_vector(9 - 1 downto 0);
  signal mat_ce1 : std_logic;
  signal mat_dout1 : std_logic_vector(32 - 1 downto 0);
  signal mat_address1 : std_logic_vector(9 - 1 downto 0);
  signal mat_dout0 : std_logic_vector(32 - 1 downto 0);
  signal mat_din1 : std_logic_vector(32 - 1 downto 0) := (others => '0');
  signal mat_we1 : std_logic := '0';
  signal mat_ce0 : std_logic := '1';
  signal row_we0 : std_logic;
  signal row_din0 : std_logic_vector(32 - 1 downto 0);
  signal row_address0 : std_logic_vector(5 - 1 downto 0);
  signal row_ce1 : std_logic;
  signal row_dout1 : std_logic_vector(32 - 1 downto 0);
  signal row_address1 : std_logic_vector(5 - 1 downto 0);
  signal row_dout0 : std_logic_vector(32 - 1 downto 0);
  signal row_din1 : std_logic_vector(32 - 1 downto 0) := (others => '0');
  signal row_we1 : std_logic := '0';
  signal row_ce0 : std_logic := '1';
  signal col_we0 : std_logic;
  signal col_din0 : std_logic_vector(32 - 1 downto 0);
  signal col_address0 : std_logic_vector(5 - 1 downto 0);
  signal col_ce1 : std_logic;
  signal col_dout1 : std_logic_vector(32 - 1 downto 0);
  signal col_address1 : std_logic_vector(5 - 1 downto 0);
  signal col_dout0 : std_logic_vector(32 - 1 downto 0);
  signal col_din1 : std_logic_vector(32 - 1 downto 0) := (others => '0');
  signal col_we1 : std_logic := '0';
  signal col_ce0 : std_logic := '1';
  signal a_we0 : std_logic;
  signal a_din0 : std_logic_vector(32 - 1 downto 0);
  signal a_address0 : std_logic_vector(5 - 1 downto 0);
  signal a_ce1 : std_logic;
  signal a_dout1 : std_logic_vector(32 - 1 downto 0);
  signal a_address1 : std_logic_vector(5 - 1 downto 0);
  signal a_dout0 : std_logic_vector(32 - 1 downto 0);
  signal a_din1 : std_logic_vector(32 - 1 downto 0) := (others => '0');
  signal a_we1 : std_logic := '0';
  signal a_ce0 : std_logic := '1';
  signal mat_end_valid : std_logic;
  signal mat_end_ready : std_logic;
  signal row_end_valid : std_logic;
  signal row_end_ready : std_logic;
  signal col_end_valid : std_logic;
  signal col_end_ready : std_logic;
  signal a_end_valid : std_logic;
  signal a_end_ready : std_logic;
  signal end_valid : std_logic;
  signal end_ready : std_logic;

  signal tb_temp_idle : std_logic := '1';
  shared variable transaction_idx : INTEGER := 0;
begin

  duv_inst: entity work.matrix_power
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
    col_end_ready => col_end_ready,
    col_end_valid => col_end_valid,
    col_loadAddr => col_address1,
    col_loadData => col_dout1,
    col_loadEn => col_ce1,
    col_start_ready => open,
    col_start_valid => '1',
    col_storeAddr => col_address0,
    col_storeData => col_din0,
    col_storeEn => col_we0,
    end_ready => end_ready,
    end_valid => end_valid,
    mat_end_ready => mat_end_ready,
    mat_end_valid => mat_end_valid,
    mat_loadAddr => mat_address1,
    mat_loadData => mat_dout1,
    mat_loadEn => mat_ce1,
    mat_start_ready => open,
    mat_start_valid => '1',
    mat_storeAddr => mat_address0,
    mat_storeData => mat_din0,
    mat_storeEn => mat_we0,
    row_end_ready => row_end_ready,
    row_end_valid => row_end_valid,
    row_loadAddr => row_address1,
    row_loadData => row_dout1,
    row_loadEn => row_ce1,
    row_start_ready => open,
    row_start_valid => '1',
    row_storeAddr => row_address0,
    row_storeData => row_din0,
    row_storeEn => row_we0,
    rst => tb_rst,
    start_ready => open,
    start_valid => '1'
  );

  mem_inst_mat: entity work.two_port_RAM
  generic map(
    TV_IN => INPUT_mat,
    TV_OUT => OUTPUT_mat,
    DATA_WIDTH => DATA_WIDTH_mat,
    ADDR_WIDTH => ADDR_WIDTH_mat,
    DEPTH => DATA_DEPTH_mat
  )
  port map(
    address0 => mat_address0,
    address1 => mat_address1,
    ce0 => '1',
    ce1 => mat_ce1,
    clk => tb_clk,
    din0 => mat_din0,
    din1 => mat_din1,
    done => tb_stop,
    dout0 => mat_dout0,
    dout1 => mat_dout1,
    rst => tb_rst,
    we0 => mat_we0,
    we1 => mat_we1
  );

  mem_inst_row: entity work.two_port_RAM
  generic map(
    TV_IN => INPUT_row,
    TV_OUT => OUTPUT_row,
    DATA_WIDTH => DATA_WIDTH_row,
    ADDR_WIDTH => ADDR_WIDTH_row,
    DEPTH => DATA_DEPTH_row
  )
  port map(
    address0 => row_address0,
    address1 => row_address1,
    ce0 => '1',
    ce1 => row_ce1,
    clk => tb_clk,
    din0 => row_din0,
    din1 => row_din1,
    done => tb_stop,
    dout0 => row_dout0,
    dout1 => row_dout1,
    rst => tb_rst,
    we0 => row_we0,
    we1 => row_we1
  );

  mem_inst_col: entity work.two_port_RAM
  generic map(
    TV_IN => INPUT_col,
    TV_OUT => OUTPUT_col,
    DATA_WIDTH => DATA_WIDTH_col,
    ADDR_WIDTH => ADDR_WIDTH_col,
    DEPTH => DATA_DEPTH_col
  )
  port map(
    address0 => col_address0,
    address1 => col_address1,
    ce0 => '1',
    ce1 => col_ce1,
    clk => tb_clk,
    din0 => col_din0,
    din1 => col_din1,
    done => tb_stop,
    dout0 => col_dout0,
    dout1 => col_dout1,
    rst => tb_rst,
    we0 => col_we0,
    we1 => col_we1
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

  join_valids: entity work.tb_join
  generic map(
    SIZE => 5
  )
  port map(
    ins_ready(0) => mat_end_ready,
    ins_ready(1) => row_end_ready,
    ins_ready(2) => col_end_ready,
    ins_ready(3) => a_end_ready,
    ins_ready(4) => end_ready,
    ins_valid(0) => mat_end_valid,
    ins_valid(1) => row_end_valid,
    ins_valid(2) => col_end_valid,
    ins_valid(3) => a_end_valid,
    ins_valid(4) => end_valid,
    outs_ready => tb_global_ready,
    outs_valid => tb_global_valid
  );


  write_output_transactor_mat_runtime_proc : process
    file fp             : TEXT;
    variable fstatus    : FILE_OPEN_STATUS;
    variable token_line : LINE;
    variable token      : STRING(1 to 1024);
  begin
    file_open(fstatus, fp, OUTPUT_mat , WRITE_MODE);
    if (fstatus /= OPEN_OK) then
      assert false report "Open file " & OUTPUT_mat & " failed!!!" severity note;
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
    file_open(fstatus, fp, OUTPUT_mat, APPEND_MODE);
    if (fstatus /= OPEN_OK) then
      assert false report "Open file " & OUTPUT_mat & " failed!!!" severity note;
      assert false report "ERROR: Simulation using HLS TB failed." severity failure;
    end if;
    write(token_line, string'("[[[/runtime]]]"));
    writeline(fp, token_line);
    file_close(fp);
    wait;
  end process;

  write_output_transactor_row_runtime_proc : process
    file fp             : TEXT;
    variable fstatus    : FILE_OPEN_STATUS;
    variable token_line : LINE;
    variable token      : STRING(1 to 1024);
  begin
    file_open(fstatus, fp, OUTPUT_row , WRITE_MODE);
    if (fstatus /= OPEN_OK) then
      assert false report "Open file " & OUTPUT_row & " failed!!!" severity note;
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
    file_open(fstatus, fp, OUTPUT_row, APPEND_MODE);
    if (fstatus /= OPEN_OK) then
      assert false report "Open file " & OUTPUT_row & " failed!!!" severity note;
      assert false report "ERROR: Simulation using HLS TB failed." severity failure;
    end if;
    write(token_line, string'("[[[/runtime]]]"));
    writeline(fp, token_line);
    file_close(fp);
    wait;
  end process;

  write_output_transactor_col_runtime_proc : process
    file fp             : TEXT;
    variable fstatus    : FILE_OPEN_STATUS;
    variable token_line : LINE;
    variable token      : STRING(1 to 1024);
  begin
    file_open(fstatus, fp, OUTPUT_col , WRITE_MODE);
    if (fstatus /= OPEN_OK) then
      assert false report "Open file " & OUTPUT_col & " failed!!!" severity note;
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
    file_open(fstatus, fp, OUTPUT_col, APPEND_MODE);
    if (fstatus /= OPEN_OK) then
      assert false report "Open file " & OUTPUT_col & " failed!!!" severity note;
      assert false report "ERROR: Simulation using HLS TB failed." severity failure;
    end if;
    write(token_line, string'("[[[/runtime]]]"));
    writeline(fp, token_line);
    file_close(fp);
    wait;
  end process;

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
