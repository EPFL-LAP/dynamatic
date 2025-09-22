
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

  constant INPUT_a : STRING := "/home/shundroid/dynamatic/integration-test/bisection/out_1/sim/INPUT_VECTORS/input_a.dat";
  constant OUTPUT_a : STRING := "/home/shundroid/dynamatic/integration-test/bisection/out_1/sim/HDL_OUT/output_a.dat";
  constant DATA_WIDTH_a : INTEGER := 32;
  constant INPUT_b : STRING := "/home/shundroid/dynamatic/integration-test/bisection/out_1/sim/INPUT_VECTORS/input_b.dat";
  constant OUTPUT_b : STRING := "/home/shundroid/dynamatic/integration-test/bisection/out_1/sim/HDL_OUT/output_b.dat";
  constant DATA_WIDTH_b : INTEGER := 32;
  constant INPUT_tol : STRING := "/home/shundroid/dynamatic/integration-test/bisection/out_1/sim/INPUT_VECTORS/input_tol.dat";
  constant OUTPUT_tol : STRING := "/home/shundroid/dynamatic/integration-test/bisection/out_1/sim/HDL_OUT/output_tol.dat";
  constant DATA_WIDTH_tol : INTEGER := 32;
  constant INPUT_out0 : STRING := "";
  constant OUTPUT_out0 : STRING := "/home/shundroid/dynamatic/integration-test/bisection/out_1/sim/HDL_OUT/output_out0.dat";
  constant DATA_WIDTH_out0 : INTEGER := 32;
  constant HALF_CLK_PERIOD : TIME := 2 ns;
  constant RESET_LATENCY : TIME := 10 ns;
  constant TRANSACTION_NUM : INTEGER := 1;
  signal tb_clk : std_logic := '0';
  signal tb_rst : std_logic := '0';
  signal tb_start_valid : std_logic := '0';
  signal tb_start_ready : std_logic := '0';
  signal tb_started : std_logic;
  signal tb_global_valid : std_logic;
  signal tb_global_ready : std_logic;
  signal tb_stop : std_logic;
  signal a_ce0 : std_logic;
  signal a_we0 : std_logic;
  signal a_din0 : std_logic_vector(32 - 1 downto 0);
  signal a_dout0 : std_logic_vector(32 - 1 downto 0);
  signal a_dout0_valid : std_logic;
  signal a_dout0_ready : std_logic;
  signal b_ce0 : std_logic;
  signal b_we0 : std_logic;
  signal b_din0 : std_logic_vector(32 - 1 downto 0);
  signal b_dout0 : std_logic_vector(32 - 1 downto 0);
  signal b_dout0_valid : std_logic;
  signal b_dout0_ready : std_logic;
  signal tol_ce0 : std_logic;
  signal tol_we0 : std_logic;
  signal tol_din0 : std_logic_vector(32 - 1 downto 0);
  signal tol_dout0 : std_logic_vector(32 - 1 downto 0);
  signal tol_dout0_valid : std_logic;
  signal tol_dout0_ready : std_logic;
  signal out0_valid : std_logic;
  signal out0_ready : std_logic;
  signal out0_ce0 : std_logic;
  signal out0_we0 : std_logic;
  signal out0_din0 : std_logic_vector(32 - 1 downto 0);
  signal out0_dout0 : std_logic_vector(32 - 1 downto 0);
  signal out0_dout0_valid : std_logic;
  signal out0_dout0_ready : std_logic;
  signal end_valid : std_logic;
  signal end_ready : std_logic;

  signal tb_temp_idle : std_logic := '1';
  shared variable transaction_idx : INTEGER := 0;
begin

  duv_inst: entity work.bisection
  port map(
    a => a_dout0,
    a_ready => a_dout0_ready,
    a_valid => a_dout0_valid,
    b => b_dout0,
    b_ready => b_dout0_ready,
    b_valid => b_dout0_valid,
    clk => tb_clk,
    end_ready => end_ready,
    end_valid => end_valid,
    out0 => out0_din0,
    out0_ready => out0_ready,
    out0_valid => out0_valid,
    rst => tb_rst,
    start_ready => open,
    start_valid => '1',
    tol => tol_dout0,
    tol_ready => tol_dout0_ready,
    tol_valid => tol_dout0_valid
  );

  arg_inst_a: entity work.single_argument
  generic map(
    TV_IN => INPUT_a,
    TV_OUT => OUTPUT_a,
    DATA_WIDTH => DATA_WIDTH_a
  )
  port map(
    ce0 => '1',
    clk => tb_clk,
    din0 => (others => '0'),
    done => tb_temp_idle,
    dout0 => a_dout0,
    dout0_ready => a_dout0_ready,
    dout0_valid => a_dout0_valid,
    rst => tb_rst,
    we0 => '0'
  );

  arg_inst_b: entity work.single_argument
  generic map(
    TV_IN => INPUT_b,
    TV_OUT => OUTPUT_b,
    DATA_WIDTH => DATA_WIDTH_b
  )
  port map(
    ce0 => '1',
    clk => tb_clk,
    din0 => (others => '0'),
    done => tb_temp_idle,
    dout0 => b_dout0,
    dout0_ready => b_dout0_ready,
    dout0_valid => b_dout0_valid,
    rst => tb_rst,
    we0 => '0'
  );

  arg_inst_tol: entity work.single_argument
  generic map(
    TV_IN => INPUT_tol,
    TV_OUT => OUTPUT_tol,
    DATA_WIDTH => DATA_WIDTH_tol
  )
  port map(
    ce0 => '1',
    clk => tb_clk,
    din0 => (others => '0'),
    done => tb_temp_idle,
    dout0 => tol_dout0,
    dout0_ready => tol_dout0_ready,
    dout0_valid => tol_dout0_valid,
    rst => tb_rst,
    we0 => '0'
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
    SIZE => 2
  )
  port map(
    ins_ready(0) => out0_ready,
    ins_ready(1) => end_ready,
    ins_valid(0) => out0_valid,
    ins_valid(1) => end_valid,
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

  write_output_transactor_b_runtime_proc : process
    file fp             : TEXT;
    variable fstatus    : FILE_OPEN_STATUS;
    variable token_line : LINE;
    variable token      : STRING(1 to 1024);
  begin
    file_open(fstatus, fp, OUTPUT_b , WRITE_MODE);
    if (fstatus /= OPEN_OK) then
      assert false report "Open file " & OUTPUT_b & " failed!!!" severity note;
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
    file_open(fstatus, fp, OUTPUT_b, APPEND_MODE);
    if (fstatus /= OPEN_OK) then
      assert false report "Open file " & OUTPUT_b & " failed!!!" severity note;
      assert false report "ERROR: Simulation using HLS TB failed." severity failure;
    end if;
    write(token_line, string'("[[[/runtime]]]"));
    writeline(fp, token_line);
    file_close(fp);
    wait;
  end process;

  write_output_transactor_tol_runtime_proc : process
    file fp             : TEXT;
    variable fstatus    : FILE_OPEN_STATUS;
    variable token_line : LINE;
    variable token      : STRING(1 to 1024);
  begin
    file_open(fstatus, fp, OUTPUT_tol , WRITE_MODE);
    if (fstatus /= OPEN_OK) then
      assert false report "Open file " & OUTPUT_tol & " failed!!!" severity note;
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
    file_open(fstatus, fp, OUTPUT_tol, APPEND_MODE);
    if (fstatus /= OPEN_OK) then
      assert false report "Open file " & OUTPUT_tol & " failed!!!" severity note;
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
