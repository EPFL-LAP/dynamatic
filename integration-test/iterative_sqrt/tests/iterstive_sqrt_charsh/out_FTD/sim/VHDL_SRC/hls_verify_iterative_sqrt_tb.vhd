
library IEEE;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
use ieee.std_logic_textio.all;
use ieee.numeric_std.all;
use std.textio.all;
use work.sim_package.all;
entity iterative_sqrt_wrapper_tb is
end entity iterative_sqrt_wrapper_tb;

architecture behavior of iterative_sqrt_wrapper_tb is

  constant HALF_CLK_PERIOD : TIME := 2.00 ns;
  constant TRANSACTION_NUM : INTEGER := 1;
  constant INPUT_out0 : STRING := "";
  constant OUTPUT_out0 : STRING := "../VHDL_OUT/output_out0.dat";
  constant DATA_WIDTH_out0 : INTEGER := 32;
  constant INPUT_n : STRING := "../INPUT_VECTORS/input_n.dat";
  constant OUTPUT_n : STRING := "";
  constant DATA_WIDTH_n : INTEGER := 32;
  signal tb_clk : std_logic := '0';
  signal tb_start_value : std_logic := '0';
  signal tb_rst : std_logic := '0';
  signal tb_start_valid : std_logic := '0';
  signal tb_start_ready : std_logic;
  signal tb_started : std_logic;
  signal tb_end_valid : std_logic;
  signal tb_end_ready : std_logic;
  signal tb_out0_valid : std_logic;
  signal tb_out0_ready : std_logic;
  signal tb_global_valid : std_logic;
  signal tb_global_ready : std_logic;
  signal tb_stop : std_logic;
  signal out0_mem_ce0 : std_logic;
  signal out0_mem_we0 : std_logic;
  signal out0_mem_dout0 : std_logic_vector(DATA_WIDTH_out0 - 1 downto 0);
  signal out0_mem_din0 : std_logic_vector(DATA_WIDTH_out0 - 1 downto 0);
  signal out0_mem_dout0_valid : std_logic;
  signal out0_mem_dout0_ready : std_logic;
  signal n_mem_ce0 : std_logic;
  signal n_mem_we0 : std_logic;
  signal n_mem_dout0 : std_logic_vector(DATA_WIDTH_n - 1 downto 0);
  signal n_mem_din0 : std_logic_vector(DATA_WIDTH_n - 1 downto 0);
  signal n_mem_dout0_valid : std_logic;
  signal n_mem_dout0_ready : std_logic;

  signal tb_temp_idle : std_logic := '1';
  shared variable transaction_idx : INTEGER := 0;
begin

  duv_inst: entity work.iterative_sqrt_wrapper
  port map(
    clk => tb_clk,
    end_ready => tb_end_ready,
    end_valid => tb_end_valid,
    n => n_mem_dout0,
    n_ready => n_mem_dout0_ready,
    n_valid => n_mem_dout0_valid,
    out0 => out0_mem_din0,
    out0_ready => tb_out0_ready,
    out0_valid => tb_out0_valid,
    rst => tb_rst,
    start_ready => tb_start_ready,
    start_valid => tb_start_valid
  );
  arg_inst_out0: entity work.single_argument
  generic map(
    TV_IN => INPUT_out0,
    TV_OUT => OUTPUT_out0,
    DATA_WIDTH => DATA_WIDTH_out0)
  port map(
    ce0 => '1',
    clk => tb_clk,
    din0 => out0_mem_din0,
    done => tb_temp_idle,
    dout0 => out0_mem_dout0,
    dout0_ready => out0_mem_dout0_ready,
    dout0_valid => out0_mem_dout0_valid,
    rst => tb_rst,
    we0 => tb_out0_valid
  );


  arg_inst_n: entity work.single_argument
  generic map(
    TV_IN => INPUT_n,
    TV_OUT => OUTPUT_n,
    DATA_WIDTH => DATA_WIDTH_n)
  port map(
    ce0 => '1',
    clk => tb_clk,
    din0 => (others => '0'),
    done => tb_temp_idle,
    dout0 => n_mem_dout0,
    dout0_ready => n_mem_dout0_ready,
    dout0_valid => n_mem_dout0_valid,
    rst => tb_rst,
    we0 => '0'
  );


  join_valids: entity work.tb_join
  generic map(
    SIZE => 2)
  port map(
    ins_ready(0) => tb_out0_ready,
    ins_ready(1) => tb_end_ready,
    ins_valid(0) => tb_out0_valid,
    ins_valid(1) => tb_end_valid,
    outs_ready => tb_global_ready,
    outs_valid => tb_global_valid
  );
  -- Write [[[runtime]]], [[[/runtime]]] for output transactor

  write_output_transactor_out0_runtime_proc : process
    file fp             : TEXT;
    variable fstatus    : FILE_OPEN_STATUS;
    variable token_line : LINE;
    variable token      : STRING(1 to 1024);
  begin
    file_open(fstatus, fp, OUTPUT_out0 , WRITE_MODE);
    if (fstatus /= OPEN_OK) then
      assert false
      report "Open file " & OUTPUT_out0 & " failed!!!"
      severity note;
      assert false
      report "ERROR: Simulation using HLS TB failed."
      severity failure;
    end if;
    write(token_line, string'("[[[runtime]]]"));
    writeline(fp, token_line);
    file_close(fp);
    while transaction_idx /= TRANSACTION_NUM loop
      wait until tb_clk'event and tb_clk = '1';
    end loop;
    wait until tb_clk'event and tb_clk = '1';
    wait until tb_clk'event and tb_clk = '1';
    file_open(fstatus, fp, OUTPUT_out0 , APPEND_MODE);
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
    report "simulation done!"
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
    wait for 10 ns;
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
