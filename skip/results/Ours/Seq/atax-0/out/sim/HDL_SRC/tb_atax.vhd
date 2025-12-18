
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

  constant INPUT_A : STRING := "/beta/rouzbeh/hope/dynamatic/skip/runs/run_2025-12-16_00-22-51/atax-0/out/sim/INPUT_VECTORS/input_A.dat";
  constant OUTPUT_A : STRING := "/beta/rouzbeh/hope/dynamatic/skip/runs/run_2025-12-16_00-22-51/atax-0/out/sim/HDL_OUT/output_A.dat";
  constant DATA_WIDTH_A : INTEGER := 32;
  constant ADDR_WIDTH_A : INTEGER := 9;
  constant DATA_DEPTH_A : INTEGER := 400;
  constant INPUT_x : STRING := "/beta/rouzbeh/hope/dynamatic/skip/runs/run_2025-12-16_00-22-51/atax-0/out/sim/INPUT_VECTORS/input_x.dat";
  constant OUTPUT_x : STRING := "/beta/rouzbeh/hope/dynamatic/skip/runs/run_2025-12-16_00-22-51/atax-0/out/sim/HDL_OUT/output_x.dat";
  constant DATA_WIDTH_x : INTEGER := 32;
  constant ADDR_WIDTH_x : INTEGER := 5;
  constant DATA_DEPTH_x : INTEGER := 20;
  constant INPUT_y : STRING := "/beta/rouzbeh/hope/dynamatic/skip/runs/run_2025-12-16_00-22-51/atax-0/out/sim/INPUT_VECTORS/input_y.dat";
  constant OUTPUT_y : STRING := "/beta/rouzbeh/hope/dynamatic/skip/runs/run_2025-12-16_00-22-51/atax-0/out/sim/HDL_OUT/output_y.dat";
  constant DATA_WIDTH_y : INTEGER := 32;
  constant ADDR_WIDTH_y : INTEGER := 5;
  constant DATA_DEPTH_y : INTEGER := 20;
  constant INPUT_tmp : STRING := "/beta/rouzbeh/hope/dynamatic/skip/runs/run_2025-12-16_00-22-51/atax-0/out/sim/INPUT_VECTORS/input_tmp.dat";
  constant OUTPUT_tmp : STRING := "/beta/rouzbeh/hope/dynamatic/skip/runs/run_2025-12-16_00-22-51/atax-0/out/sim/HDL_OUT/output_tmp.dat";
  constant DATA_WIDTH_tmp : INTEGER := 32;
  constant ADDR_WIDTH_tmp : INTEGER := 5;
  constant DATA_DEPTH_tmp : INTEGER := 20;
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
  signal A_we0 : std_logic;
  signal A_din0 : std_logic_vector(32 - 1 downto 0);
  signal A_address0 : std_logic_vector(9 - 1 downto 0);
  signal A_ce1 : std_logic;
  signal A_dout1 : std_logic_vector(32 - 1 downto 0);
  signal A_address1 : std_logic_vector(9 - 1 downto 0);
  signal A_dout0 : std_logic_vector(32 - 1 downto 0);
  signal A_din1 : std_logic_vector(32 - 1 downto 0) := (others => '0');
  signal A_we1 : std_logic := '0';
  signal A_ce0 : std_logic := '1';
  signal x_we0 : std_logic;
  signal x_din0 : std_logic_vector(32 - 1 downto 0);
  signal x_address0 : std_logic_vector(5 - 1 downto 0);
  signal x_ce1 : std_logic;
  signal x_dout1 : std_logic_vector(32 - 1 downto 0);
  signal x_address1 : std_logic_vector(5 - 1 downto 0);
  signal x_dout0 : std_logic_vector(32 - 1 downto 0);
  signal x_din1 : std_logic_vector(32 - 1 downto 0) := (others => '0');
  signal x_we1 : std_logic := '0';
  signal x_ce0 : std_logic := '1';
  signal y_we0 : std_logic;
  signal y_din0 : std_logic_vector(32 - 1 downto 0);
  signal y_address0 : std_logic_vector(5 - 1 downto 0);
  signal y_ce1 : std_logic;
  signal y_dout1 : std_logic_vector(32 - 1 downto 0);
  signal y_address1 : std_logic_vector(5 - 1 downto 0);
  signal y_dout0 : std_logic_vector(32 - 1 downto 0);
  signal y_din1 : std_logic_vector(32 - 1 downto 0) := (others => '0');
  signal y_we1 : std_logic := '0';
  signal y_ce0 : std_logic := '1';
  signal tmp_we0 : std_logic;
  signal tmp_din0 : std_logic_vector(32 - 1 downto 0);
  signal tmp_address0 : std_logic_vector(5 - 1 downto 0);
  signal tmp_ce1 : std_logic;
  signal tmp_dout1 : std_logic_vector(32 - 1 downto 0);
  signal tmp_address1 : std_logic_vector(5 - 1 downto 0);
  signal tmp_dout0 : std_logic_vector(32 - 1 downto 0);
  signal tmp_din1 : std_logic_vector(32 - 1 downto 0) := (others => '0');
  signal tmp_we1 : std_logic := '0';
  signal tmp_ce0 : std_logic := '1';
  signal A_end_valid : std_logic;
  signal A_end_ready : std_logic;
  signal x_end_valid : std_logic;
  signal x_end_ready : std_logic;
  signal y_end_valid : std_logic;
  signal y_end_ready : std_logic;
  signal tmp_end_valid : std_logic;
  signal tmp_end_ready : std_logic;
  signal end_valid : std_logic;
  signal end_ready : std_logic;

  signal tb_temp_idle : std_logic := '1';
  shared variable transaction_idx : INTEGER := 0;
begin

  duv_inst: entity work.atax
  port map(
    A_end_ready => A_end_ready,
    A_end_valid => A_end_valid,
    A_loadAddr => A_address1,
    A_loadData => A_dout1,
    A_loadEn => A_ce1,
    A_start_ready => open,
    A_start_valid => '1',
    A_storeAddr => A_address0,
    A_storeData => A_din0,
    A_storeEn => A_we0,
    clk => tb_clk,
    end_ready => end_ready,
    end_valid => end_valid,
    rst => tb_rst,
    start_ready => open,
    start_valid => '1',
    tmp_end_ready => tmp_end_ready,
    tmp_end_valid => tmp_end_valid,
    tmp_loadAddr => tmp_address1,
    tmp_loadData => tmp_dout1,
    tmp_loadEn => tmp_ce1,
    tmp_start_ready => open,
    tmp_start_valid => '1',
    tmp_storeAddr => tmp_address0,
    tmp_storeData => tmp_din0,
    tmp_storeEn => tmp_we0,
    x_end_ready => x_end_ready,
    x_end_valid => x_end_valid,
    x_loadAddr => x_address1,
    x_loadData => x_dout1,
    x_loadEn => x_ce1,
    x_start_ready => open,
    x_start_valid => '1',
    x_storeAddr => x_address0,
    x_storeData => x_din0,
    x_storeEn => x_we0,
    y_end_ready => y_end_ready,
    y_end_valid => y_end_valid,
    y_loadAddr => y_address1,
    y_loadData => y_dout1,
    y_loadEn => y_ce1,
    y_start_ready => open,
    y_start_valid => '1',
    y_storeAddr => y_address0,
    y_storeData => y_din0,
    y_storeEn => y_we0
  );

  mem_inst_A: entity work.two_port_RAM
  generic map(
    TV_IN => INPUT_A,
    TV_OUT => OUTPUT_A,
    DATA_WIDTH => DATA_WIDTH_A,
    ADDR_WIDTH => ADDR_WIDTH_A,
    DEPTH => DATA_DEPTH_A
  )
  port map(
    address0 => A_address0,
    address1 => A_address1,
    ce0 => '1',
    ce1 => A_ce1,
    clk => tb_clk,
    din0 => A_din0,
    din1 => A_din1,
    done => tb_stop,
    dout0 => A_dout0,
    dout1 => A_dout1,
    rst => tb_rst,
    we0 => A_we0,
    we1 => A_we1
  );

  mem_inst_x: entity work.two_port_RAM
  generic map(
    TV_IN => INPUT_x,
    TV_OUT => OUTPUT_x,
    DATA_WIDTH => DATA_WIDTH_x,
    ADDR_WIDTH => ADDR_WIDTH_x,
    DEPTH => DATA_DEPTH_x
  )
  port map(
    address0 => x_address0,
    address1 => x_address1,
    ce0 => '1',
    ce1 => x_ce1,
    clk => tb_clk,
    din0 => x_din0,
    din1 => x_din1,
    done => tb_stop,
    dout0 => x_dout0,
    dout1 => x_dout1,
    rst => tb_rst,
    we0 => x_we0,
    we1 => x_we1
  );

  mem_inst_y: entity work.two_port_RAM
  generic map(
    TV_IN => INPUT_y,
    TV_OUT => OUTPUT_y,
    DATA_WIDTH => DATA_WIDTH_y,
    ADDR_WIDTH => ADDR_WIDTH_y,
    DEPTH => DATA_DEPTH_y
  )
  port map(
    address0 => y_address0,
    address1 => y_address1,
    ce0 => '1',
    ce1 => y_ce1,
    clk => tb_clk,
    din0 => y_din0,
    din1 => y_din1,
    done => tb_stop,
    dout0 => y_dout0,
    dout1 => y_dout1,
    rst => tb_rst,
    we0 => y_we0,
    we1 => y_we1
  );

  mem_inst_tmp: entity work.two_port_RAM
  generic map(
    TV_IN => INPUT_tmp,
    TV_OUT => OUTPUT_tmp,
    DATA_WIDTH => DATA_WIDTH_tmp,
    ADDR_WIDTH => ADDR_WIDTH_tmp,
    DEPTH => DATA_DEPTH_tmp
  )
  port map(
    address0 => tmp_address0,
    address1 => tmp_address1,
    ce0 => '1',
    ce1 => tmp_ce1,
    clk => tb_clk,
    din0 => tmp_din0,
    din1 => tmp_din1,
    done => tb_stop,
    dout0 => tmp_dout0,
    dout1 => tmp_dout1,
    rst => tb_rst,
    we0 => tmp_we0,
    we1 => tmp_we1
  );

  join_valids: entity work.tb_join
  generic map(
    SIZE => 5
  )
  port map(
    ins_ready(0) => A_end_ready,
    ins_ready(1) => x_end_ready,
    ins_ready(2) => y_end_ready,
    ins_ready(3) => tmp_end_ready,
    ins_ready(4) => end_ready,
    ins_valid(0) => A_end_valid,
    ins_valid(1) => x_end_valid,
    ins_valid(2) => y_end_valid,
    ins_valid(3) => tmp_end_valid,
    ins_valid(4) => end_valid,
    outs_ready => tb_global_ready,
    outs_valid => tb_global_valid
  );


  write_output_transactor_A_runtime_proc : process
    file fp             : TEXT;
    variable fstatus    : FILE_OPEN_STATUS;
    variable token_line : LINE;
    variable token      : STRING(1 to 1024);
  begin
    file_open(fstatus, fp, OUTPUT_A , WRITE_MODE);
    if (fstatus /= OPEN_OK) then
      assert false report "Open file " & OUTPUT_A & " failed!!!" severity note;
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
    file_open(fstatus, fp, OUTPUT_A, APPEND_MODE);
    if (fstatus /= OPEN_OK) then
      assert false report "Open file " & OUTPUT_A & " failed!!!" severity note;
      assert false report "ERROR: Simulation using HLS TB failed." severity failure;
    end if;
    write(token_line, string'("[[[/runtime]]]"));
    writeline(fp, token_line);
    file_close(fp);
    wait;
  end process;

  write_output_transactor_x_runtime_proc : process
    file fp             : TEXT;
    variable fstatus    : FILE_OPEN_STATUS;
    variable token_line : LINE;
    variable token      : STRING(1 to 1024);
  begin
    file_open(fstatus, fp, OUTPUT_x , WRITE_MODE);
    if (fstatus /= OPEN_OK) then
      assert false report "Open file " & OUTPUT_x & " failed!!!" severity note;
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
    file_open(fstatus, fp, OUTPUT_x, APPEND_MODE);
    if (fstatus /= OPEN_OK) then
      assert false report "Open file " & OUTPUT_x & " failed!!!" severity note;
      assert false report "ERROR: Simulation using HLS TB failed." severity failure;
    end if;
    write(token_line, string'("[[[/runtime]]]"));
    writeline(fp, token_line);
    file_close(fp);
    wait;
  end process;

  write_output_transactor_y_runtime_proc : process
    file fp             : TEXT;
    variable fstatus    : FILE_OPEN_STATUS;
    variable token_line : LINE;
    variable token      : STRING(1 to 1024);
  begin
    file_open(fstatus, fp, OUTPUT_y , WRITE_MODE);
    if (fstatus /= OPEN_OK) then
      assert false report "Open file " & OUTPUT_y & " failed!!!" severity note;
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
    file_open(fstatus, fp, OUTPUT_y, APPEND_MODE);
    if (fstatus /= OPEN_OK) then
      assert false report "Open file " & OUTPUT_y & " failed!!!" severity note;
      assert false report "ERROR: Simulation using HLS TB failed." severity failure;
    end if;
    write(token_line, string'("[[[/runtime]]]"));
    writeline(fp, token_line);
    file_close(fp);
    wait;
  end process;

  write_output_transactor_tmp_runtime_proc : process
    file fp             : TEXT;
    variable fstatus    : FILE_OPEN_STATUS;
    variable token_line : LINE;
    variable token      : STRING(1 to 1024);
  begin
    file_open(fstatus, fp, OUTPUT_tmp , WRITE_MODE);
    if (fstatus /= OPEN_OK) then
      assert false report "Open file " & OUTPUT_tmp & " failed!!!" severity note;
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
    file_open(fstatus, fp, OUTPUT_tmp, APPEND_MODE);
    if (fstatus /= OPEN_OK) then
      assert false report "Open file " & OUTPUT_tmp & " failed!!!" severity note;
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
