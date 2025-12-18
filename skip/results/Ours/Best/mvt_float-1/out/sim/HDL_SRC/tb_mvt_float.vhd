
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

  constant INPUT_A : STRING := "/beta/rouzbeh/hope/dynamatic/skip/runs/run_2025-12-17_01-09-18/mvt_float-1/out/sim/INPUT_VECTORS/input_A.dat";
  constant OUTPUT_A : STRING := "/beta/rouzbeh/hope/dynamatic/skip/runs/run_2025-12-17_01-09-18/mvt_float-1/out/sim/HDL_OUT/output_A.dat";
  constant DATA_WIDTH_A : INTEGER := 32;
  constant ADDR_WIDTH_A : INTEGER := 10;
  constant DATA_DEPTH_A : INTEGER := 900;
  constant INPUT_x1 : STRING := "/beta/rouzbeh/hope/dynamatic/skip/runs/run_2025-12-17_01-09-18/mvt_float-1/out/sim/INPUT_VECTORS/input_x1.dat";
  constant OUTPUT_x1 : STRING := "/beta/rouzbeh/hope/dynamatic/skip/runs/run_2025-12-17_01-09-18/mvt_float-1/out/sim/HDL_OUT/output_x1.dat";
  constant DATA_WIDTH_x1 : INTEGER := 32;
  constant ADDR_WIDTH_x1 : INTEGER := 5;
  constant DATA_DEPTH_x1 : INTEGER := 30;
  constant INPUT_x2 : STRING := "/beta/rouzbeh/hope/dynamatic/skip/runs/run_2025-12-17_01-09-18/mvt_float-1/out/sim/INPUT_VECTORS/input_x2.dat";
  constant OUTPUT_x2 : STRING := "/beta/rouzbeh/hope/dynamatic/skip/runs/run_2025-12-17_01-09-18/mvt_float-1/out/sim/HDL_OUT/output_x2.dat";
  constant DATA_WIDTH_x2 : INTEGER := 32;
  constant ADDR_WIDTH_x2 : INTEGER := 5;
  constant DATA_DEPTH_x2 : INTEGER := 30;
  constant INPUT_y1 : STRING := "/beta/rouzbeh/hope/dynamatic/skip/runs/run_2025-12-17_01-09-18/mvt_float-1/out/sim/INPUT_VECTORS/input_y1.dat";
  constant OUTPUT_y1 : STRING := "/beta/rouzbeh/hope/dynamatic/skip/runs/run_2025-12-17_01-09-18/mvt_float-1/out/sim/HDL_OUT/output_y1.dat";
  constant DATA_WIDTH_y1 : INTEGER := 32;
  constant ADDR_WIDTH_y1 : INTEGER := 5;
  constant DATA_DEPTH_y1 : INTEGER := 30;
  constant INPUT_y2 : STRING := "/beta/rouzbeh/hope/dynamatic/skip/runs/run_2025-12-17_01-09-18/mvt_float-1/out/sim/INPUT_VECTORS/input_y2.dat";
  constant OUTPUT_y2 : STRING := "/beta/rouzbeh/hope/dynamatic/skip/runs/run_2025-12-17_01-09-18/mvt_float-1/out/sim/HDL_OUT/output_y2.dat";
  constant DATA_WIDTH_y2 : INTEGER := 32;
  constant ADDR_WIDTH_y2 : INTEGER := 5;
  constant DATA_DEPTH_y2 : INTEGER := 30;
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
  signal A_address0 : std_logic_vector(10 - 1 downto 0);
  signal A_ce1 : std_logic;
  signal A_dout1 : std_logic_vector(32 - 1 downto 0);
  signal A_address1 : std_logic_vector(10 - 1 downto 0);
  signal A_dout0 : std_logic_vector(32 - 1 downto 0);
  signal A_din1 : std_logic_vector(32 - 1 downto 0) := (others => '0');
  signal A_we1 : std_logic := '0';
  signal A_ce0 : std_logic := '1';
  signal x1_we0 : std_logic;
  signal x1_din0 : std_logic_vector(32 - 1 downto 0);
  signal x1_address0 : std_logic_vector(5 - 1 downto 0);
  signal x1_ce1 : std_logic;
  signal x1_dout1 : std_logic_vector(32 - 1 downto 0);
  signal x1_address1 : std_logic_vector(5 - 1 downto 0);
  signal x1_dout0 : std_logic_vector(32 - 1 downto 0);
  signal x1_din1 : std_logic_vector(32 - 1 downto 0) := (others => '0');
  signal x1_we1 : std_logic := '0';
  signal x1_ce0 : std_logic := '1';
  signal x2_we0 : std_logic;
  signal x2_din0 : std_logic_vector(32 - 1 downto 0);
  signal x2_address0 : std_logic_vector(5 - 1 downto 0);
  signal x2_ce1 : std_logic;
  signal x2_dout1 : std_logic_vector(32 - 1 downto 0);
  signal x2_address1 : std_logic_vector(5 - 1 downto 0);
  signal x2_dout0 : std_logic_vector(32 - 1 downto 0);
  signal x2_din1 : std_logic_vector(32 - 1 downto 0) := (others => '0');
  signal x2_we1 : std_logic := '0';
  signal x2_ce0 : std_logic := '1';
  signal y1_we0 : std_logic;
  signal y1_din0 : std_logic_vector(32 - 1 downto 0);
  signal y1_address0 : std_logic_vector(5 - 1 downto 0);
  signal y1_ce1 : std_logic;
  signal y1_dout1 : std_logic_vector(32 - 1 downto 0);
  signal y1_address1 : std_logic_vector(5 - 1 downto 0);
  signal y1_dout0 : std_logic_vector(32 - 1 downto 0);
  signal y1_din1 : std_logic_vector(32 - 1 downto 0) := (others => '0');
  signal y1_we1 : std_logic := '0';
  signal y1_ce0 : std_logic := '1';
  signal y2_we0 : std_logic;
  signal y2_din0 : std_logic_vector(32 - 1 downto 0);
  signal y2_address0 : std_logic_vector(5 - 1 downto 0);
  signal y2_ce1 : std_logic;
  signal y2_dout1 : std_logic_vector(32 - 1 downto 0);
  signal y2_address1 : std_logic_vector(5 - 1 downto 0);
  signal y2_dout0 : std_logic_vector(32 - 1 downto 0);
  signal y2_din1 : std_logic_vector(32 - 1 downto 0) := (others => '0');
  signal y2_we1 : std_logic := '0';
  signal y2_ce0 : std_logic := '1';
  signal A_end_valid : std_logic;
  signal A_end_ready : std_logic;
  signal x1_end_valid : std_logic;
  signal x1_end_ready : std_logic;
  signal x2_end_valid : std_logic;
  signal x2_end_ready : std_logic;
  signal y1_end_valid : std_logic;
  signal y1_end_ready : std_logic;
  signal y2_end_valid : std_logic;
  signal y2_end_ready : std_logic;
  signal end_valid : std_logic;
  signal end_ready : std_logic;

  signal tb_temp_idle : std_logic := '1';
  shared variable transaction_idx : INTEGER := 0;
begin

  duv_inst: entity work.mvt_float
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
    x1_end_ready => x1_end_ready,
    x1_end_valid => x1_end_valid,
    x1_loadAddr => x1_address1,
    x1_loadData => x1_dout1,
    x1_loadEn => x1_ce1,
    x1_start_ready => open,
    x1_start_valid => '1',
    x1_storeAddr => x1_address0,
    x1_storeData => x1_din0,
    x1_storeEn => x1_we0,
    x2_end_ready => x2_end_ready,
    x2_end_valid => x2_end_valid,
    x2_loadAddr => x2_address1,
    x2_loadData => x2_dout1,
    x2_loadEn => x2_ce1,
    x2_start_ready => open,
    x2_start_valid => '1',
    x2_storeAddr => x2_address0,
    x2_storeData => x2_din0,
    x2_storeEn => x2_we0,
    y1_end_ready => y1_end_ready,
    y1_end_valid => y1_end_valid,
    y1_loadAddr => y1_address1,
    y1_loadData => y1_dout1,
    y1_loadEn => y1_ce1,
    y1_start_ready => open,
    y1_start_valid => '1',
    y1_storeAddr => y1_address0,
    y1_storeData => y1_din0,
    y1_storeEn => y1_we0,
    y2_end_ready => y2_end_ready,
    y2_end_valid => y2_end_valid,
    y2_loadAddr => y2_address1,
    y2_loadData => y2_dout1,
    y2_loadEn => y2_ce1,
    y2_start_ready => open,
    y2_start_valid => '1',
    y2_storeAddr => y2_address0,
    y2_storeData => y2_din0,
    y2_storeEn => y2_we0
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

  mem_inst_x1: entity work.two_port_RAM
  generic map(
    TV_IN => INPUT_x1,
    TV_OUT => OUTPUT_x1,
    DATA_WIDTH => DATA_WIDTH_x1,
    ADDR_WIDTH => ADDR_WIDTH_x1,
    DEPTH => DATA_DEPTH_x1
  )
  port map(
    address0 => x1_address0,
    address1 => x1_address1,
    ce0 => '1',
    ce1 => x1_ce1,
    clk => tb_clk,
    din0 => x1_din0,
    din1 => x1_din1,
    done => tb_stop,
    dout0 => x1_dout0,
    dout1 => x1_dout1,
    rst => tb_rst,
    we0 => x1_we0,
    we1 => x1_we1
  );

  mem_inst_x2: entity work.two_port_RAM
  generic map(
    TV_IN => INPUT_x2,
    TV_OUT => OUTPUT_x2,
    DATA_WIDTH => DATA_WIDTH_x2,
    ADDR_WIDTH => ADDR_WIDTH_x2,
    DEPTH => DATA_DEPTH_x2
  )
  port map(
    address0 => x2_address0,
    address1 => x2_address1,
    ce0 => '1',
    ce1 => x2_ce1,
    clk => tb_clk,
    din0 => x2_din0,
    din1 => x2_din1,
    done => tb_stop,
    dout0 => x2_dout0,
    dout1 => x2_dout1,
    rst => tb_rst,
    we0 => x2_we0,
    we1 => x2_we1
  );

  mem_inst_y1: entity work.two_port_RAM
  generic map(
    TV_IN => INPUT_y1,
    TV_OUT => OUTPUT_y1,
    DATA_WIDTH => DATA_WIDTH_y1,
    ADDR_WIDTH => ADDR_WIDTH_y1,
    DEPTH => DATA_DEPTH_y1
  )
  port map(
    address0 => y1_address0,
    address1 => y1_address1,
    ce0 => '1',
    ce1 => y1_ce1,
    clk => tb_clk,
    din0 => y1_din0,
    din1 => y1_din1,
    done => tb_stop,
    dout0 => y1_dout0,
    dout1 => y1_dout1,
    rst => tb_rst,
    we0 => y1_we0,
    we1 => y1_we1
  );

  mem_inst_y2: entity work.two_port_RAM
  generic map(
    TV_IN => INPUT_y2,
    TV_OUT => OUTPUT_y2,
    DATA_WIDTH => DATA_WIDTH_y2,
    ADDR_WIDTH => ADDR_WIDTH_y2,
    DEPTH => DATA_DEPTH_y2
  )
  port map(
    address0 => y2_address0,
    address1 => y2_address1,
    ce0 => '1',
    ce1 => y2_ce1,
    clk => tb_clk,
    din0 => y2_din0,
    din1 => y2_din1,
    done => tb_stop,
    dout0 => y2_dout0,
    dout1 => y2_dout1,
    rst => tb_rst,
    we0 => y2_we0,
    we1 => y2_we1
  );

  join_valids: entity work.tb_join
  generic map(
    SIZE => 6
  )
  port map(
    ins_ready(0) => A_end_ready,
    ins_ready(1) => x1_end_ready,
    ins_ready(2) => x2_end_ready,
    ins_ready(3) => y1_end_ready,
    ins_ready(4) => y2_end_ready,
    ins_ready(5) => end_ready,
    ins_valid(0) => A_end_valid,
    ins_valid(1) => x1_end_valid,
    ins_valid(2) => x2_end_valid,
    ins_valid(3) => y1_end_valid,
    ins_valid(4) => y2_end_valid,
    ins_valid(5) => end_valid,
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

  write_output_transactor_x1_runtime_proc : process
    file fp             : TEXT;
    variable fstatus    : FILE_OPEN_STATUS;
    variable token_line : LINE;
    variable token      : STRING(1 to 1024);
  begin
    file_open(fstatus, fp, OUTPUT_x1 , WRITE_MODE);
    if (fstatus /= OPEN_OK) then
      assert false report "Open file " & OUTPUT_x1 & " failed!!!" severity note;
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
    file_open(fstatus, fp, OUTPUT_x1, APPEND_MODE);
    if (fstatus /= OPEN_OK) then
      assert false report "Open file " & OUTPUT_x1 & " failed!!!" severity note;
      assert false report "ERROR: Simulation using HLS TB failed." severity failure;
    end if;
    write(token_line, string'("[[[/runtime]]]"));
    writeline(fp, token_line);
    file_close(fp);
    wait;
  end process;

  write_output_transactor_x2_runtime_proc : process
    file fp             : TEXT;
    variable fstatus    : FILE_OPEN_STATUS;
    variable token_line : LINE;
    variable token      : STRING(1 to 1024);
  begin
    file_open(fstatus, fp, OUTPUT_x2 , WRITE_MODE);
    if (fstatus /= OPEN_OK) then
      assert false report "Open file " & OUTPUT_x2 & " failed!!!" severity note;
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
    file_open(fstatus, fp, OUTPUT_x2, APPEND_MODE);
    if (fstatus /= OPEN_OK) then
      assert false report "Open file " & OUTPUT_x2 & " failed!!!" severity note;
      assert false report "ERROR: Simulation using HLS TB failed." severity failure;
    end if;
    write(token_line, string'("[[[/runtime]]]"));
    writeline(fp, token_line);
    file_close(fp);
    wait;
  end process;

  write_output_transactor_y1_runtime_proc : process
    file fp             : TEXT;
    variable fstatus    : FILE_OPEN_STATUS;
    variable token_line : LINE;
    variable token      : STRING(1 to 1024);
  begin
    file_open(fstatus, fp, OUTPUT_y1 , WRITE_MODE);
    if (fstatus /= OPEN_OK) then
      assert false report "Open file " & OUTPUT_y1 & " failed!!!" severity note;
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
    file_open(fstatus, fp, OUTPUT_y1, APPEND_MODE);
    if (fstatus /= OPEN_OK) then
      assert false report "Open file " & OUTPUT_y1 & " failed!!!" severity note;
      assert false report "ERROR: Simulation using HLS TB failed." severity failure;
    end if;
    write(token_line, string'("[[[/runtime]]]"));
    writeline(fp, token_line);
    file_close(fp);
    wait;
  end process;

  write_output_transactor_y2_runtime_proc : process
    file fp             : TEXT;
    variable fstatus    : FILE_OPEN_STATUS;
    variable token_line : LINE;
    variable token      : STRING(1 to 1024);
  begin
    file_open(fstatus, fp, OUTPUT_y2 , WRITE_MODE);
    if (fstatus /= OPEN_OK) then
      assert false report "Open file " & OUTPUT_y2 & " failed!!!" severity note;
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
    file_open(fstatus, fp, OUTPUT_y2, APPEND_MODE);
    if (fstatus /= OPEN_OK) then
      assert false report "Open file " & OUTPUT_y2 & " failed!!!" severity note;
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
