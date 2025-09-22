
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

  constant INPUT_d1 : STRING := "/home/shundroid/dynamatic/integration-test/subdiag_fast/out_v1/sim/INPUT_VECTORS/input_d1.dat";
  constant OUTPUT_d1 : STRING := "/home/shundroid/dynamatic/integration-test/subdiag_fast/out_v1/sim/HDL_OUT/output_d1.dat";
  constant DATA_WIDTH_d1 : INTEGER := 32;
  constant ADDR_WIDTH_d1 : INTEGER := 10;
  constant DATA_DEPTH_d1 : INTEGER := 1000;
  constant INPUT_d2 : STRING := "/home/shundroid/dynamatic/integration-test/subdiag_fast/out_v1/sim/INPUT_VECTORS/input_d2.dat";
  constant OUTPUT_d2 : STRING := "/home/shundroid/dynamatic/integration-test/subdiag_fast/out_v1/sim/HDL_OUT/output_d2.dat";
  constant DATA_WIDTH_d2 : INTEGER := 32;
  constant ADDR_WIDTH_d2 : INTEGER := 10;
  constant DATA_DEPTH_d2 : INTEGER := 1000;
  constant INPUT_e : STRING := "/home/shundroid/dynamatic/integration-test/subdiag_fast/out_v1/sim/INPUT_VECTORS/input_e.dat";
  constant OUTPUT_e : STRING := "/home/shundroid/dynamatic/integration-test/subdiag_fast/out_v1/sim/HDL_OUT/output_e.dat";
  constant DATA_WIDTH_e : INTEGER := 32;
  constant ADDR_WIDTH_e : INTEGER := 10;
  constant DATA_DEPTH_e : INTEGER := 1000;
  constant INPUT_out0 : STRING := "";
  constant OUTPUT_out0 : STRING := "/home/shundroid/dynamatic/integration-test/subdiag_fast/out_v1/sim/HDL_OUT/output_out0.dat";
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
  signal d1_we0 : std_logic;
  signal d1_din0 : std_logic_vector(32 - 1 downto 0);
  signal d1_address0 : std_logic_vector(10 - 1 downto 0);
  signal d1_ce1 : std_logic;
  signal d1_dout1 : std_logic_vector(32 - 1 downto 0);
  signal d1_address1 : std_logic_vector(10 - 1 downto 0);
  signal d1_dout0 : std_logic_vector(32 - 1 downto 0);
  signal d1_din1 : std_logic_vector(32 - 1 downto 0) := (others => '0');
  signal d1_we1 : std_logic := '0';
  signal d1_ce0 : std_logic := '1';
  signal d2_we0 : std_logic;
  signal d2_din0 : std_logic_vector(32 - 1 downto 0);
  signal d2_address0 : std_logic_vector(10 - 1 downto 0);
  signal d2_ce1 : std_logic;
  signal d2_dout1 : std_logic_vector(32 - 1 downto 0);
  signal d2_address1 : std_logic_vector(10 - 1 downto 0);
  signal d2_dout0 : std_logic_vector(32 - 1 downto 0);
  signal d2_din1 : std_logic_vector(32 - 1 downto 0) := (others => '0');
  signal d2_we1 : std_logic := '0';
  signal d2_ce0 : std_logic := '1';
  signal e_we0 : std_logic;
  signal e_din0 : std_logic_vector(32 - 1 downto 0);
  signal e_address0 : std_logic_vector(10 - 1 downto 0);
  signal e_ce1 : std_logic;
  signal e_dout1 : std_logic_vector(32 - 1 downto 0);
  signal e_address1 : std_logic_vector(10 - 1 downto 0);
  signal e_dout0 : std_logic_vector(32 - 1 downto 0);
  signal e_din1 : std_logic_vector(32 - 1 downto 0) := (others => '0');
  signal e_we1 : std_logic := '0';
  signal e_ce0 : std_logic := '1';
  signal out0_valid : std_logic;
  signal out0_ready : std_logic;
  signal out0_ce0 : std_logic;
  signal out0_we0 : std_logic;
  signal out0_din0 : std_logic_vector(32 - 1 downto 0);
  signal out0_dout0 : std_logic_vector(32 - 1 downto 0);
  signal out0_dout0_valid : std_logic;
  signal out0_dout0_ready : std_logic;
  signal d1_end_valid : std_logic;
  signal d1_end_ready : std_logic;
  signal d2_end_valid : std_logic;
  signal d2_end_ready : std_logic;
  signal e_end_valid : std_logic;
  signal e_end_ready : std_logic;
  signal end_valid : std_logic;
  signal end_ready : std_logic;

  signal tb_temp_idle : std_logic := '1';
  shared variable transaction_idx : INTEGER := 0;
begin

  duv_inst: entity work.subdiag_fast
  port map(
    clk => tb_clk,
    d1_end_ready => d1_end_ready,
    d1_end_valid => d1_end_valid,
    d1_loadAddr => d1_address1,
    d1_loadData => d1_dout1,
    d1_loadEn => d1_ce1,
    d1_start_ready => open,
    d1_start_valid => '1',
    d1_storeAddr => d1_address0,
    d1_storeData => d1_din0,
    d1_storeEn => d1_we0,
    d2_end_ready => d2_end_ready,
    d2_end_valid => d2_end_valid,
    d2_loadAddr => d2_address1,
    d2_loadData => d2_dout1,
    d2_loadEn => d2_ce1,
    d2_start_ready => open,
    d2_start_valid => '1',
    d2_storeAddr => d2_address0,
    d2_storeData => d2_din0,
    d2_storeEn => d2_we0,
    e_end_ready => e_end_ready,
    e_end_valid => e_end_valid,
    e_loadAddr => e_address1,
    e_loadData => e_dout1,
    e_loadEn => e_ce1,
    e_start_ready => open,
    e_start_valid => '1',
    e_storeAddr => e_address0,
    e_storeData => e_din0,
    e_storeEn => e_we0,
    end_ready => end_ready,
    end_valid => end_valid,
    out0 => out0_din0,
    out0_ready => out0_ready,
    out0_valid => out0_valid,
    rst => tb_rst,
    start_ready => open,
    start_valid => '1'
  );

  mem_inst_d1: entity work.two_port_RAM
  generic map(
    TV_IN => INPUT_d1,
    TV_OUT => OUTPUT_d1,
    DATA_WIDTH => DATA_WIDTH_d1,
    ADDR_WIDTH => ADDR_WIDTH_d1,
    DEPTH => DATA_DEPTH_d1
  )
  port map(
    address0 => d1_address0,
    address1 => d1_address1,
    ce0 => '1',
    ce1 => d1_ce1,
    clk => tb_clk,
    din0 => d1_din0,
    din1 => d1_din1,
    done => tb_stop,
    dout0 => d1_dout0,
    dout1 => d1_dout1,
    rst => tb_rst,
    we0 => d1_we0,
    we1 => d1_we1
  );

  mem_inst_d2: entity work.two_port_RAM
  generic map(
    TV_IN => INPUT_d2,
    TV_OUT => OUTPUT_d2,
    DATA_WIDTH => DATA_WIDTH_d2,
    ADDR_WIDTH => ADDR_WIDTH_d2,
    DEPTH => DATA_DEPTH_d2
  )
  port map(
    address0 => d2_address0,
    address1 => d2_address1,
    ce0 => '1',
    ce1 => d2_ce1,
    clk => tb_clk,
    din0 => d2_din0,
    din1 => d2_din1,
    done => tb_stop,
    dout0 => d2_dout0,
    dout1 => d2_dout1,
    rst => tb_rst,
    we0 => d2_we0,
    we1 => d2_we1
  );

  mem_inst_e: entity work.two_port_RAM
  generic map(
    TV_IN => INPUT_e,
    TV_OUT => OUTPUT_e,
    DATA_WIDTH => DATA_WIDTH_e,
    ADDR_WIDTH => ADDR_WIDTH_e,
    DEPTH => DATA_DEPTH_e
  )
  port map(
    address0 => e_address0,
    address1 => e_address1,
    ce0 => '1',
    ce1 => e_ce1,
    clk => tb_clk,
    din0 => e_din0,
    din1 => e_din1,
    done => tb_stop,
    dout0 => e_dout0,
    dout1 => e_dout1,
    rst => tb_rst,
    we0 => e_we0,
    we1 => e_we1
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
    SIZE => 5
  )
  port map(
    ins_ready(0) => out0_ready,
    ins_ready(1) => d1_end_ready,
    ins_ready(2) => d2_end_ready,
    ins_ready(3) => e_end_ready,
    ins_ready(4) => end_ready,
    ins_valid(0) => out0_valid,
    ins_valid(1) => d1_end_valid,
    ins_valid(2) => d2_end_valid,
    ins_valid(3) => e_end_valid,
    ins_valid(4) => end_valid,
    outs_ready => tb_global_ready,
    outs_valid => tb_global_valid
  );


  write_output_transactor_d1_runtime_proc : process
    file fp             : TEXT;
    variable fstatus    : FILE_OPEN_STATUS;
    variable token_line : LINE;
    variable token      : STRING(1 to 1024);
  begin
    file_open(fstatus, fp, OUTPUT_d1 , WRITE_MODE);
    if (fstatus /= OPEN_OK) then
      assert false report "Open file " & OUTPUT_d1 & " failed!!!" severity note;
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
    file_open(fstatus, fp, OUTPUT_d1, APPEND_MODE);
    if (fstatus /= OPEN_OK) then
      assert false report "Open file " & OUTPUT_d1 & " failed!!!" severity note;
      assert false report "ERROR: Simulation using HLS TB failed." severity failure;
    end if;
    write(token_line, string'("[[[/runtime]]]"));
    writeline(fp, token_line);
    file_close(fp);
    wait;
  end process;

  write_output_transactor_d2_runtime_proc : process
    file fp             : TEXT;
    variable fstatus    : FILE_OPEN_STATUS;
    variable token_line : LINE;
    variable token      : STRING(1 to 1024);
  begin
    file_open(fstatus, fp, OUTPUT_d2 , WRITE_MODE);
    if (fstatus /= OPEN_OK) then
      assert false report "Open file " & OUTPUT_d2 & " failed!!!" severity note;
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
    file_open(fstatus, fp, OUTPUT_d2, APPEND_MODE);
    if (fstatus /= OPEN_OK) then
      assert false report "Open file " & OUTPUT_d2 & " failed!!!" severity note;
      assert false report "ERROR: Simulation using HLS TB failed." severity failure;
    end if;
    write(token_line, string'("[[[/runtime]]]"));
    writeline(fp, token_line);
    file_close(fp);
    wait;
  end process;

  write_output_transactor_e_runtime_proc : process
    file fp             : TEXT;
    variable fstatus    : FILE_OPEN_STATUS;
    variable token_line : LINE;
    variable token      : STRING(1 to 1024);
  begin
    file_open(fstatus, fp, OUTPUT_e , WRITE_MODE);
    if (fstatus /= OPEN_OK) then
      assert false report "Open file " & OUTPUT_e & " failed!!!" severity note;
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
    file_open(fstatus, fp, OUTPUT_e, APPEND_MODE);
    if (fstatus /= OPEN_OK) then
      assert false report "Open file " & OUTPUT_e & " failed!!!" severity note;
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
