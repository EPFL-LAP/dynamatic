
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

  constant INPUT_a : STRING := "/home/shundroid/dynamatic/integration-test/nested_loop/out/sim/INPUT_VECTORS/input_a.dat";
  constant OUTPUT_a : STRING := "/home/shundroid/dynamatic/integration-test/nested_loop/out/sim/HDL_OUT/output_a.dat";
  constant DATA_WIDTH_a : INTEGER := 32;
  constant ADDR_WIDTH_a : INTEGER := 10;
  constant DATA_DEPTH_a : INTEGER := 1000;
  constant INPUT_b : STRING := "/home/shundroid/dynamatic/integration-test/nested_loop/out/sim/INPUT_VECTORS/input_b.dat";
  constant OUTPUT_b : STRING := "/home/shundroid/dynamatic/integration-test/nested_loop/out/sim/HDL_OUT/output_b.dat";
  constant DATA_WIDTH_b : INTEGER := 32;
  constant ADDR_WIDTH_b : INTEGER := 10;
  constant DATA_DEPTH_b : INTEGER := 1000;
  constant INPUT_c : STRING := "/home/shundroid/dynamatic/integration-test/nested_loop/out/sim/INPUT_VECTORS/input_c.dat";
  constant OUTPUT_c : STRING := "/home/shundroid/dynamatic/integration-test/nested_loop/out/sim/HDL_OUT/output_c.dat";
  constant DATA_WIDTH_c : INTEGER := 32;
  constant ADDR_WIDTH_c : INTEGER := 10;
  constant DATA_DEPTH_c : INTEGER := 1000;
  constant HALF_CLK_PERIOD : TIME := 2 ns;
  constant RESET_PERIOD : TIME := 10 ns;
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
  signal b_we0 : std_logic;
  signal b_din0 : std_logic_vector(32 - 1 downto 0);
  signal b_address0 : std_logic_vector(10 - 1 downto 0);
  signal b_ce1 : std_logic;
  signal b_dout1 : std_logic_vector(32 - 1 downto 0);
  signal b_address1 : std_logic_vector(10 - 1 downto 0);
  signal b_dout0 : std_logic_vector(32 - 1 downto 0);
  signal b_din1 : std_logic_vector(32 - 1 downto 0) := (others => '0');
  signal b_we1 : std_logic := '0';
  signal b_ce0 : std_logic := '1';
  signal c_we0 : std_logic;
  signal c_din0 : std_logic_vector(32 - 1 downto 0);
  signal c_address0 : std_logic_vector(10 - 1 downto 0);
  signal c_ce1 : std_logic;
  signal c_dout1 : std_logic_vector(32 - 1 downto 0);
  signal c_address1 : std_logic_vector(10 - 1 downto 0);
  signal c_dout0 : std_logic_vector(32 - 1 downto 0);
  signal c_din1 : std_logic_vector(32 - 1 downto 0) := (others => '0');
  signal c_we1 : std_logic := '0';
  signal c_ce0 : std_logic := '1';
  signal a_end_valid : std_logic;
  signal a_end_ready : std_logic;
  signal b_end_valid : std_logic;
  signal b_end_ready : std_logic;
  signal c_end_valid : std_logic;
  signal c_end_ready : std_logic;
  signal end_valid : std_logic;
  signal end_ready : std_logic;

  signal tb_temp_idle : std_logic := '1';
  shared variable transaction_idx : INTEGER := 0;
begin

  duv_inst: entity work.nested_loop
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
    b_end_ready => b_end_ready,
    b_end_valid => b_end_valid,
    b_loadAddr => b_address1,
    b_loadData => b_dout1,
    b_loadEn => b_ce1,
    b_start_ready => open,
    b_start_valid => '1',
    b_storeAddr => b_address0,
    b_storeData => b_din0,
    b_storeEn => b_we0,
    c_end_ready => c_end_ready,
    c_end_valid => c_end_valid,
    c_loadAddr => c_address1,
    c_loadData => c_dout1,
    c_loadEn => c_ce1,
    c_start_ready => open,
    c_start_valid => '1',
    c_storeAddr => c_address0,
    c_storeData => c_din0,
    c_storeEn => c_we0,
    clk => tb_clk,
    end_ready => end_ready,
    end_valid => end_valid,
    rst => tb_rst,
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

  mem_inst_b: entity work.two_port_RAM
  generic map(
    TV_IN => INPUT_b,
    TV_OUT => OUTPUT_b,
    DATA_WIDTH => DATA_WIDTH_b,
    ADDR_WIDTH => ADDR_WIDTH_b,
    DEPTH => DATA_DEPTH_b
  )
  port map(
    address0 => b_address0,
    address1 => b_address1,
    ce0 => '1',
    ce1 => b_ce1,
    clk => tb_clk,
    din0 => b_din0,
    din1 => b_din1,
    done => tb_stop,
    dout0 => b_dout0,
    dout1 => b_dout1,
    rst => tb_rst,
    we0 => b_we0,
    we1 => b_we1
  );

  mem_inst_c: entity work.two_port_RAM
  generic map(
    TV_IN => INPUT_c,
    TV_OUT => OUTPUT_c,
    DATA_WIDTH => DATA_WIDTH_c,
    ADDR_WIDTH => ADDR_WIDTH_c,
    DEPTH => DATA_DEPTH_c
  )
  port map(
    address0 => c_address0,
    address1 => c_address1,
    ce0 => '1',
    ce1 => c_ce1,
    clk => tb_clk,
    din0 => c_din0,
    din1 => c_din1,
    done => tb_stop,
    dout0 => c_dout0,
    dout1 => c_dout1,
    rst => tb_rst,
    we0 => c_we0,
    we1 => c_we1
  );

  join_valids: entity work.tb_join
  generic map(
    SIZE => 4
  )
  port map(
    ins_ready(0) => a_end_ready,
    ins_ready(1) => b_end_ready,
    ins_ready(2) => c_end_ready,
    ins_ready(3) => end_ready,
    ins_valid(0) => a_end_valid,
    ins_valid(1) => b_end_valid,
    ins_valid(2) => c_end_valid,
    ins_valid(3) => end_valid,
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

  write_output_transactor_c_runtime_proc : process
    file fp             : TEXT;
    variable fstatus    : FILE_OPEN_STATUS;
    variable token_line : LINE;
    variable token      : STRING(1 to 1024);
  begin
    file_open(fstatus, fp, OUTPUT_c , WRITE_MODE);
    if (fstatus /= OPEN_OK) then
      assert false report "Open file " & OUTPUT_c & " failed!!!" severity note;
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
    file_open(fstatus, fp, OUTPUT_c, APPEND_MODE);
    if (fstatus /= OPEN_OK) then
      assert false report "Open file " & OUTPUT_c & " failed!!!" severity note;
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
    wait for RESET_PERIOD;
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
