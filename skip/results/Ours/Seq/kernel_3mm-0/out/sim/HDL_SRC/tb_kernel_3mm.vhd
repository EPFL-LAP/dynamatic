
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

  constant INPUT_A : STRING := "/beta/rouzbeh/hope/dynamatic/skip/runs/run_2025-12-16_00-24-33/kernel_3mm-0/out/sim/INPUT_VECTORS/input_A.dat";
  constant OUTPUT_A : STRING := "/beta/rouzbeh/hope/dynamatic/skip/runs/run_2025-12-16_00-24-33/kernel_3mm-0/out/sim/HDL_OUT/output_A.dat";
  constant DATA_WIDTH_A : INTEGER := 32;
  constant ADDR_WIDTH_A : INTEGER := 7;
  constant DATA_DEPTH_A : INTEGER := 100;
  constant INPUT_B : STRING := "/beta/rouzbeh/hope/dynamatic/skip/runs/run_2025-12-16_00-24-33/kernel_3mm-0/out/sim/INPUT_VECTORS/input_B.dat";
  constant OUTPUT_B : STRING := "/beta/rouzbeh/hope/dynamatic/skip/runs/run_2025-12-16_00-24-33/kernel_3mm-0/out/sim/HDL_OUT/output_B.dat";
  constant DATA_WIDTH_B : INTEGER := 32;
  constant ADDR_WIDTH_B : INTEGER := 7;
  constant DATA_DEPTH_B : INTEGER := 100;
  constant INPUT_C : STRING := "/beta/rouzbeh/hope/dynamatic/skip/runs/run_2025-12-16_00-24-33/kernel_3mm-0/out/sim/INPUT_VECTORS/input_C.dat";
  constant OUTPUT_C : STRING := "/beta/rouzbeh/hope/dynamatic/skip/runs/run_2025-12-16_00-24-33/kernel_3mm-0/out/sim/HDL_OUT/output_C.dat";
  constant DATA_WIDTH_C : INTEGER := 32;
  constant ADDR_WIDTH_C : INTEGER := 7;
  constant DATA_DEPTH_C : INTEGER := 100;
  constant INPUT_D : STRING := "/beta/rouzbeh/hope/dynamatic/skip/runs/run_2025-12-16_00-24-33/kernel_3mm-0/out/sim/INPUT_VECTORS/input_D.dat";
  constant OUTPUT_D : STRING := "/beta/rouzbeh/hope/dynamatic/skip/runs/run_2025-12-16_00-24-33/kernel_3mm-0/out/sim/HDL_OUT/output_D.dat";
  constant DATA_WIDTH_D : INTEGER := 32;
  constant ADDR_WIDTH_D : INTEGER := 7;
  constant DATA_DEPTH_D : INTEGER := 100;
  constant INPUT_E : STRING := "/beta/rouzbeh/hope/dynamatic/skip/runs/run_2025-12-16_00-24-33/kernel_3mm-0/out/sim/INPUT_VECTORS/input_E.dat";
  constant OUTPUT_E : STRING := "/beta/rouzbeh/hope/dynamatic/skip/runs/run_2025-12-16_00-24-33/kernel_3mm-0/out/sim/HDL_OUT/output_E.dat";
  constant DATA_WIDTH_E : INTEGER := 32;
  constant ADDR_WIDTH_E : INTEGER := 7;
  constant DATA_DEPTH_E : INTEGER := 100;
  constant INPUT_F : STRING := "/beta/rouzbeh/hope/dynamatic/skip/runs/run_2025-12-16_00-24-33/kernel_3mm-0/out/sim/INPUT_VECTORS/input_F.dat";
  constant OUTPUT_F : STRING := "/beta/rouzbeh/hope/dynamatic/skip/runs/run_2025-12-16_00-24-33/kernel_3mm-0/out/sim/HDL_OUT/output_F.dat";
  constant DATA_WIDTH_F : INTEGER := 32;
  constant ADDR_WIDTH_F : INTEGER := 7;
  constant DATA_DEPTH_F : INTEGER := 100;
  constant INPUT_G : STRING := "/beta/rouzbeh/hope/dynamatic/skip/runs/run_2025-12-16_00-24-33/kernel_3mm-0/out/sim/INPUT_VECTORS/input_G.dat";
  constant OUTPUT_G : STRING := "/beta/rouzbeh/hope/dynamatic/skip/runs/run_2025-12-16_00-24-33/kernel_3mm-0/out/sim/HDL_OUT/output_G.dat";
  constant DATA_WIDTH_G : INTEGER := 32;
  constant ADDR_WIDTH_G : INTEGER := 7;
  constant DATA_DEPTH_G : INTEGER := 100;
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
  signal A_address0 : std_logic_vector(7 - 1 downto 0);
  signal A_ce1 : std_logic;
  signal A_dout1 : std_logic_vector(32 - 1 downto 0);
  signal A_address1 : std_logic_vector(7 - 1 downto 0);
  signal A_dout0 : std_logic_vector(32 - 1 downto 0);
  signal A_din1 : std_logic_vector(32 - 1 downto 0) := (others => '0');
  signal A_we1 : std_logic := '0';
  signal A_ce0 : std_logic := '1';
  signal B_we0 : std_logic;
  signal B_din0 : std_logic_vector(32 - 1 downto 0);
  signal B_address0 : std_logic_vector(7 - 1 downto 0);
  signal B_ce1 : std_logic;
  signal B_dout1 : std_logic_vector(32 - 1 downto 0);
  signal B_address1 : std_logic_vector(7 - 1 downto 0);
  signal B_dout0 : std_logic_vector(32 - 1 downto 0);
  signal B_din1 : std_logic_vector(32 - 1 downto 0) := (others => '0');
  signal B_we1 : std_logic := '0';
  signal B_ce0 : std_logic := '1';
  signal C_we0 : std_logic;
  signal C_din0 : std_logic_vector(32 - 1 downto 0);
  signal C_address0 : std_logic_vector(7 - 1 downto 0);
  signal C_ce1 : std_logic;
  signal C_dout1 : std_logic_vector(32 - 1 downto 0);
  signal C_address1 : std_logic_vector(7 - 1 downto 0);
  signal C_dout0 : std_logic_vector(32 - 1 downto 0);
  signal C_din1 : std_logic_vector(32 - 1 downto 0) := (others => '0');
  signal C_we1 : std_logic := '0';
  signal C_ce0 : std_logic := '1';
  signal D_we0 : std_logic;
  signal D_din0 : std_logic_vector(32 - 1 downto 0);
  signal D_address0 : std_logic_vector(7 - 1 downto 0);
  signal D_ce1 : std_logic;
  signal D_dout1 : std_logic_vector(32 - 1 downto 0);
  signal D_address1 : std_logic_vector(7 - 1 downto 0);
  signal D_dout0 : std_logic_vector(32 - 1 downto 0);
  signal D_din1 : std_logic_vector(32 - 1 downto 0) := (others => '0');
  signal D_we1 : std_logic := '0';
  signal D_ce0 : std_logic := '1';
  signal E_we0 : std_logic;
  signal E_din0 : std_logic_vector(32 - 1 downto 0);
  signal E_address0 : std_logic_vector(7 - 1 downto 0);
  signal E_ce1 : std_logic;
  signal E_dout1 : std_logic_vector(32 - 1 downto 0);
  signal E_address1 : std_logic_vector(7 - 1 downto 0);
  signal E_dout0 : std_logic_vector(32 - 1 downto 0);
  signal E_din1 : std_logic_vector(32 - 1 downto 0) := (others => '0');
  signal E_we1 : std_logic := '0';
  signal E_ce0 : std_logic := '1';
  signal F_we0 : std_logic;
  signal F_din0 : std_logic_vector(32 - 1 downto 0);
  signal F_address0 : std_logic_vector(7 - 1 downto 0);
  signal F_ce1 : std_logic;
  signal F_dout1 : std_logic_vector(32 - 1 downto 0);
  signal F_address1 : std_logic_vector(7 - 1 downto 0);
  signal F_dout0 : std_logic_vector(32 - 1 downto 0);
  signal F_din1 : std_logic_vector(32 - 1 downto 0) := (others => '0');
  signal F_we1 : std_logic := '0';
  signal F_ce0 : std_logic := '1';
  signal G_we0 : std_logic;
  signal G_din0 : std_logic_vector(32 - 1 downto 0);
  signal G_address0 : std_logic_vector(7 - 1 downto 0);
  signal G_ce1 : std_logic;
  signal G_dout1 : std_logic_vector(32 - 1 downto 0);
  signal G_address1 : std_logic_vector(7 - 1 downto 0);
  signal G_dout0 : std_logic_vector(32 - 1 downto 0);
  signal G_din1 : std_logic_vector(32 - 1 downto 0) := (others => '0');
  signal G_we1 : std_logic := '0';
  signal G_ce0 : std_logic := '1';
  signal A_end_valid : std_logic;
  signal A_end_ready : std_logic;
  signal B_end_valid : std_logic;
  signal B_end_ready : std_logic;
  signal C_end_valid : std_logic;
  signal C_end_ready : std_logic;
  signal D_end_valid : std_logic;
  signal D_end_ready : std_logic;
  signal E_end_valid : std_logic;
  signal E_end_ready : std_logic;
  signal F_end_valid : std_logic;
  signal F_end_ready : std_logic;
  signal G_end_valid : std_logic;
  signal G_end_ready : std_logic;
  signal end_valid : std_logic;
  signal end_ready : std_logic;

  signal tb_temp_idle : std_logic := '1';
  shared variable transaction_idx : INTEGER := 0;
begin

  duv_inst: entity work.kernel_3mm
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
    B_end_ready => B_end_ready,
    B_end_valid => B_end_valid,
    B_loadAddr => B_address1,
    B_loadData => B_dout1,
    B_loadEn => B_ce1,
    B_start_ready => open,
    B_start_valid => '1',
    B_storeAddr => B_address0,
    B_storeData => B_din0,
    B_storeEn => B_we0,
    C_end_ready => C_end_ready,
    C_end_valid => C_end_valid,
    C_loadAddr => C_address1,
    C_loadData => C_dout1,
    C_loadEn => C_ce1,
    C_start_ready => open,
    C_start_valid => '1',
    C_storeAddr => C_address0,
    C_storeData => C_din0,
    C_storeEn => C_we0,
    D_end_ready => D_end_ready,
    D_end_valid => D_end_valid,
    D_loadAddr => D_address1,
    D_loadData => D_dout1,
    D_loadEn => D_ce1,
    D_start_ready => open,
    D_start_valid => '1',
    D_storeAddr => D_address0,
    D_storeData => D_din0,
    D_storeEn => D_we0,
    E_end_ready => E_end_ready,
    E_end_valid => E_end_valid,
    E_loadAddr => E_address1,
    E_loadData => E_dout1,
    E_loadEn => E_ce1,
    E_start_ready => open,
    E_start_valid => '1',
    E_storeAddr => E_address0,
    E_storeData => E_din0,
    E_storeEn => E_we0,
    F_end_ready => F_end_ready,
    F_end_valid => F_end_valid,
    F_loadAddr => F_address1,
    F_loadData => F_dout1,
    F_loadEn => F_ce1,
    F_start_ready => open,
    F_start_valid => '1',
    F_storeAddr => F_address0,
    F_storeData => F_din0,
    F_storeEn => F_we0,
    G_end_ready => G_end_ready,
    G_end_valid => G_end_valid,
    G_loadAddr => G_address1,
    G_loadData => G_dout1,
    G_loadEn => G_ce1,
    G_start_ready => open,
    G_start_valid => '1',
    G_storeAddr => G_address0,
    G_storeData => G_din0,
    G_storeEn => G_we0,
    clk => tb_clk,
    end_ready => end_ready,
    end_valid => end_valid,
    rst => tb_rst,
    start_ready => open,
    start_valid => '1'
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

  mem_inst_B: entity work.two_port_RAM
  generic map(
    TV_IN => INPUT_B,
    TV_OUT => OUTPUT_B,
    DATA_WIDTH => DATA_WIDTH_B,
    ADDR_WIDTH => ADDR_WIDTH_B,
    DEPTH => DATA_DEPTH_B
  )
  port map(
    address0 => B_address0,
    address1 => B_address1,
    ce0 => '1',
    ce1 => B_ce1,
    clk => tb_clk,
    din0 => B_din0,
    din1 => B_din1,
    done => tb_stop,
    dout0 => B_dout0,
    dout1 => B_dout1,
    rst => tb_rst,
    we0 => B_we0,
    we1 => B_we1
  );

  mem_inst_C: entity work.two_port_RAM
  generic map(
    TV_IN => INPUT_C,
    TV_OUT => OUTPUT_C,
    DATA_WIDTH => DATA_WIDTH_C,
    ADDR_WIDTH => ADDR_WIDTH_C,
    DEPTH => DATA_DEPTH_C
  )
  port map(
    address0 => C_address0,
    address1 => C_address1,
    ce0 => '1',
    ce1 => C_ce1,
    clk => tb_clk,
    din0 => C_din0,
    din1 => C_din1,
    done => tb_stop,
    dout0 => C_dout0,
    dout1 => C_dout1,
    rst => tb_rst,
    we0 => C_we0,
    we1 => C_we1
  );

  mem_inst_D: entity work.two_port_RAM
  generic map(
    TV_IN => INPUT_D,
    TV_OUT => OUTPUT_D,
    DATA_WIDTH => DATA_WIDTH_D,
    ADDR_WIDTH => ADDR_WIDTH_D,
    DEPTH => DATA_DEPTH_D
  )
  port map(
    address0 => D_address0,
    address1 => D_address1,
    ce0 => '1',
    ce1 => D_ce1,
    clk => tb_clk,
    din0 => D_din0,
    din1 => D_din1,
    done => tb_stop,
    dout0 => D_dout0,
    dout1 => D_dout1,
    rst => tb_rst,
    we0 => D_we0,
    we1 => D_we1
  );

  mem_inst_E: entity work.two_port_RAM
  generic map(
    TV_IN => INPUT_E,
    TV_OUT => OUTPUT_E,
    DATA_WIDTH => DATA_WIDTH_E,
    ADDR_WIDTH => ADDR_WIDTH_E,
    DEPTH => DATA_DEPTH_E
  )
  port map(
    address0 => E_address0,
    address1 => E_address1,
    ce0 => '1',
    ce1 => E_ce1,
    clk => tb_clk,
    din0 => E_din0,
    din1 => E_din1,
    done => tb_stop,
    dout0 => E_dout0,
    dout1 => E_dout1,
    rst => tb_rst,
    we0 => E_we0,
    we1 => E_we1
  );

  mem_inst_F: entity work.two_port_RAM
  generic map(
    TV_IN => INPUT_F,
    TV_OUT => OUTPUT_F,
    DATA_WIDTH => DATA_WIDTH_F,
    ADDR_WIDTH => ADDR_WIDTH_F,
    DEPTH => DATA_DEPTH_F
  )
  port map(
    address0 => F_address0,
    address1 => F_address1,
    ce0 => '1',
    ce1 => F_ce1,
    clk => tb_clk,
    din0 => F_din0,
    din1 => F_din1,
    done => tb_stop,
    dout0 => F_dout0,
    dout1 => F_dout1,
    rst => tb_rst,
    we0 => F_we0,
    we1 => F_we1
  );

  mem_inst_G: entity work.two_port_RAM
  generic map(
    TV_IN => INPUT_G,
    TV_OUT => OUTPUT_G,
    DATA_WIDTH => DATA_WIDTH_G,
    ADDR_WIDTH => ADDR_WIDTH_G,
    DEPTH => DATA_DEPTH_G
  )
  port map(
    address0 => G_address0,
    address1 => G_address1,
    ce0 => '1',
    ce1 => G_ce1,
    clk => tb_clk,
    din0 => G_din0,
    din1 => G_din1,
    done => tb_stop,
    dout0 => G_dout0,
    dout1 => G_dout1,
    rst => tb_rst,
    we0 => G_we0,
    we1 => G_we1
  );

  join_valids: entity work.tb_join
  generic map(
    SIZE => 8
  )
  port map(
    ins_ready(0) => A_end_ready,
    ins_ready(1) => B_end_ready,
    ins_ready(2) => C_end_ready,
    ins_ready(3) => D_end_ready,
    ins_ready(4) => E_end_ready,
    ins_ready(5) => F_end_ready,
    ins_ready(6) => G_end_ready,
    ins_ready(7) => end_ready,
    ins_valid(0) => A_end_valid,
    ins_valid(1) => B_end_valid,
    ins_valid(2) => C_end_valid,
    ins_valid(3) => D_end_valid,
    ins_valid(4) => E_end_valid,
    ins_valid(5) => F_end_valid,
    ins_valid(6) => G_end_valid,
    ins_valid(7) => end_valid,
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

  write_output_transactor_B_runtime_proc : process
    file fp             : TEXT;
    variable fstatus    : FILE_OPEN_STATUS;
    variable token_line : LINE;
    variable token      : STRING(1 to 1024);
  begin
    file_open(fstatus, fp, OUTPUT_B , WRITE_MODE);
    if (fstatus /= OPEN_OK) then
      assert false report "Open file " & OUTPUT_B & " failed!!!" severity note;
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
    file_open(fstatus, fp, OUTPUT_B, APPEND_MODE);
    if (fstatus /= OPEN_OK) then
      assert false report "Open file " & OUTPUT_B & " failed!!!" severity note;
      assert false report "ERROR: Simulation using HLS TB failed." severity failure;
    end if;
    write(token_line, string'("[[[/runtime]]]"));
    writeline(fp, token_line);
    file_close(fp);
    wait;
  end process;

  write_output_transactor_C_runtime_proc : process
    file fp             : TEXT;
    variable fstatus    : FILE_OPEN_STATUS;
    variable token_line : LINE;
    variable token      : STRING(1 to 1024);
  begin
    file_open(fstatus, fp, OUTPUT_C , WRITE_MODE);
    if (fstatus /= OPEN_OK) then
      assert false report "Open file " & OUTPUT_C & " failed!!!" severity note;
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
    file_open(fstatus, fp, OUTPUT_C, APPEND_MODE);
    if (fstatus /= OPEN_OK) then
      assert false report "Open file " & OUTPUT_C & " failed!!!" severity note;
      assert false report "ERROR: Simulation using HLS TB failed." severity failure;
    end if;
    write(token_line, string'("[[[/runtime]]]"));
    writeline(fp, token_line);
    file_close(fp);
    wait;
  end process;

  write_output_transactor_D_runtime_proc : process
    file fp             : TEXT;
    variable fstatus    : FILE_OPEN_STATUS;
    variable token_line : LINE;
    variable token      : STRING(1 to 1024);
  begin
    file_open(fstatus, fp, OUTPUT_D , WRITE_MODE);
    if (fstatus /= OPEN_OK) then
      assert false report "Open file " & OUTPUT_D & " failed!!!" severity note;
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
    file_open(fstatus, fp, OUTPUT_D, APPEND_MODE);
    if (fstatus /= OPEN_OK) then
      assert false report "Open file " & OUTPUT_D & " failed!!!" severity note;
      assert false report "ERROR: Simulation using HLS TB failed." severity failure;
    end if;
    write(token_line, string'("[[[/runtime]]]"));
    writeline(fp, token_line);
    file_close(fp);
    wait;
  end process;

  write_output_transactor_E_runtime_proc : process
    file fp             : TEXT;
    variable fstatus    : FILE_OPEN_STATUS;
    variable token_line : LINE;
    variable token      : STRING(1 to 1024);
  begin
    file_open(fstatus, fp, OUTPUT_E , WRITE_MODE);
    if (fstatus /= OPEN_OK) then
      assert false report "Open file " & OUTPUT_E & " failed!!!" severity note;
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
    file_open(fstatus, fp, OUTPUT_E, APPEND_MODE);
    if (fstatus /= OPEN_OK) then
      assert false report "Open file " & OUTPUT_E & " failed!!!" severity note;
      assert false report "ERROR: Simulation using HLS TB failed." severity failure;
    end if;
    write(token_line, string'("[[[/runtime]]]"));
    writeline(fp, token_line);
    file_close(fp);
    wait;
  end process;

  write_output_transactor_F_runtime_proc : process
    file fp             : TEXT;
    variable fstatus    : FILE_OPEN_STATUS;
    variable token_line : LINE;
    variable token      : STRING(1 to 1024);
  begin
    file_open(fstatus, fp, OUTPUT_F , WRITE_MODE);
    if (fstatus /= OPEN_OK) then
      assert false report "Open file " & OUTPUT_F & " failed!!!" severity note;
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
    file_open(fstatus, fp, OUTPUT_F, APPEND_MODE);
    if (fstatus /= OPEN_OK) then
      assert false report "Open file " & OUTPUT_F & " failed!!!" severity note;
      assert false report "ERROR: Simulation using HLS TB failed." severity failure;
    end if;
    write(token_line, string'("[[[/runtime]]]"));
    writeline(fp, token_line);
    file_close(fp);
    wait;
  end process;

  write_output_transactor_G_runtime_proc : process
    file fp             : TEXT;
    variable fstatus    : FILE_OPEN_STATUS;
    variable token_line : LINE;
    variable token      : STRING(1 to 1024);
  begin
    file_open(fstatus, fp, OUTPUT_G , WRITE_MODE);
    if (fstatus /= OPEN_OK) then
      assert false report "Open file " & OUTPUT_G & " failed!!!" severity note;
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
    file_open(fstatus, fp, OUTPUT_G, APPEND_MODE);
    if (fstatus /= OPEN_OK) then
      assert false report "Open file " & OUTPUT_G & " failed!!!" severity note;
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
