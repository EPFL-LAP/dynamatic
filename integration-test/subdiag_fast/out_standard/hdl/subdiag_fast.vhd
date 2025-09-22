library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity subdiag_fast is
  port (
    d1_loadData : in std_logic_vector(31 downto 0);
    d2_loadData : in std_logic_vector(31 downto 0);
    e_loadData : in std_logic_vector(31 downto 0);
    d1_start_valid : in std_logic;
    d2_start_valid : in std_logic;
    e_start_valid : in std_logic;
    start_valid : in std_logic;
    clk : in std_logic;
    rst : in std_logic;
    out0_ready : in std_logic;
    d1_end_ready : in std_logic;
    d2_end_ready : in std_logic;
    e_end_ready : in std_logic;
    end_ready : in std_logic;
    d1_start_ready : out std_logic;
    d2_start_ready : out std_logic;
    e_start_ready : out std_logic;
    start_ready : out std_logic;
    out0 : out std_logic_vector(31 downto 0);
    out0_valid : out std_logic;
    d1_end_valid : out std_logic;
    d2_end_valid : out std_logic;
    e_end_valid : out std_logic;
    end_valid : out std_logic;
    d1_loadEn : out std_logic;
    d1_loadAddr : out std_logic_vector(9 downto 0);
    d1_storeEn : out std_logic;
    d1_storeAddr : out std_logic_vector(9 downto 0);
    d1_storeData : out std_logic_vector(31 downto 0);
    d2_loadEn : out std_logic;
    d2_loadAddr : out std_logic_vector(9 downto 0);
    d2_storeEn : out std_logic;
    d2_storeAddr : out std_logic_vector(9 downto 0);
    d2_storeData : out std_logic_vector(31 downto 0);
    e_loadEn : out std_logic;
    e_loadAddr : out std_logic_vector(9 downto 0);
    e_storeEn : out std_logic;
    e_storeAddr : out std_logic_vector(9 downto 0);
    e_storeData : out std_logic_vector(31 downto 0)
  );
end entity;

architecture behavioral of subdiag_fast is

  signal fork0_outs_0_valid : std_logic;
  signal fork0_outs_0_ready : std_logic;
  signal fork0_outs_1_valid : std_logic;
  signal fork0_outs_1_ready : std_logic;
  signal fork0_outs_2_valid : std_logic;
  signal fork0_outs_2_ready : std_logic;
  signal mem_controller3_ldData_0 : std_logic_vector(31 downto 0);
  signal mem_controller3_ldData_0_valid : std_logic;
  signal mem_controller3_ldData_0_ready : std_logic;
  signal mem_controller3_memEnd_valid : std_logic;
  signal mem_controller3_memEnd_ready : std_logic;
  signal mem_controller3_loadEn : std_logic;
  signal mem_controller3_loadAddr : std_logic_vector(9 downto 0);
  signal mem_controller3_storeEn : std_logic;
  signal mem_controller3_storeAddr : std_logic_vector(9 downto 0);
  signal mem_controller3_storeData : std_logic_vector(31 downto 0);
  signal mem_controller4_ldData_0 : std_logic_vector(31 downto 0);
  signal mem_controller4_ldData_0_valid : std_logic;
  signal mem_controller4_ldData_0_ready : std_logic;
  signal mem_controller4_memEnd_valid : std_logic;
  signal mem_controller4_memEnd_ready : std_logic;
  signal mem_controller4_loadEn : std_logic;
  signal mem_controller4_loadAddr : std_logic_vector(9 downto 0);
  signal mem_controller4_storeEn : std_logic;
  signal mem_controller4_storeAddr : std_logic_vector(9 downto 0);
  signal mem_controller4_storeData : std_logic_vector(31 downto 0);
  signal mem_controller5_ldData_0 : std_logic_vector(31 downto 0);
  signal mem_controller5_ldData_0_valid : std_logic;
  signal mem_controller5_ldData_0_ready : std_logic;
  signal mem_controller5_memEnd_valid : std_logic;
  signal mem_controller5_memEnd_ready : std_logic;
  signal mem_controller5_loadEn : std_logic;
  signal mem_controller5_loadAddr : std_logic_vector(9 downto 0);
  signal mem_controller5_storeEn : std_logic;
  signal mem_controller5_storeAddr : std_logic_vector(9 downto 0);
  signal mem_controller5_storeData : std_logic_vector(31 downto 0);
  signal constant0_outs : std_logic_vector(0 downto 0);
  signal constant0_outs_valid : std_logic;
  signal constant0_outs_ready : std_logic;
  signal extsi3_outs : std_logic_vector(10 downto 0);
  signal extsi3_outs_valid : std_logic;
  signal extsi3_outs_ready : std_logic;
  signal mux0_outs : std_logic_vector(10 downto 0);
  signal mux0_outs_valid : std_logic;
  signal mux0_outs_ready : std_logic;
  signal buffer0_outs : std_logic_vector(10 downto 0);
  signal buffer0_outs_valid : std_logic;
  signal buffer0_outs_ready : std_logic;
  signal fork1_outs_0 : std_logic_vector(10 downto 0);
  signal fork1_outs_0_valid : std_logic;
  signal fork1_outs_0_ready : std_logic;
  signal fork1_outs_1 : std_logic_vector(10 downto 0);
  signal fork1_outs_1_valid : std_logic;
  signal fork1_outs_1_ready : std_logic;
  signal fork1_outs_2 : std_logic_vector(10 downto 0);
  signal fork1_outs_2_valid : std_logic;
  signal fork1_outs_2_ready : std_logic;
  signal fork1_outs_3 : std_logic_vector(10 downto 0);
  signal fork1_outs_3_valid : std_logic;
  signal fork1_outs_3_ready : std_logic;
  signal trunci0_outs : std_logic_vector(9 downto 0);
  signal trunci0_outs_valid : std_logic;
  signal trunci0_outs_ready : std_logic;
  signal trunci1_outs : std_logic_vector(9 downto 0);
  signal trunci1_outs_valid : std_logic;
  signal trunci1_outs_ready : std_logic;
  signal trunci2_outs : std_logic_vector(9 downto 0);
  signal trunci2_outs_valid : std_logic;
  signal trunci2_outs_ready : std_logic;
  signal control_merge0_outs_valid : std_logic;
  signal control_merge0_outs_ready : std_logic;
  signal control_merge0_index : std_logic_vector(0 downto 0);
  signal control_merge0_index_valid : std_logic;
  signal control_merge0_index_ready : std_logic;
  signal source0_outs_valid : std_logic;
  signal source0_outs_ready : std_logic;
  signal constant4_outs : std_logic_vector(31 downto 0);
  signal constant4_outs_valid : std_logic;
  signal constant4_outs_ready : std_logic;
  signal load0_addrOut : std_logic_vector(9 downto 0);
  signal load0_addrOut_valid : std_logic;
  signal load0_addrOut_ready : std_logic;
  signal load0_dataOut : std_logic_vector(31 downto 0);
  signal load0_dataOut_valid : std_logic;
  signal load0_dataOut_ready : std_logic;
  signal load1_addrOut : std_logic_vector(9 downto 0);
  signal load1_addrOut_valid : std_logic;
  signal load1_addrOut_ready : std_logic;
  signal load1_dataOut : std_logic_vector(31 downto 0);
  signal load1_dataOut_valid : std_logic;
  signal load1_dataOut_ready : std_logic;
  signal addf0_result : std_logic_vector(31 downto 0);
  signal addf0_result_valid : std_logic;
  signal addf0_result_ready : std_logic;
  signal load2_addrOut : std_logic_vector(9 downto 0);
  signal load2_addrOut_valid : std_logic;
  signal load2_addrOut_ready : std_logic;
  signal load2_dataOut : std_logic_vector(31 downto 0);
  signal load2_dataOut_valid : std_logic;
  signal load2_dataOut_ready : std_logic;
  signal mulf0_result : std_logic_vector(31 downto 0);
  signal mulf0_result_valid : std_logic;
  signal mulf0_result_ready : std_logic;
  signal buffer3_outs : std_logic_vector(31 downto 0);
  signal buffer3_outs_valid : std_logic;
  signal buffer3_outs_ready : std_logic;
  signal cmpf0_result : std_logic_vector(0 downto 0);
  signal cmpf0_result_valid : std_logic;
  signal cmpf0_result_ready : std_logic;
  signal fork2_outs_0 : std_logic_vector(0 downto 0);
  signal fork2_outs_0_valid : std_logic;
  signal fork2_outs_0_ready : std_logic;
  signal fork2_outs_1 : std_logic_vector(0 downto 0);
  signal fork2_outs_1_valid : std_logic;
  signal fork2_outs_1_ready : std_logic;
  signal buffer1_outs : std_logic_vector(10 downto 0);
  signal buffer1_outs_valid : std_logic;
  signal buffer1_outs_ready : std_logic;
  signal cond_br2_trueOut : std_logic_vector(10 downto 0);
  signal cond_br2_trueOut_valid : std_logic;
  signal cond_br2_trueOut_ready : std_logic;
  signal cond_br2_falseOut : std_logic_vector(10 downto 0);
  signal cond_br2_falseOut_valid : std_logic;
  signal cond_br2_falseOut_ready : std_logic;
  signal extsi4_outs : std_logic_vector(11 downto 0);
  signal extsi4_outs_valid : std_logic;
  signal extsi4_outs_ready : std_logic;
  signal buffer2_outs_valid : std_logic;
  signal buffer2_outs_ready : std_logic;
  signal cond_br3_trueOut_valid : std_logic;
  signal cond_br3_trueOut_ready : std_logic;
  signal cond_br3_falseOut_valid : std_logic;
  signal cond_br3_falseOut_ready : std_logic;
  signal extsi5_outs : std_logic_vector(11 downto 0);
  signal extsi5_outs_valid : std_logic;
  signal extsi5_outs_ready : std_logic;
  signal source1_outs_valid : std_logic;
  signal source1_outs_ready : std_logic;
  signal constant1_outs : std_logic_vector(1 downto 0);
  signal constant1_outs_valid : std_logic;
  signal constant1_outs_ready : std_logic;
  signal extsi6_outs : std_logic_vector(11 downto 0);
  signal extsi6_outs_valid : std_logic;
  signal extsi6_outs_ready : std_logic;
  signal source2_outs_valid : std_logic;
  signal source2_outs_ready : std_logic;
  signal constant2_outs : std_logic_vector(10 downto 0);
  signal constant2_outs_valid : std_logic;
  signal constant2_outs_ready : std_logic;
  signal extsi7_outs : std_logic_vector(11 downto 0);
  signal extsi7_outs_valid : std_logic;
  signal extsi7_outs_ready : std_logic;
  signal addi0_result : std_logic_vector(11 downto 0);
  signal addi0_result_valid : std_logic;
  signal addi0_result_ready : std_logic;
  signal fork3_outs_0 : std_logic_vector(11 downto 0);
  signal fork3_outs_0_valid : std_logic;
  signal fork3_outs_0_ready : std_logic;
  signal fork3_outs_1 : std_logic_vector(11 downto 0);
  signal fork3_outs_1_valid : std_logic;
  signal fork3_outs_1_ready : std_logic;
  signal cmpi0_result : std_logic_vector(0 downto 0);
  signal cmpi0_result_valid : std_logic;
  signal cmpi0_result_ready : std_logic;
  signal fork4_outs_0 : std_logic_vector(0 downto 0);
  signal fork4_outs_0_valid : std_logic;
  signal fork4_outs_0_ready : std_logic;
  signal fork4_outs_1 : std_logic_vector(0 downto 0);
  signal fork4_outs_1_valid : std_logic;
  signal fork4_outs_1_ready : std_logic;
  signal cond_br4_trueOut : std_logic_vector(11 downto 0);
  signal cond_br4_trueOut_valid : std_logic;
  signal cond_br4_trueOut_ready : std_logic;
  signal cond_br4_falseOut : std_logic_vector(11 downto 0);
  signal cond_br4_falseOut_valid : std_logic;
  signal cond_br4_falseOut_ready : std_logic;
  signal trunci3_outs : std_logic_vector(10 downto 0);
  signal trunci3_outs_valid : std_logic;
  signal trunci3_outs_ready : std_logic;
  signal cond_br5_trueOut_valid : std_logic;
  signal cond_br5_trueOut_ready : std_logic;
  signal cond_br5_falseOut_valid : std_logic;
  signal cond_br5_falseOut_ready : std_logic;
  signal mux1_outs : std_logic_vector(11 downto 0);
  signal mux1_outs_valid : std_logic;
  signal mux1_outs_ready : std_logic;
  signal extsi8_outs : std_logic_vector(31 downto 0);
  signal extsi8_outs_valid : std_logic;
  signal extsi8_outs_ready : std_logic;
  signal control_merge2_outs_valid : std_logic;
  signal control_merge2_outs_ready : std_logic;
  signal control_merge2_index : std_logic_vector(0 downto 0);
  signal control_merge2_index_valid : std_logic;
  signal control_merge2_index_ready : std_logic;
  signal fork5_outs_0_valid : std_logic;
  signal fork5_outs_0_ready : std_logic;
  signal fork5_outs_1_valid : std_logic;
  signal fork5_outs_1_ready : std_logic;
  signal fork5_outs_2_valid : std_logic;
  signal fork5_outs_2_ready : std_logic;

begin

  out0 <= extsi8_outs;
  out0_valid <= extsi8_outs_valid;
  extsi8_outs_ready <= out0_ready;
  d1_end_valid <= mem_controller5_memEnd_valid;
  mem_controller5_memEnd_ready <= d1_end_ready;
  d2_end_valid <= mem_controller4_memEnd_valid;
  mem_controller4_memEnd_ready <= d2_end_ready;
  e_end_valid <= mem_controller3_memEnd_valid;
  mem_controller3_memEnd_ready <= e_end_ready;
  end_valid <= fork0_outs_1_valid;
  fork0_outs_1_ready <= end_ready;
  d1_loadEn <= mem_controller5_loadEn;
  d1_loadAddr <= mem_controller5_loadAddr;
  d1_storeEn <= mem_controller5_storeEn;
  d1_storeAddr <= mem_controller5_storeAddr;
  d1_storeData <= mem_controller5_storeData;
  d2_loadEn <= mem_controller4_loadEn;
  d2_loadAddr <= mem_controller4_loadAddr;
  d2_storeEn <= mem_controller4_storeEn;
  d2_storeAddr <= mem_controller4_storeAddr;
  d2_storeData <= mem_controller4_storeData;
  e_loadEn <= mem_controller3_loadEn;
  e_loadAddr <= mem_controller3_loadAddr;
  e_storeEn <= mem_controller3_storeEn;
  e_storeAddr <= mem_controller3_storeAddr;
  e_storeData <= mem_controller3_storeData;

  fork0 : entity work.handshake_fork_0(arch)
    port map(
      ins_valid => start_valid,
      ins_ready => start_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork0_outs_0_valid,
      outs_valid(1) => fork0_outs_1_valid,
      outs_valid(2) => fork0_outs_2_valid,
      outs_ready(0) => fork0_outs_0_ready,
      outs_ready(1) => fork0_outs_1_ready,
      outs_ready(2) => fork0_outs_2_ready
    );

  mem_controller3 : entity work.handshake_mem_controller_0(arch)
    port map(
      loadData => e_loadData,
      memStart_valid => e_start_valid,
      memStart_ready => e_start_ready,
      ldAddr(0) => load2_addrOut,
      ldAddr_valid(0) => load2_addrOut_valid,
      ldAddr_ready(0) => load2_addrOut_ready,
      ctrlEnd_valid => fork5_outs_2_valid,
      ctrlEnd_ready => fork5_outs_2_ready,
      clk => clk,
      rst => rst,
      ldData(0) => mem_controller3_ldData_0,
      ldData_valid(0) => mem_controller3_ldData_0_valid,
      ldData_ready(0) => mem_controller3_ldData_0_ready,
      memEnd_valid => mem_controller3_memEnd_valid,
      memEnd_ready => mem_controller3_memEnd_ready,
      loadEn => mem_controller3_loadEn,
      loadAddr => mem_controller3_loadAddr,
      storeEn => mem_controller3_storeEn,
      storeAddr => mem_controller3_storeAddr,
      storeData => mem_controller3_storeData
    );

  mem_controller4 : entity work.handshake_mem_controller_1(arch)
    port map(
      loadData => d2_loadData,
      memStart_valid => d2_start_valid,
      memStart_ready => d2_start_ready,
      ldAddr(0) => load1_addrOut,
      ldAddr_valid(0) => load1_addrOut_valid,
      ldAddr_ready(0) => load1_addrOut_ready,
      ctrlEnd_valid => fork5_outs_1_valid,
      ctrlEnd_ready => fork5_outs_1_ready,
      clk => clk,
      rst => rst,
      ldData(0) => mem_controller4_ldData_0,
      ldData_valid(0) => mem_controller4_ldData_0_valid,
      ldData_ready(0) => mem_controller4_ldData_0_ready,
      memEnd_valid => mem_controller4_memEnd_valid,
      memEnd_ready => mem_controller4_memEnd_ready,
      loadEn => mem_controller4_loadEn,
      loadAddr => mem_controller4_loadAddr,
      storeEn => mem_controller4_storeEn,
      storeAddr => mem_controller4_storeAddr,
      storeData => mem_controller4_storeData
    );

  mem_controller5 : entity work.handshake_mem_controller_2(arch)
    port map(
      loadData => d1_loadData,
      memStart_valid => d1_start_valid,
      memStart_ready => d1_start_ready,
      ldAddr(0) => load0_addrOut,
      ldAddr_valid(0) => load0_addrOut_valid,
      ldAddr_ready(0) => load0_addrOut_ready,
      ctrlEnd_valid => fork5_outs_0_valid,
      ctrlEnd_ready => fork5_outs_0_ready,
      clk => clk,
      rst => rst,
      ldData(0) => mem_controller5_ldData_0,
      ldData_valid(0) => mem_controller5_ldData_0_valid,
      ldData_ready(0) => mem_controller5_ldData_0_ready,
      memEnd_valid => mem_controller5_memEnd_valid,
      memEnd_ready => mem_controller5_memEnd_ready,
      loadEn => mem_controller5_loadEn,
      loadAddr => mem_controller5_loadAddr,
      storeEn => mem_controller5_storeEn,
      storeAddr => mem_controller5_storeAddr,
      storeData => mem_controller5_storeData
    );

  constant0 : entity work.handshake_constant_0(arch)
    port map(
      ctrl_valid => fork0_outs_0_valid,
      ctrl_ready => fork0_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => constant0_outs,
      outs_valid => constant0_outs_valid,
      outs_ready => constant0_outs_ready
    );

  extsi3 : entity work.handshake_extsi_0(arch)
    port map(
      ins => constant0_outs,
      ins_valid => constant0_outs_valid,
      ins_ready => constant0_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi3_outs,
      outs_valid => extsi3_outs_valid,
      outs_ready => extsi3_outs_ready
    );

  mux0 : entity work.handshake_mux_0(arch)
    port map(
      index => control_merge0_index,
      index_valid => control_merge0_index_valid,
      index_ready => control_merge0_index_ready,
      ins(0) => extsi3_outs,
      ins(1) => trunci3_outs,
      ins_valid(0) => extsi3_outs_valid,
      ins_valid(1) => trunci3_outs_valid,
      ins_ready(0) => extsi3_outs_ready,
      ins_ready(1) => trunci3_outs_ready,
      clk => clk,
      rst => rst,
      outs => mux0_outs,
      outs_valid => mux0_outs_valid,
      outs_ready => mux0_outs_ready
    );

  buffer0 : entity work.handshake_buffer_0(arch)
    port map(
      ins => mux0_outs,
      ins_valid => mux0_outs_valid,
      ins_ready => mux0_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer0_outs,
      outs_valid => buffer0_outs_valid,
      outs_ready => buffer0_outs_ready
    );

  fork1 : entity work.handshake_fork_1(arch)
    port map(
      ins => buffer0_outs,
      ins_valid => buffer0_outs_valid,
      ins_ready => buffer0_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork1_outs_0,
      outs(1) => fork1_outs_1,
      outs(2) => fork1_outs_2,
      outs(3) => fork1_outs_3,
      outs_valid(0) => fork1_outs_0_valid,
      outs_valid(1) => fork1_outs_1_valid,
      outs_valid(2) => fork1_outs_2_valid,
      outs_valid(3) => fork1_outs_3_valid,
      outs_ready(0) => fork1_outs_0_ready,
      outs_ready(1) => fork1_outs_1_ready,
      outs_ready(2) => fork1_outs_2_ready,
      outs_ready(3) => fork1_outs_3_ready
    );

  trunci0 : entity work.handshake_trunci_0(arch)
    port map(
      ins => fork1_outs_0,
      ins_valid => fork1_outs_0_valid,
      ins_ready => fork1_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => trunci0_outs,
      outs_valid => trunci0_outs_valid,
      outs_ready => trunci0_outs_ready
    );

  trunci1 : entity work.handshake_trunci_0(arch)
    port map(
      ins => fork1_outs_1,
      ins_valid => fork1_outs_1_valid,
      ins_ready => fork1_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => trunci1_outs,
      outs_valid => trunci1_outs_valid,
      outs_ready => trunci1_outs_ready
    );

  trunci2 : entity work.handshake_trunci_0(arch)
    port map(
      ins => fork1_outs_2,
      ins_valid => fork1_outs_2_valid,
      ins_ready => fork1_outs_2_ready,
      clk => clk,
      rst => rst,
      outs => trunci2_outs,
      outs_valid => trunci2_outs_valid,
      outs_ready => trunci2_outs_ready
    );

  control_merge0 : entity work.handshake_control_merge_0(arch)
    port map(
      ins_valid(0) => fork0_outs_2_valid,
      ins_valid(1) => cond_br5_trueOut_valid,
      ins_ready(0) => fork0_outs_2_ready,
      ins_ready(1) => cond_br5_trueOut_ready,
      clk => clk,
      rst => rst,
      outs_valid => control_merge0_outs_valid,
      outs_ready => control_merge0_outs_ready,
      index => control_merge0_index,
      index_valid => control_merge0_index_valid,
      index_ready => control_merge0_index_ready
    );

  source0 : entity work.handshake_source_0(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source0_outs_valid,
      outs_ready => source0_outs_ready
    );

  constant4 : entity work.handshake_constant_1(arch)
    port map(
      ctrl_valid => source0_outs_valid,
      ctrl_ready => source0_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant4_outs,
      outs_valid => constant4_outs_valid,
      outs_ready => constant4_outs_ready
    );

  load0 : entity work.handshake_load_0(arch)
    port map(
      addrIn => trunci2_outs,
      addrIn_valid => trunci2_outs_valid,
      addrIn_ready => trunci2_outs_ready,
      dataFromMem => mem_controller5_ldData_0,
      dataFromMem_valid => mem_controller5_ldData_0_valid,
      dataFromMem_ready => mem_controller5_ldData_0_ready,
      clk => clk,
      rst => rst,
      addrOut => load0_addrOut,
      addrOut_valid => load0_addrOut_valid,
      addrOut_ready => load0_addrOut_ready,
      dataOut => load0_dataOut,
      dataOut_valid => load0_dataOut_valid,
      dataOut_ready => load0_dataOut_ready
    );

  load1 : entity work.handshake_load_1(arch)
    port map(
      addrIn => trunci1_outs,
      addrIn_valid => trunci1_outs_valid,
      addrIn_ready => trunci1_outs_ready,
      dataFromMem => mem_controller4_ldData_0,
      dataFromMem_valid => mem_controller4_ldData_0_valid,
      dataFromMem_ready => mem_controller4_ldData_0_ready,
      clk => clk,
      rst => rst,
      addrOut => load1_addrOut,
      addrOut_valid => load1_addrOut_valid,
      addrOut_ready => load1_addrOut_ready,
      dataOut => load1_dataOut,
      dataOut_valid => load1_dataOut_valid,
      dataOut_ready => load1_dataOut_ready
    );

  addf0 : entity work.handshake_addf_0(arch)
    port map(
      lhs => load0_dataOut,
      lhs_valid => load0_dataOut_valid,
      lhs_ready => load0_dataOut_ready,
      rhs => load1_dataOut,
      rhs_valid => load1_dataOut_valid,
      rhs_ready => load1_dataOut_ready,
      clk => clk,
      rst => rst,
      result => addf0_result,
      result_valid => addf0_result_valid,
      result_ready => addf0_result_ready
    );

  load2 : entity work.handshake_load_2(arch)
    port map(
      addrIn => trunci0_outs,
      addrIn_valid => trunci0_outs_valid,
      addrIn_ready => trunci0_outs_ready,
      dataFromMem => mem_controller3_ldData_0,
      dataFromMem_valid => mem_controller3_ldData_0_valid,
      dataFromMem_ready => mem_controller3_ldData_0_ready,
      clk => clk,
      rst => rst,
      addrOut => load2_addrOut,
      addrOut_valid => load2_addrOut_valid,
      addrOut_ready => load2_addrOut_ready,
      dataOut => load2_dataOut,
      dataOut_valid => load2_dataOut_valid,
      dataOut_ready => load2_dataOut_ready
    );

  mulf0 : entity work.handshake_mulf_0(arch)
    port map(
      lhs => addf0_result,
      lhs_valid => addf0_result_valid,
      lhs_ready => addf0_result_ready,
      rhs => constant4_outs,
      rhs_valid => constant4_outs_valid,
      rhs_ready => constant4_outs_ready,
      clk => clk,
      rst => rst,
      result => mulf0_result,
      result_valid => mulf0_result_valid,
      result_ready => mulf0_result_ready
    );

  buffer3 : entity work.handshake_buffer_1(arch)
    port map(
      ins => load2_dataOut,
      ins_valid => load2_dataOut_valid,
      ins_ready => load2_dataOut_ready,
      clk => clk,
      rst => rst,
      outs => buffer3_outs,
      outs_valid => buffer3_outs_valid,
      outs_ready => buffer3_outs_ready
    );

  cmpf0 : entity work.handshake_cmpf_0(arch)
    port map(
      lhs => buffer3_outs,
      lhs_valid => buffer3_outs_valid,
      lhs_ready => buffer3_outs_ready,
      rhs => mulf0_result,
      rhs_valid => mulf0_result_valid,
      rhs_ready => mulf0_result_ready,
      clk => clk,
      rst => rst,
      result => cmpf0_result,
      result_valid => cmpf0_result_valid,
      result_ready => cmpf0_result_ready
    );

  fork2 : entity work.handshake_fork_2(arch)
    port map(
      ins => cmpf0_result,
      ins_valid => cmpf0_result_valid,
      ins_ready => cmpf0_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork2_outs_0,
      outs(1) => fork2_outs_1,
      outs_valid(0) => fork2_outs_0_valid,
      outs_valid(1) => fork2_outs_1_valid,
      outs_ready(0) => fork2_outs_0_ready,
      outs_ready(1) => fork2_outs_1_ready
    );

  buffer1 : entity work.handshake_buffer_2(arch)
    port map(
      ins => fork1_outs_3,
      ins_valid => fork1_outs_3_valid,
      ins_ready => fork1_outs_3_ready,
      clk => clk,
      rst => rst,
      outs => buffer1_outs,
      outs_valid => buffer1_outs_valid,
      outs_ready => buffer1_outs_ready
    );

  cond_br2 : entity work.handshake_cond_br_0(arch)
    port map(
      condition => fork2_outs_0,
      condition_valid => fork2_outs_0_valid,
      condition_ready => fork2_outs_0_ready,
      data => buffer1_outs,
      data_valid => buffer1_outs_valid,
      data_ready => buffer1_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br2_trueOut,
      trueOut_valid => cond_br2_trueOut_valid,
      trueOut_ready => cond_br2_trueOut_ready,
      falseOut => cond_br2_falseOut,
      falseOut_valid => cond_br2_falseOut_valid,
      falseOut_ready => cond_br2_falseOut_ready
    );

  extsi4 : entity work.handshake_extsi_1(arch)
    port map(
      ins => cond_br2_falseOut,
      ins_valid => cond_br2_falseOut_valid,
      ins_ready => cond_br2_falseOut_ready,
      clk => clk,
      rst => rst,
      outs => extsi4_outs,
      outs_valid => extsi4_outs_valid,
      outs_ready => extsi4_outs_ready
    );

  buffer2 : entity work.handshake_buffer_3(arch)
    port map(
      ins_valid => control_merge0_outs_valid,
      ins_ready => control_merge0_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer2_outs_valid,
      outs_ready => buffer2_outs_ready
    );

  cond_br3 : entity work.handshake_cond_br_1(arch)
    port map(
      condition => fork2_outs_1,
      condition_valid => fork2_outs_1_valid,
      condition_ready => fork2_outs_1_ready,
      data_valid => buffer2_outs_valid,
      data_ready => buffer2_outs_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br3_trueOut_valid,
      trueOut_ready => cond_br3_trueOut_ready,
      falseOut_valid => cond_br3_falseOut_valid,
      falseOut_ready => cond_br3_falseOut_ready
    );

  extsi5 : entity work.handshake_extsi_1(arch)
    port map(
      ins => cond_br2_trueOut,
      ins_valid => cond_br2_trueOut_valid,
      ins_ready => cond_br2_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => extsi5_outs,
      outs_valid => extsi5_outs_valid,
      outs_ready => extsi5_outs_ready
    );

  source1 : entity work.handshake_source_0(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source1_outs_valid,
      outs_ready => source1_outs_ready
    );

  constant1 : entity work.handshake_constant_2(arch)
    port map(
      ctrl_valid => source1_outs_valid,
      ctrl_ready => source1_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant1_outs,
      outs_valid => constant1_outs_valid,
      outs_ready => constant1_outs_ready
    );

  extsi6 : entity work.handshake_extsi_2(arch)
    port map(
      ins => constant1_outs,
      ins_valid => constant1_outs_valid,
      ins_ready => constant1_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi6_outs,
      outs_valid => extsi6_outs_valid,
      outs_ready => extsi6_outs_ready
    );

  source2 : entity work.handshake_source_0(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source2_outs_valid,
      outs_ready => source2_outs_ready
    );

  constant2 : entity work.handshake_constant_3(arch)
    port map(
      ctrl_valid => source2_outs_valid,
      ctrl_ready => source2_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant2_outs,
      outs_valid => constant2_outs_valid,
      outs_ready => constant2_outs_ready
    );

  extsi7 : entity work.handshake_extsi_1(arch)
    port map(
      ins => constant2_outs,
      ins_valid => constant2_outs_valid,
      ins_ready => constant2_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi7_outs,
      outs_valid => extsi7_outs_valid,
      outs_ready => extsi7_outs_ready
    );

  addi0 : entity work.handshake_addi_0(arch)
    port map(
      lhs => extsi5_outs,
      lhs_valid => extsi5_outs_valid,
      lhs_ready => extsi5_outs_ready,
      rhs => extsi6_outs,
      rhs_valid => extsi6_outs_valid,
      rhs_ready => extsi6_outs_ready,
      clk => clk,
      rst => rst,
      result => addi0_result,
      result_valid => addi0_result_valid,
      result_ready => addi0_result_ready
    );

  fork3 : entity work.handshake_fork_3(arch)
    port map(
      ins => addi0_result,
      ins_valid => addi0_result_valid,
      ins_ready => addi0_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork3_outs_0,
      outs(1) => fork3_outs_1,
      outs_valid(0) => fork3_outs_0_valid,
      outs_valid(1) => fork3_outs_1_valid,
      outs_ready(0) => fork3_outs_0_ready,
      outs_ready(1) => fork3_outs_1_ready
    );

  cmpi0 : entity work.handshake_cmpi_0(arch)
    port map(
      lhs => fork3_outs_1,
      lhs_valid => fork3_outs_1_valid,
      lhs_ready => fork3_outs_1_ready,
      rhs => extsi7_outs,
      rhs_valid => extsi7_outs_valid,
      rhs_ready => extsi7_outs_ready,
      clk => clk,
      rst => rst,
      result => cmpi0_result,
      result_valid => cmpi0_result_valid,
      result_ready => cmpi0_result_ready
    );

  fork4 : entity work.handshake_fork_2(arch)
    port map(
      ins => cmpi0_result,
      ins_valid => cmpi0_result_valid,
      ins_ready => cmpi0_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork4_outs_0,
      outs(1) => fork4_outs_1,
      outs_valid(0) => fork4_outs_0_valid,
      outs_valid(1) => fork4_outs_1_valid,
      outs_ready(0) => fork4_outs_0_ready,
      outs_ready(1) => fork4_outs_1_ready
    );

  cond_br4 : entity work.handshake_cond_br_2(arch)
    port map(
      condition => fork4_outs_0,
      condition_valid => fork4_outs_0_valid,
      condition_ready => fork4_outs_0_ready,
      data => fork3_outs_0,
      data_valid => fork3_outs_0_valid,
      data_ready => fork3_outs_0_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br4_trueOut,
      trueOut_valid => cond_br4_trueOut_valid,
      trueOut_ready => cond_br4_trueOut_ready,
      falseOut => cond_br4_falseOut,
      falseOut_valid => cond_br4_falseOut_valid,
      falseOut_ready => cond_br4_falseOut_ready
    );

  trunci3 : entity work.handshake_trunci_1(arch)
    port map(
      ins => cond_br4_trueOut,
      ins_valid => cond_br4_trueOut_valid,
      ins_ready => cond_br4_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => trunci3_outs,
      outs_valid => trunci3_outs_valid,
      outs_ready => trunci3_outs_ready
    );

  cond_br5 : entity work.handshake_cond_br_1(arch)
    port map(
      condition => fork4_outs_1,
      condition_valid => fork4_outs_1_valid,
      condition_ready => fork4_outs_1_ready,
      data_valid => cond_br3_trueOut_valid,
      data_ready => cond_br3_trueOut_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br5_trueOut_valid,
      trueOut_ready => cond_br5_trueOut_ready,
      falseOut_valid => cond_br5_falseOut_valid,
      falseOut_ready => cond_br5_falseOut_ready
    );

  mux1 : entity work.handshake_mux_1(arch)
    port map(
      index => control_merge2_index,
      index_valid => control_merge2_index_valid,
      index_ready => control_merge2_index_ready,
      ins(0) => extsi4_outs,
      ins(1) => cond_br4_falseOut,
      ins_valid(0) => extsi4_outs_valid,
      ins_valid(1) => cond_br4_falseOut_valid,
      ins_ready(0) => extsi4_outs_ready,
      ins_ready(1) => cond_br4_falseOut_ready,
      clk => clk,
      rst => rst,
      outs => mux1_outs,
      outs_valid => mux1_outs_valid,
      outs_ready => mux1_outs_ready
    );

  extsi8 : entity work.handshake_extsi_3(arch)
    port map(
      ins => mux1_outs,
      ins_valid => mux1_outs_valid,
      ins_ready => mux1_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi8_outs,
      outs_valid => extsi8_outs_valid,
      outs_ready => extsi8_outs_ready
    );

  control_merge2 : entity work.handshake_control_merge_0(arch)
    port map(
      ins_valid(0) => cond_br3_falseOut_valid,
      ins_valid(1) => cond_br5_falseOut_valid,
      ins_ready(0) => cond_br3_falseOut_ready,
      ins_ready(1) => cond_br5_falseOut_ready,
      clk => clk,
      rst => rst,
      outs_valid => control_merge2_outs_valid,
      outs_ready => control_merge2_outs_ready,
      index => control_merge2_index,
      index_valid => control_merge2_index_valid,
      index_ready => control_merge2_index_ready
    );

  fork5 : entity work.handshake_fork_0(arch)
    port map(
      ins_valid => control_merge2_outs_valid,
      ins_ready => control_merge2_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork5_outs_0_valid,
      outs_valid(1) => fork5_outs_1_valid,
      outs_valid(2) => fork5_outs_2_valid,
      outs_ready(0) => fork5_outs_0_ready,
      outs_ready(1) => fork5_outs_1_ready,
      outs_ready(2) => fork5_outs_2_ready
    );

end architecture;
