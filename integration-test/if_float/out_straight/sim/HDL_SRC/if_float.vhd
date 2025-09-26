library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity if_float is
  port (
    x0 : in std_logic_vector(31 downto 0);
    x0_valid : in std_logic;
    a_loadData : in std_logic_vector(31 downto 0);
    minus_trace_loadData : in std_logic_vector(31 downto 0);
    a_start_valid : in std_logic;
    minus_trace_start_valid : in std_logic;
    start_valid : in std_logic;
    clk : in std_logic;
    rst : in std_logic;
    out0_ready : in std_logic;
    a_end_ready : in std_logic;
    minus_trace_end_ready : in std_logic;
    end_ready : in std_logic;
    x0_ready : out std_logic;
    a_start_ready : out std_logic;
    minus_trace_start_ready : out std_logic;
    start_ready : out std_logic;
    out0 : out std_logic_vector(31 downto 0);
    out0_valid : out std_logic;
    a_end_valid : out std_logic;
    minus_trace_end_valid : out std_logic;
    end_valid : out std_logic;
    a_loadEn : out std_logic;
    a_loadAddr : out std_logic_vector(6 downto 0);
    a_storeEn : out std_logic;
    a_storeAddr : out std_logic_vector(6 downto 0);
    a_storeData : out std_logic_vector(31 downto 0);
    minus_trace_loadEn : out std_logic;
    minus_trace_loadAddr : out std_logic_vector(6 downto 0);
    minus_trace_storeEn : out std_logic;
    minus_trace_storeAddr : out std_logic_vector(6 downto 0);
    minus_trace_storeData : out std_logic_vector(31 downto 0)
  );
end entity;

architecture behavioral of if_float is

  signal fork0_outs_0_valid : std_logic;
  signal fork0_outs_0_ready : std_logic;
  signal fork0_outs_1_valid : std_logic;
  signal fork0_outs_1_ready : std_logic;
  signal fork0_outs_2_valid : std_logic;
  signal fork0_outs_2_ready : std_logic;
  signal mem_controller2_memEnd_valid : std_logic;
  signal mem_controller2_memEnd_ready : std_logic;
  signal mem_controller2_loadEn : std_logic;
  signal mem_controller2_loadAddr : std_logic_vector(6 downto 0);
  signal mem_controller2_storeEn : std_logic;
  signal mem_controller2_storeAddr : std_logic_vector(6 downto 0);
  signal mem_controller2_storeData : std_logic_vector(31 downto 0);
  signal mem_controller3_ldData_0 : std_logic_vector(31 downto 0);
  signal mem_controller3_ldData_0_valid : std_logic;
  signal mem_controller3_ldData_0_ready : std_logic;
  signal mem_controller3_memEnd_valid : std_logic;
  signal mem_controller3_memEnd_ready : std_logic;
  signal mem_controller3_loadEn : std_logic;
  signal mem_controller3_loadAddr : std_logic_vector(6 downto 0);
  signal mem_controller3_storeEn : std_logic;
  signal mem_controller3_storeAddr : std_logic_vector(6 downto 0);
  signal mem_controller3_storeData : std_logic_vector(31 downto 0);
  signal constant2_outs : std_logic_vector(0 downto 0);
  signal constant2_outs_valid : std_logic;
  signal constant2_outs_ready : std_logic;
  signal extsi4_outs : std_logic_vector(7 downto 0);
  signal extsi4_outs_valid : std_logic;
  signal extsi4_outs_ready : std_logic;
  signal mux0_outs : std_logic_vector(7 downto 0);
  signal mux0_outs_valid : std_logic;
  signal mux0_outs_ready : std_logic;
  signal buffer0_outs : std_logic_vector(7 downto 0);
  signal buffer0_outs_valid : std_logic;
  signal buffer0_outs_ready : std_logic;
  signal buffer1_outs : std_logic_vector(7 downto 0);
  signal buffer1_outs_valid : std_logic;
  signal buffer1_outs_ready : std_logic;
  signal fork14_outs_0 : std_logic_vector(7 downto 0);
  signal fork14_outs_0_valid : std_logic;
  signal fork14_outs_0_ready : std_logic;
  signal fork14_outs_1 : std_logic_vector(7 downto 0);
  signal fork14_outs_1_valid : std_logic;
  signal fork14_outs_1_ready : std_logic;
  signal fork14_outs_2 : std_logic_vector(7 downto 0);
  signal fork14_outs_2_valid : std_logic;
  signal fork14_outs_2_ready : std_logic;
  signal trunci0_outs : std_logic_vector(6 downto 0);
  signal trunci0_outs_valid : std_logic;
  signal trunci0_outs_ready : std_logic;
  signal buffer6_outs : std_logic_vector(0 downto 0);
  signal buffer6_outs_valid : std_logic;
  signal buffer6_outs_ready : std_logic;
  signal mux1_outs : std_logic_vector(31 downto 0);
  signal mux1_outs_valid : std_logic;
  signal mux1_outs_ready : std_logic;
  signal buffer4_outs : std_logic_vector(31 downto 0);
  signal buffer4_outs_valid : std_logic;
  signal buffer4_outs_ready : std_logic;
  signal fork15_outs_0 : std_logic_vector(31 downto 0);
  signal fork15_outs_0_valid : std_logic;
  signal fork15_outs_0_ready : std_logic;
  signal fork15_outs_1 : std_logic_vector(31 downto 0);
  signal fork15_outs_1_valid : std_logic;
  signal fork15_outs_1_ready : std_logic;
  signal fork15_outs_2 : std_logic_vector(31 downto 0);
  signal fork15_outs_2_valid : std_logic;
  signal fork15_outs_2_ready : std_logic;
  signal fork15_outs_3 : std_logic_vector(31 downto 0);
  signal fork15_outs_3_valid : std_logic;
  signal fork15_outs_3_ready : std_logic;
  signal control_merge0_outs_valid : std_logic;
  signal control_merge0_outs_ready : std_logic;
  signal control_merge0_index : std_logic_vector(0 downto 0);
  signal control_merge0_index_valid : std_logic;
  signal control_merge0_index_ready : std_logic;
  signal buffer5_outs_valid : std_logic;
  signal buffer5_outs_ready : std_logic;
  signal fork16_outs_0_valid : std_logic;
  signal fork16_outs_0_ready : std_logic;
  signal fork16_outs_1_valid : std_logic;
  signal fork16_outs_1_ready : std_logic;
  signal fork3_outs_0 : std_logic_vector(0 downto 0);
  signal fork3_outs_0_valid : std_logic;
  signal fork3_outs_0_ready : std_logic;
  signal fork3_outs_1 : std_logic_vector(0 downto 0);
  signal fork3_outs_1_valid : std_logic;
  signal fork3_outs_1_ready : std_logic;
  signal source0_outs_valid : std_logic;
  signal source0_outs_ready : std_logic;
  signal constant6_outs : std_logic_vector(31 downto 0);
  signal constant6_outs_valid : std_logic;
  signal constant6_outs_ready : std_logic;
  signal source1_outs_valid : std_logic;
  signal source1_outs_ready : std_logic;
  signal constant7_outs : std_logic_vector(31 downto 0);
  signal constant7_outs_valid : std_logic;
  signal constant7_outs_ready : std_logic;
  signal load0_addrOut : std_logic_vector(6 downto 0);
  signal load0_addrOut_valid : std_logic;
  signal load0_addrOut_ready : std_logic;
  signal load0_dataOut : std_logic_vector(31 downto 0);
  signal load0_dataOut_valid : std_logic;
  signal load0_dataOut_ready : std_logic;
  signal mulf0_result : std_logic_vector(31 downto 0);
  signal mulf0_result_valid : std_logic;
  signal mulf0_result_ready : std_logic;
  signal mulf1_result : std_logic_vector(31 downto 0);
  signal mulf1_result_valid : std_logic;
  signal mulf1_result_ready : std_logic;
  signal addf0_result : std_logic_vector(31 downto 0);
  signal addf0_result_valid : std_logic;
  signal addf0_result_ready : std_logic;
  signal cmpf0_result : std_logic_vector(0 downto 0);
  signal cmpf0_result_valid : std_logic;
  signal cmpf0_result_ready : std_logic;
  signal fork37_outs_0 : std_logic_vector(0 downto 0);
  signal fork37_outs_0_valid : std_logic;
  signal fork37_outs_0_ready : std_logic;
  signal fork37_outs_1 : std_logic_vector(0 downto 0);
  signal fork37_outs_1_valid : std_logic;
  signal fork37_outs_1_ready : std_logic;
  signal fork37_outs_2 : std_logic_vector(0 downto 0);
  signal fork37_outs_2_valid : std_logic;
  signal fork37_outs_2_ready : std_logic;
  signal fork37_outs_3 : std_logic_vector(0 downto 0);
  signal fork37_outs_3_valid : std_logic;
  signal fork37_outs_3_ready : std_logic;
  signal fork37_outs_4 : std_logic_vector(0 downto 0);
  signal fork37_outs_4_valid : std_logic;
  signal fork37_outs_4_ready : std_logic;
  signal fork37_outs_5 : std_logic_vector(0 downto 0);
  signal fork37_outs_5_valid : std_logic;
  signal fork37_outs_5_ready : std_logic;
  signal fork37_outs_6 : std_logic_vector(0 downto 0);
  signal fork37_outs_6_valid : std_logic;
  signal fork37_outs_6_ready : std_logic;
  signal fork37_outs_7 : std_logic_vector(0 downto 0);
  signal fork37_outs_7_valid : std_logic;
  signal fork37_outs_7_ready : std_logic;
  signal fork37_outs_8 : std_logic_vector(0 downto 0);
  signal fork37_outs_8_valid : std_logic;
  signal fork37_outs_8_ready : std_logic;
  signal fork37_outs_9 : std_logic_vector(0 downto 0);
  signal fork37_outs_9_valid : std_logic;
  signal fork37_outs_9_ready : std_logic;
  signal buffer8_outs : std_logic_vector(0 downto 0);
  signal buffer8_outs_valid : std_logic;
  signal buffer8_outs_ready : std_logic;
  signal not0_outs : std_logic_vector(0 downto 0);
  signal not0_outs_valid : std_logic;
  signal not0_outs_ready : std_logic;
  signal fork38_outs_0 : std_logic_vector(0 downto 0);
  signal fork38_outs_0_valid : std_logic;
  signal fork38_outs_0_ready : std_logic;
  signal fork38_outs_1 : std_logic_vector(0 downto 0);
  signal fork38_outs_1_valid : std_logic;
  signal fork38_outs_1_ready : std_logic;
  signal fork38_outs_2 : std_logic_vector(0 downto 0);
  signal fork38_outs_2_valid : std_logic;
  signal fork38_outs_2_ready : std_logic;
  signal buffer2_outs : std_logic_vector(7 downto 0);
  signal buffer2_outs_valid : std_logic;
  signal buffer2_outs_ready : std_logic;
  signal buffer3_outs : std_logic_vector(7 downto 0);
  signal buffer3_outs_valid : std_logic;
  signal buffer3_outs_ready : std_logic;
  signal source2_outs_valid : std_logic;
  signal source2_outs_ready : std_logic;
  signal constant8_outs : std_logic_vector(31 downto 0);
  signal constant8_outs_valid : std_logic;
  signal constant8_outs_ready : std_logic;
  signal mulf2_result : std_logic_vector(31 downto 0);
  signal mulf2_result_valid : std_logic;
  signal mulf2_result_ready : std_logic;
  signal passer6_result : std_logic_vector(31 downto 0);
  signal passer6_result_valid : std_logic;
  signal passer6_result_ready : std_logic;
  signal buffer9_outs : std_logic_vector(31 downto 0);
  signal buffer9_outs_valid : std_logic;
  signal buffer9_outs_ready : std_logic;
  signal passer7_result : std_logic_vector(7 downto 0);
  signal passer7_result_valid : std_logic;
  signal passer7_result_ready : std_logic;
  signal buffer10_outs_valid : std_logic;
  signal buffer10_outs_ready : std_logic;
  signal passer8_result_valid : std_logic;
  signal passer8_result_ready : std_logic;
  signal buffer11_outs : std_logic_vector(7 downto 0);
  signal buffer11_outs_valid : std_logic;
  signal buffer11_outs_ready : std_logic;
  signal fork34_outs_0 : std_logic_vector(7 downto 0);
  signal fork34_outs_0_valid : std_logic;
  signal fork34_outs_0_ready : std_logic;
  signal fork34_outs_1 : std_logic_vector(7 downto 0);
  signal fork34_outs_1_valid : std_logic;
  signal fork34_outs_1_ready : std_logic;
  signal passer9_result : std_logic_vector(6 downto 0);
  signal passer9_result_valid : std_logic;
  signal passer9_result_ready : std_logic;
  signal trunci1_outs : std_logic_vector(6 downto 0);
  signal trunci1_outs_valid : std_logic;
  signal trunci1_outs_ready : std_logic;
  signal buffer12_outs : std_logic_vector(31 downto 0);
  signal buffer12_outs_valid : std_logic;
  signal buffer12_outs_ready : std_logic;
  signal passer10_result : std_logic_vector(31 downto 0);
  signal passer10_result_valid : std_logic;
  signal passer10_result_ready : std_logic;
  signal fork35_outs_0 : std_logic_vector(31 downto 0);
  signal fork35_outs_0_valid : std_logic;
  signal fork35_outs_0_ready : std_logic;
  signal fork35_outs_1 : std_logic_vector(31 downto 0);
  signal fork35_outs_1_valid : std_logic;
  signal fork35_outs_1_ready : std_logic;
  signal fork36_outs_0_valid : std_logic;
  signal fork36_outs_0_ready : std_logic;
  signal fork36_outs_1_valid : std_logic;
  signal fork36_outs_1_ready : std_logic;
  signal constant3_outs : std_logic_vector(1 downto 0);
  signal constant3_outs_valid : std_logic;
  signal constant3_outs_ready : std_logic;
  signal passer11_result : std_logic_vector(31 downto 0);
  signal passer11_result_valid : std_logic;
  signal passer11_result_ready : std_logic;
  signal extsi1_outs : std_logic_vector(31 downto 0);
  signal extsi1_outs_valid : std_logic;
  signal extsi1_outs_ready : std_logic;
  signal source3_outs_valid : std_logic;
  signal source3_outs_ready : std_logic;
  signal constant9_outs : std_logic_vector(31 downto 0);
  signal constant9_outs_valid : std_logic;
  signal constant9_outs_ready : std_logic;
  signal store0_addrOut : std_logic_vector(6 downto 0);
  signal store0_addrOut_valid : std_logic;
  signal store0_addrOut_ready : std_logic;
  signal store0_dataToMem : std_logic_vector(31 downto 0);
  signal store0_dataToMem_valid : std_logic;
  signal store0_dataToMem_ready : std_logic;
  signal divf0_result : std_logic_vector(31 downto 0);
  signal divf0_result_valid : std_logic;
  signal divf0_result_ready : std_logic;
  signal buffer7_outs : std_logic_vector(0 downto 0);
  signal buffer7_outs_valid : std_logic;
  signal buffer7_outs_ready : std_logic;
  signal passer12_result : std_logic_vector(31 downto 0);
  signal passer12_result_valid : std_logic;
  signal passer12_result_ready : std_logic;
  signal passer13_result : std_logic_vector(7 downto 0);
  signal passer13_result_valid : std_logic;
  signal passer13_result_ready : std_logic;
  signal passer14_result_valid : std_logic;
  signal passer14_result_ready : std_logic;
  signal buffer13_outs : std_logic_vector(31 downto 0);
  signal buffer13_outs_valid : std_logic;
  signal buffer13_outs_ready : std_logic;
  signal mux2_outs : std_logic_vector(31 downto 0);
  signal mux2_outs_valid : std_logic;
  signal mux2_outs_ready : std_logic;
  signal buffer16_outs : std_logic_vector(31 downto 0);
  signal buffer16_outs_valid : std_logic;
  signal buffer16_outs_ready : std_logic;
  signal fork8_outs_0 : std_logic_vector(31 downto 0);
  signal fork8_outs_0_valid : std_logic;
  signal fork8_outs_0_ready : std_logic;
  signal fork8_outs_1 : std_logic_vector(31 downto 0);
  signal fork8_outs_1_valid : std_logic;
  signal fork8_outs_1_ready : std_logic;
  signal buffer14_outs : std_logic_vector(7 downto 0);
  signal buffer14_outs_valid : std_logic;
  signal buffer14_outs_ready : std_logic;
  signal mux3_outs : std_logic_vector(7 downto 0);
  signal mux3_outs_valid : std_logic;
  signal mux3_outs_ready : std_logic;
  signal buffer17_outs : std_logic_vector(7 downto 0);
  signal buffer17_outs_valid : std_logic;
  signal buffer17_outs_ready : std_logic;
  signal extsi5_outs : std_logic_vector(8 downto 0);
  signal extsi5_outs_valid : std_logic;
  signal extsi5_outs_ready : std_logic;
  signal buffer15_outs_valid : std_logic;
  signal buffer15_outs_ready : std_logic;
  signal mux4_outs_valid : std_logic;
  signal mux4_outs_ready : std_logic;
  signal source4_outs_valid : std_logic;
  signal source4_outs_ready : std_logic;
  signal constant4_outs : std_logic_vector(1 downto 0);
  signal constant4_outs_valid : std_logic;
  signal constant4_outs_ready : std_logic;
  signal extsi6_outs : std_logic_vector(8 downto 0);
  signal extsi6_outs_valid : std_logic;
  signal extsi6_outs_ready : std_logic;
  signal source5_outs_valid : std_logic;
  signal source5_outs_ready : std_logic;
  signal constant12_outs : std_logic_vector(7 downto 0);
  signal constant12_outs_valid : std_logic;
  signal constant12_outs_ready : std_logic;
  signal extsi7_outs : std_logic_vector(8 downto 0);
  signal extsi7_outs_valid : std_logic;
  signal extsi7_outs_ready : std_logic;
  signal addf1_result : std_logic_vector(31 downto 0);
  signal addf1_result_valid : std_logic;
  signal addf1_result_ready : std_logic;
  signal addi0_result : std_logic_vector(8 downto 0);
  signal addi0_result_valid : std_logic;
  signal addi0_result_ready : std_logic;
  signal fork10_outs_0 : std_logic_vector(8 downto 0);
  signal fork10_outs_0_valid : std_logic;
  signal fork10_outs_0_ready : std_logic;
  signal fork10_outs_1 : std_logic_vector(8 downto 0);
  signal fork10_outs_1_valid : std_logic;
  signal fork10_outs_1_ready : std_logic;
  signal trunci2_outs : std_logic_vector(7 downto 0);
  signal trunci2_outs_valid : std_logic;
  signal trunci2_outs_ready : std_logic;
  signal cmpi0_result : std_logic_vector(0 downto 0);
  signal cmpi0_result_valid : std_logic;
  signal cmpi0_result_ready : std_logic;
  signal fork11_outs_0 : std_logic_vector(0 downto 0);
  signal fork11_outs_0_valid : std_logic;
  signal fork11_outs_0_ready : std_logic;
  signal fork11_outs_1 : std_logic_vector(0 downto 0);
  signal fork11_outs_1_valid : std_logic;
  signal fork11_outs_1_ready : std_logic;
  signal fork11_outs_2 : std_logic_vector(0 downto 0);
  signal fork11_outs_2_valid : std_logic;
  signal fork11_outs_2_ready : std_logic;
  signal cond_br5_trueOut : std_logic_vector(7 downto 0);
  signal cond_br5_trueOut_valid : std_logic;
  signal cond_br5_trueOut_ready : std_logic;
  signal cond_br5_falseOut : std_logic_vector(7 downto 0);
  signal cond_br5_falseOut_valid : std_logic;
  signal cond_br5_falseOut_ready : std_logic;
  signal buffer19_outs : std_logic_vector(0 downto 0);
  signal buffer19_outs_valid : std_logic;
  signal buffer19_outs_ready : std_logic;
  signal cond_br6_trueOut : std_logic_vector(31 downto 0);
  signal cond_br6_trueOut_valid : std_logic;
  signal cond_br6_trueOut_ready : std_logic;
  signal cond_br6_falseOut : std_logic_vector(31 downto 0);
  signal cond_br6_falseOut_valid : std_logic;
  signal cond_br6_falseOut_ready : std_logic;
  signal buffer18_outs_valid : std_logic;
  signal buffer18_outs_ready : std_logic;
  signal cond_br7_trueOut_valid : std_logic;
  signal cond_br7_trueOut_ready : std_logic;
  signal cond_br7_falseOut_valid : std_logic;
  signal cond_br7_falseOut_ready : std_logic;
  signal fork12_outs_0_valid : std_logic;
  signal fork12_outs_0_ready : std_logic;
  signal fork12_outs_1_valid : std_logic;
  signal fork12_outs_1_ready : std_logic;

begin

  out0 <= cond_br6_falseOut;
  out0_valid <= cond_br6_falseOut_valid;
  cond_br6_falseOut_ready <= out0_ready;
  a_end_valid <= mem_controller3_memEnd_valid;
  mem_controller3_memEnd_ready <= a_end_ready;
  minus_trace_end_valid <= mem_controller2_memEnd_valid;
  mem_controller2_memEnd_ready <= minus_trace_end_ready;
  end_valid <= fork0_outs_1_valid;
  fork0_outs_1_ready <= end_ready;
  a_loadEn <= mem_controller3_loadEn;
  a_loadAddr <= mem_controller3_loadAddr;
  a_storeEn <= mem_controller3_storeEn;
  a_storeAddr <= mem_controller3_storeAddr;
  a_storeData <= mem_controller3_storeData;
  minus_trace_loadEn <= mem_controller2_loadEn;
  minus_trace_loadAddr <= mem_controller2_loadAddr;
  minus_trace_storeEn <= mem_controller2_storeEn;
  minus_trace_storeAddr <= mem_controller2_storeAddr;
  minus_trace_storeData <= mem_controller2_storeData;

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

  mem_controller2 : entity work.handshake_mem_controller_0(arch)
    port map(
      loadData => minus_trace_loadData,
      memStart_valid => minus_trace_start_valid,
      memStart_ready => minus_trace_start_ready,
      ctrl(0) => passer11_result,
      ctrl_valid(0) => passer11_result_valid,
      ctrl_ready(0) => passer11_result_ready,
      stAddr(0) => store0_addrOut,
      stAddr_valid(0) => store0_addrOut_valid,
      stAddr_ready(0) => store0_addrOut_ready,
      stData(0) => store0_dataToMem,
      stData_valid(0) => store0_dataToMem_valid,
      stData_ready(0) => store0_dataToMem_ready,
      ctrlEnd_valid => fork12_outs_1_valid,
      ctrlEnd_ready => fork12_outs_1_ready,
      clk => clk,
      rst => rst,
      memEnd_valid => mem_controller2_memEnd_valid,
      memEnd_ready => mem_controller2_memEnd_ready,
      loadEn => mem_controller2_loadEn,
      loadAddr => mem_controller2_loadAddr,
      storeEn => mem_controller2_storeEn,
      storeAddr => mem_controller2_storeAddr,
      storeData => mem_controller2_storeData
    );

  mem_controller3 : entity work.handshake_mem_controller_1(arch)
    port map(
      loadData => a_loadData,
      memStart_valid => a_start_valid,
      memStart_ready => a_start_ready,
      ldAddr(0) => load0_addrOut,
      ldAddr_valid(0) => load0_addrOut_valid,
      ldAddr_ready(0) => load0_addrOut_ready,
      ctrlEnd_valid => fork12_outs_0_valid,
      ctrlEnd_ready => fork12_outs_0_ready,
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

  constant2 : entity work.handshake_constant_0(arch)
    port map(
      ctrl_valid => fork0_outs_0_valid,
      ctrl_ready => fork0_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => constant2_outs,
      outs_valid => constant2_outs_valid,
      outs_ready => constant2_outs_ready
    );

  extsi4 : entity work.handshake_extsi_0(arch)
    port map(
      ins => constant2_outs,
      ins_valid => constant2_outs_valid,
      ins_ready => constant2_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi4_outs,
      outs_valid => extsi4_outs_valid,
      outs_ready => extsi4_outs_ready
    );

  mux0 : entity work.handshake_mux_0(arch)
    port map(
      index => fork3_outs_0,
      index_valid => fork3_outs_0_valid,
      index_ready => fork3_outs_0_ready,
      ins(0) => extsi4_outs,
      ins(1) => cond_br5_trueOut,
      ins_valid(0) => extsi4_outs_valid,
      ins_valid(1) => cond_br5_trueOut_valid,
      ins_ready(0) => extsi4_outs_ready,
      ins_ready(1) => cond_br5_trueOut_ready,
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

  buffer1 : entity work.handshake_buffer_1(arch)
    port map(
      ins => buffer0_outs,
      ins_valid => buffer0_outs_valid,
      ins_ready => buffer0_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer1_outs,
      outs_valid => buffer1_outs_valid,
      outs_ready => buffer1_outs_ready
    );

  fork14 : entity work.handshake_fork_1(arch)
    port map(
      ins => buffer1_outs,
      ins_valid => buffer1_outs_valid,
      ins_ready => buffer1_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork14_outs_0,
      outs(1) => fork14_outs_1,
      outs(2) => fork14_outs_2,
      outs_valid(0) => fork14_outs_0_valid,
      outs_valid(1) => fork14_outs_1_valid,
      outs_valid(2) => fork14_outs_2_valid,
      outs_ready(0) => fork14_outs_0_ready,
      outs_ready(1) => fork14_outs_1_ready,
      outs_ready(2) => fork14_outs_2_ready
    );

  trunci0 : entity work.handshake_trunci_0(arch)
    port map(
      ins => fork14_outs_0,
      ins_valid => fork14_outs_0_valid,
      ins_ready => fork14_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => trunci0_outs,
      outs_valid => trunci0_outs_valid,
      outs_ready => trunci0_outs_ready
    );

  buffer6 : entity work.handshake_buffer_2(arch)
    port map(
      ins => fork3_outs_1,
      ins_valid => fork3_outs_1_valid,
      ins_ready => fork3_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => buffer6_outs,
      outs_valid => buffer6_outs_valid,
      outs_ready => buffer6_outs_ready
    );

  mux1 : entity work.handshake_mux_1(arch)
    port map(
      index => buffer6_outs,
      index_valid => buffer6_outs_valid,
      index_ready => buffer6_outs_ready,
      ins(0) => x0,
      ins(1) => cond_br6_trueOut,
      ins_valid(0) => x0_valid,
      ins_valid(1) => cond_br6_trueOut_valid,
      ins_ready(0) => x0_ready,
      ins_ready(1) => cond_br6_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux1_outs,
      outs_valid => mux1_outs_valid,
      outs_ready => mux1_outs_ready
    );

  buffer4 : entity work.handshake_buffer_3(arch)
    port map(
      ins => mux1_outs,
      ins_valid => mux1_outs_valid,
      ins_ready => mux1_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer4_outs,
      outs_valid => buffer4_outs_valid,
      outs_ready => buffer4_outs_ready
    );

  fork15 : entity work.handshake_fork_2(arch)
    port map(
      ins => buffer4_outs,
      ins_valid => buffer4_outs_valid,
      ins_ready => buffer4_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork15_outs_0,
      outs(1) => fork15_outs_1,
      outs(2) => fork15_outs_2,
      outs(3) => fork15_outs_3,
      outs_valid(0) => fork15_outs_0_valid,
      outs_valid(1) => fork15_outs_1_valid,
      outs_valid(2) => fork15_outs_2_valid,
      outs_valid(3) => fork15_outs_3_valid,
      outs_ready(0) => fork15_outs_0_ready,
      outs_ready(1) => fork15_outs_1_ready,
      outs_ready(2) => fork15_outs_2_ready,
      outs_ready(3) => fork15_outs_3_ready
    );

  control_merge0 : entity work.handshake_control_merge_0(arch)
    port map(
      ins_valid(0) => fork0_outs_2_valid,
      ins_valid(1) => cond_br7_trueOut_valid,
      ins_ready(0) => fork0_outs_2_ready,
      ins_ready(1) => cond_br7_trueOut_ready,
      clk => clk,
      rst => rst,
      outs_valid => control_merge0_outs_valid,
      outs_ready => control_merge0_outs_ready,
      index => control_merge0_index,
      index_valid => control_merge0_index_valid,
      index_ready => control_merge0_index_ready
    );

  buffer5 : entity work.handshake_buffer_4(arch)
    port map(
      ins_valid => control_merge0_outs_valid,
      ins_ready => control_merge0_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer5_outs_valid,
      outs_ready => buffer5_outs_ready
    );

  fork16 : entity work.handshake_fork_3(arch)
    port map(
      ins_valid => buffer5_outs_valid,
      ins_ready => buffer5_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork16_outs_0_valid,
      outs_valid(1) => fork16_outs_1_valid,
      outs_ready(0) => fork16_outs_0_ready,
      outs_ready(1) => fork16_outs_1_ready
    );

  fork3 : entity work.handshake_fork_4(arch)
    port map(
      ins => control_merge0_index,
      ins_valid => control_merge0_index_valid,
      ins_ready => control_merge0_index_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork3_outs_0,
      outs(1) => fork3_outs_1,
      outs_valid(0) => fork3_outs_0_valid,
      outs_valid(1) => fork3_outs_1_valid,
      outs_ready(0) => fork3_outs_0_ready,
      outs_ready(1) => fork3_outs_1_ready
    );

  source0 : entity work.handshake_source_0(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source0_outs_valid,
      outs_ready => source0_outs_ready
    );

  constant6 : entity work.handshake_constant_1(arch)
    port map(
      ctrl_valid => source0_outs_valid,
      ctrl_ready => source0_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant6_outs,
      outs_valid => constant6_outs_valid,
      outs_ready => constant6_outs_ready
    );

  source1 : entity work.handshake_source_0(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source1_outs_valid,
      outs_ready => source1_outs_ready
    );

  constant7 : entity work.handshake_constant_2(arch)
    port map(
      ctrl_valid => source1_outs_valid,
      ctrl_ready => source1_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant7_outs,
      outs_valid => constant7_outs_valid,
      outs_ready => constant7_outs_ready
    );

  load0 : entity work.handshake_load_0(arch)
    port map(
      addrIn => trunci0_outs,
      addrIn_valid => trunci0_outs_valid,
      addrIn_ready => trunci0_outs_ready,
      dataFromMem => mem_controller3_ldData_0,
      dataFromMem_valid => mem_controller3_ldData_0_valid,
      dataFromMem_ready => mem_controller3_ldData_0_ready,
      clk => clk,
      rst => rst,
      addrOut => load0_addrOut,
      addrOut_valid => load0_addrOut_valid,
      addrOut_ready => load0_addrOut_ready,
      dataOut => load0_dataOut,
      dataOut_valid => load0_dataOut_valid,
      dataOut_ready => load0_dataOut_ready
    );

  mulf0 : entity work.handshake_mulf_0(arch)
    port map(
      lhs => load0_dataOut,
      lhs_valid => load0_dataOut_valid,
      lhs_ready => load0_dataOut_ready,
      rhs => fork15_outs_1,
      rhs_valid => fork15_outs_1_valid,
      rhs_ready => fork15_outs_1_ready,
      clk => clk,
      rst => rst,
      result => mulf0_result,
      result_valid => mulf0_result_valid,
      result_ready => mulf0_result_ready
    );

  mulf1 : entity work.handshake_mulf_1(arch)
    port map(
      lhs => fork15_outs_0,
      lhs_valid => fork15_outs_0_valid,
      lhs_ready => fork15_outs_0_ready,
      rhs => constant6_outs,
      rhs_valid => constant6_outs_valid,
      rhs_ready => constant6_outs_ready,
      clk => clk,
      rst => rst,
      result => mulf1_result,
      result_valid => mulf1_result_valid,
      result_ready => mulf1_result_ready
    );

  addf0 : entity work.handshake_addf_0(arch)
    port map(
      lhs => mulf0_result,
      lhs_valid => mulf0_result_valid,
      lhs_ready => mulf0_result_ready,
      rhs => mulf1_result,
      rhs_valid => mulf1_result_valid,
      rhs_ready => mulf1_result_ready,
      clk => clk,
      rst => rst,
      result => addf0_result,
      result_valid => addf0_result_valid,
      result_ready => addf0_result_ready
    );

  cmpf0 : entity work.handshake_cmpf_0(arch)
    port map(
      lhs => addf0_result,
      lhs_valid => addf0_result_valid,
      lhs_ready => addf0_result_ready,
      rhs => constant7_outs,
      rhs_valid => constant7_outs_valid,
      rhs_ready => constant7_outs_ready,
      clk => clk,
      rst => rst,
      result => cmpf0_result,
      result_valid => cmpf0_result_valid,
      result_ready => cmpf0_result_ready
    );

  fork37 : entity work.handshake_fork_5(arch)
    port map(
      ins => cmpf0_result,
      ins_valid => cmpf0_result_valid,
      ins_ready => cmpf0_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork37_outs_0,
      outs(1) => fork37_outs_1,
      outs(2) => fork37_outs_2,
      outs(3) => fork37_outs_3,
      outs(4) => fork37_outs_4,
      outs(5) => fork37_outs_5,
      outs(6) => fork37_outs_6,
      outs(7) => fork37_outs_7,
      outs(8) => fork37_outs_8,
      outs(9) => fork37_outs_9,
      outs_valid(0) => fork37_outs_0_valid,
      outs_valid(1) => fork37_outs_1_valid,
      outs_valid(2) => fork37_outs_2_valid,
      outs_valid(3) => fork37_outs_3_valid,
      outs_valid(4) => fork37_outs_4_valid,
      outs_valid(5) => fork37_outs_5_valid,
      outs_valid(6) => fork37_outs_6_valid,
      outs_valid(7) => fork37_outs_7_valid,
      outs_valid(8) => fork37_outs_8_valid,
      outs_valid(9) => fork37_outs_9_valid,
      outs_ready(0) => fork37_outs_0_ready,
      outs_ready(1) => fork37_outs_1_ready,
      outs_ready(2) => fork37_outs_2_ready,
      outs_ready(3) => fork37_outs_3_ready,
      outs_ready(4) => fork37_outs_4_ready,
      outs_ready(5) => fork37_outs_5_ready,
      outs_ready(6) => fork37_outs_6_ready,
      outs_ready(7) => fork37_outs_7_ready,
      outs_ready(8) => fork37_outs_8_ready,
      outs_ready(9) => fork37_outs_9_ready
    );

  buffer8 : entity work.handshake_buffer_5(arch)
    port map(
      ins => fork37_outs_9,
      ins_valid => fork37_outs_9_valid,
      ins_ready => fork37_outs_9_ready,
      clk => clk,
      rst => rst,
      outs => buffer8_outs,
      outs_valid => buffer8_outs_valid,
      outs_ready => buffer8_outs_ready
    );

  not0 : entity work.handshake_not_0(arch)
    port map(
      ins => buffer8_outs,
      ins_valid => buffer8_outs_valid,
      ins_ready => buffer8_outs_ready,
      clk => clk,
      rst => rst,
      outs => not0_outs,
      outs_valid => not0_outs_valid,
      outs_ready => not0_outs_ready
    );

  fork38 : entity work.handshake_fork_6(arch)
    port map(
      ins => not0_outs,
      ins_valid => not0_outs_valid,
      ins_ready => not0_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork38_outs_0,
      outs(1) => fork38_outs_1,
      outs(2) => fork38_outs_2,
      outs_valid(0) => fork38_outs_0_valid,
      outs_valid(1) => fork38_outs_1_valid,
      outs_valid(2) => fork38_outs_2_valid,
      outs_ready(0) => fork38_outs_0_ready,
      outs_ready(1) => fork38_outs_1_ready,
      outs_ready(2) => fork38_outs_2_ready
    );

  buffer2 : entity work.handshake_buffer_0(arch)
    port map(
      ins => fork14_outs_1,
      ins_valid => fork14_outs_1_valid,
      ins_ready => fork14_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => buffer2_outs,
      outs_valid => buffer2_outs_valid,
      outs_ready => buffer2_outs_ready
    );

  buffer3 : entity work.handshake_buffer_6(arch)
    port map(
      ins => buffer2_outs,
      ins_valid => buffer2_outs_valid,
      ins_ready => buffer2_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer3_outs,
      outs_valid => buffer3_outs_valid,
      outs_ready => buffer3_outs_ready
    );

  source2 : entity work.handshake_source_0(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source2_outs_valid,
      outs_ready => source2_outs_ready
    );

  constant8 : entity work.handshake_constant_3(arch)
    port map(
      ctrl_valid => source2_outs_valid,
      ctrl_ready => source2_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant8_outs,
      outs_valid => constant8_outs_valid,
      outs_ready => constant8_outs_ready
    );

  mulf2 : entity work.handshake_mulf_2(arch)
    port map(
      lhs => fork15_outs_2,
      lhs_valid => fork15_outs_2_valid,
      lhs_ready => fork15_outs_2_ready,
      rhs => constant8_outs,
      rhs_valid => constant8_outs_valid,
      rhs_ready => constant8_outs_ready,
      clk => clk,
      rst => rst,
      result => mulf2_result,
      result_valid => mulf2_result_valid,
      result_ready => mulf2_result_ready
    );

  passer6 : entity work.handshake_passer_0(arch)
    port map(
      data => buffer9_outs,
      data_valid => buffer9_outs_valid,
      data_ready => buffer9_outs_ready,
      ctrl => fork38_outs_1,
      ctrl_valid => fork38_outs_1_valid,
      ctrl_ready => fork38_outs_1_ready,
      clk => clk,
      rst => rst,
      result => passer6_result,
      result_valid => passer6_result_valid,
      result_ready => passer6_result_ready
    );

  buffer9 : entity work.handshake_buffer_7(arch)
    port map(
      ins => mulf2_result,
      ins_valid => mulf2_result_valid,
      ins_ready => mulf2_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer9_outs,
      outs_valid => buffer9_outs_valid,
      outs_ready => buffer9_outs_ready
    );

  passer7 : entity work.handshake_passer_1(arch)
    port map(
      data => buffer3_outs,
      data_valid => buffer3_outs_valid,
      data_ready => buffer3_outs_ready,
      ctrl => fork38_outs_2,
      ctrl_valid => fork38_outs_2_valid,
      ctrl_ready => fork38_outs_2_ready,
      clk => clk,
      rst => rst,
      result => passer7_result,
      result_valid => passer7_result_valid,
      result_ready => passer7_result_ready
    );

  buffer10 : entity work.handshake_buffer_8(arch)
    port map(
      ins_valid => fork16_outs_0_valid,
      ins_ready => fork16_outs_0_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer10_outs_valid,
      outs_ready => buffer10_outs_ready
    );

  passer8 : entity work.handshake_passer_2(arch)
    port map(
      data_valid => buffer10_outs_valid,
      data_ready => buffer10_outs_ready,
      ctrl => fork38_outs_0,
      ctrl_valid => fork38_outs_0_valid,
      ctrl_ready => fork38_outs_0_ready,
      clk => clk,
      rst => rst,
      result_valid => passer8_result_valid,
      result_ready => passer8_result_ready
    );

  buffer11 : entity work.handshake_buffer_6(arch)
    port map(
      ins => fork14_outs_2,
      ins_valid => fork14_outs_2_valid,
      ins_ready => fork14_outs_2_ready,
      clk => clk,
      rst => rst,
      outs => buffer11_outs,
      outs_valid => buffer11_outs_valid,
      outs_ready => buffer11_outs_ready
    );

  fork34 : entity work.handshake_fork_7(arch)
    port map(
      ins => buffer11_outs,
      ins_valid => buffer11_outs_valid,
      ins_ready => buffer11_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork34_outs_0,
      outs(1) => fork34_outs_1,
      outs_valid(0) => fork34_outs_0_valid,
      outs_valid(1) => fork34_outs_1_valid,
      outs_ready(0) => fork34_outs_0_ready,
      outs_ready(1) => fork34_outs_1_ready
    );

  passer9 : entity work.handshake_passer_3(arch)
    port map(
      data => trunci1_outs,
      data_valid => trunci1_outs_valid,
      data_ready => trunci1_outs_ready,
      ctrl => fork37_outs_8,
      ctrl_valid => fork37_outs_8_valid,
      ctrl_ready => fork37_outs_8_ready,
      clk => clk,
      rst => rst,
      result => passer9_result,
      result_valid => passer9_result_valid,
      result_ready => passer9_result_ready
    );

  trunci1 : entity work.handshake_trunci_0(arch)
    port map(
      ins => fork34_outs_0,
      ins_valid => fork34_outs_0_valid,
      ins_ready => fork34_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => trunci1_outs,
      outs_valid => trunci1_outs_valid,
      outs_ready => trunci1_outs_ready
    );

  buffer12 : entity work.handshake_buffer_9(arch)
    port map(
      ins => fork35_outs_0,
      ins_valid => fork35_outs_0_valid,
      ins_ready => fork35_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => buffer12_outs,
      outs_valid => buffer12_outs_valid,
      outs_ready => buffer12_outs_ready
    );

  passer10 : entity work.handshake_passer_4(arch)
    port map(
      data => buffer12_outs,
      data_valid => buffer12_outs_valid,
      data_ready => buffer12_outs_ready,
      ctrl => fork37_outs_6,
      ctrl_valid => fork37_outs_6_valid,
      ctrl_ready => fork37_outs_6_ready,
      clk => clk,
      rst => rst,
      result => passer10_result,
      result_valid => passer10_result_valid,
      result_ready => passer10_result_ready
    );

  fork35 : entity work.handshake_fork_8(arch)
    port map(
      ins => fork15_outs_3,
      ins_valid => fork15_outs_3_valid,
      ins_ready => fork15_outs_3_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork35_outs_0,
      outs(1) => fork35_outs_1,
      outs_valid(0) => fork35_outs_0_valid,
      outs_valid(1) => fork35_outs_1_valid,
      outs_ready(0) => fork35_outs_0_ready,
      outs_ready(1) => fork35_outs_1_ready
    );

  fork36 : entity work.handshake_fork_3(arch)
    port map(
      ins_valid => fork16_outs_1_valid,
      ins_ready => fork16_outs_1_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork36_outs_0_valid,
      outs_valid(1) => fork36_outs_1_valid,
      outs_ready(0) => fork36_outs_0_ready,
      outs_ready(1) => fork36_outs_1_ready
    );

  constant3 : entity work.handshake_constant_4(arch)
    port map(
      ctrl_valid => fork36_outs_0_valid,
      ctrl_ready => fork36_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => constant3_outs,
      outs_valid => constant3_outs_valid,
      outs_ready => constant3_outs_ready
    );

  passer11 : entity work.handshake_passer_0(arch)
    port map(
      data => extsi1_outs,
      data_valid => extsi1_outs_valid,
      data_ready => extsi1_outs_ready,
      ctrl => fork37_outs_4,
      ctrl_valid => fork37_outs_4_valid,
      ctrl_ready => fork37_outs_4_ready,
      clk => clk,
      rst => rst,
      result => passer11_result,
      result_valid => passer11_result_valid,
      result_ready => passer11_result_ready
    );

  extsi1 : entity work.handshake_extsi_1(arch)
    port map(
      ins => constant3_outs,
      ins_valid => constant3_outs_valid,
      ins_ready => constant3_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi1_outs,
      outs_valid => extsi1_outs_valid,
      outs_ready => extsi1_outs_ready
    );

  source3 : entity work.handshake_source_0(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source3_outs_valid,
      outs_ready => source3_outs_ready
    );

  constant9 : entity work.handshake_constant_5(arch)
    port map(
      ctrl_valid => source3_outs_valid,
      ctrl_ready => source3_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant9_outs,
      outs_valid => constant9_outs_valid,
      outs_ready => constant9_outs_ready
    );

  store0 : entity work.handshake_store_0(arch)
    port map(
      addrIn => passer9_result,
      addrIn_valid => passer9_result_valid,
      addrIn_ready => passer9_result_ready,
      dataIn => passer10_result,
      dataIn_valid => passer10_result_valid,
      dataIn_ready => passer10_result_ready,
      clk => clk,
      rst => rst,
      addrOut => store0_addrOut,
      addrOut_valid => store0_addrOut_valid,
      addrOut_ready => store0_addrOut_ready,
      dataToMem => store0_dataToMem,
      dataToMem_valid => store0_dataToMem_valid,
      dataToMem_ready => store0_dataToMem_ready
    );

  divf0 : entity work.handshake_divf_0(arch)
    port map(
      lhs => fork35_outs_1,
      lhs_valid => fork35_outs_1_valid,
      lhs_ready => fork35_outs_1_ready,
      rhs => constant9_outs,
      rhs_valid => constant9_outs_valid,
      rhs_ready => constant9_outs_ready,
      clk => clk,
      rst => rst,
      result => divf0_result,
      result_valid => divf0_result_valid,
      result_ready => divf0_result_ready
    );

  buffer7 : entity work.handshake_buffer_2(arch)
    port map(
      ins => fork37_outs_5,
      ins_valid => fork37_outs_5_valid,
      ins_ready => fork37_outs_5_ready,
      clk => clk,
      rst => rst,
      outs => buffer7_outs,
      outs_valid => buffer7_outs_valid,
      outs_ready => buffer7_outs_ready
    );

  passer12 : entity work.handshake_passer_5(arch)
    port map(
      data => divf0_result,
      data_valid => divf0_result_valid,
      data_ready => divf0_result_ready,
      ctrl => buffer7_outs,
      ctrl_valid => buffer7_outs_valid,
      ctrl_ready => buffer7_outs_ready,
      clk => clk,
      rst => rst,
      result => passer12_result,
      result_valid => passer12_result_valid,
      result_ready => passer12_result_ready
    );

  passer13 : entity work.handshake_passer_1(arch)
    port map(
      data => fork34_outs_1,
      data_valid => fork34_outs_1_valid,
      data_ready => fork34_outs_1_ready,
      ctrl => fork37_outs_7,
      ctrl_valid => fork37_outs_7_valid,
      ctrl_ready => fork37_outs_7_ready,
      clk => clk,
      rst => rst,
      result => passer13_result,
      result_valid => passer13_result_valid,
      result_ready => passer13_result_ready
    );

  passer14 : entity work.handshake_passer_2(arch)
    port map(
      data_valid => fork36_outs_1_valid,
      data_ready => fork36_outs_1_ready,
      ctrl => fork37_outs_3,
      ctrl_valid => fork37_outs_3_valid,
      ctrl_ready => fork37_outs_3_ready,
      clk => clk,
      rst => rst,
      result_valid => passer14_result_valid,
      result_ready => passer14_result_ready
    );

  buffer13 : entity work.handshake_buffer_10(arch)
    port map(
      ins => passer12_result,
      ins_valid => passer12_result_valid,
      ins_ready => passer12_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer13_outs,
      outs_valid => buffer13_outs_valid,
      outs_ready => buffer13_outs_ready
    );

  mux2 : entity work.handshake_mux_2(arch)
    port map(
      index => fork37_outs_1,
      index_valid => fork37_outs_1_valid,
      index_ready => fork37_outs_1_ready,
      ins(0) => passer6_result,
      ins(1) => buffer13_outs,
      ins_valid(0) => passer6_result_valid,
      ins_valid(1) => buffer13_outs_valid,
      ins_ready(0) => passer6_result_ready,
      ins_ready(1) => buffer13_outs_ready,
      clk => clk,
      rst => rst,
      outs => mux2_outs,
      outs_valid => mux2_outs_valid,
      outs_ready => mux2_outs_ready
    );

  buffer16 : entity work.handshake_buffer_11(arch)
    port map(
      ins => mux2_outs,
      ins_valid => mux2_outs_valid,
      ins_ready => mux2_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer16_outs,
      outs_valid => buffer16_outs_valid,
      outs_ready => buffer16_outs_ready
    );

  fork8 : entity work.handshake_fork_9(arch)
    port map(
      ins => buffer16_outs,
      ins_valid => buffer16_outs_valid,
      ins_ready => buffer16_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork8_outs_0,
      outs(1) => fork8_outs_1,
      outs_valid(0) => fork8_outs_0_valid,
      outs_valid(1) => fork8_outs_1_valid,
      outs_ready(0) => fork8_outs_0_ready,
      outs_ready(1) => fork8_outs_1_ready
    );

  buffer14 : entity work.handshake_buffer_0(arch)
    port map(
      ins => passer13_result,
      ins_valid => passer13_result_valid,
      ins_ready => passer13_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer14_outs,
      outs_valid => buffer14_outs_valid,
      outs_ready => buffer14_outs_ready
    );

  mux3 : entity work.handshake_mux_0(arch)
    port map(
      index => fork37_outs_0,
      index_valid => fork37_outs_0_valid,
      index_ready => fork37_outs_0_ready,
      ins(0) => passer7_result,
      ins(1) => buffer14_outs,
      ins_valid(0) => passer7_result_valid,
      ins_valid(1) => buffer14_outs_valid,
      ins_ready(0) => passer7_result_ready,
      ins_ready(1) => buffer14_outs_ready,
      clk => clk,
      rst => rst,
      outs => mux3_outs,
      outs_valid => mux3_outs_valid,
      outs_ready => mux3_outs_ready
    );

  buffer17 : entity work.handshake_buffer_1(arch)
    port map(
      ins => mux3_outs,
      ins_valid => mux3_outs_valid,
      ins_ready => mux3_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer17_outs,
      outs_valid => buffer17_outs_valid,
      outs_ready => buffer17_outs_ready
    );

  extsi5 : entity work.handshake_extsi_2(arch)
    port map(
      ins => buffer17_outs,
      ins_valid => buffer17_outs_valid,
      ins_ready => buffer17_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi5_outs,
      outs_valid => extsi5_outs_valid,
      outs_ready => extsi5_outs_ready
    );

  buffer15 : entity work.handshake_buffer_8(arch)
    port map(
      ins_valid => passer14_result_valid,
      ins_ready => passer14_result_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer15_outs_valid,
      outs_ready => buffer15_outs_ready
    );

  mux4 : entity work.handshake_mux_3(arch)
    port map(
      index => fork37_outs_2,
      index_valid => fork37_outs_2_valid,
      index_ready => fork37_outs_2_ready,
      ins_valid(0) => passer8_result_valid,
      ins_valid(1) => buffer15_outs_valid,
      ins_ready(0) => passer8_result_ready,
      ins_ready(1) => buffer15_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux4_outs_valid,
      outs_ready => mux4_outs_ready
    );

  source4 : entity work.handshake_source_0(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source4_outs_valid,
      outs_ready => source4_outs_ready
    );

  constant4 : entity work.handshake_constant_4(arch)
    port map(
      ctrl_valid => source4_outs_valid,
      ctrl_ready => source4_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant4_outs,
      outs_valid => constant4_outs_valid,
      outs_ready => constant4_outs_ready
    );

  extsi6 : entity work.handshake_extsi_3(arch)
    port map(
      ins => constant4_outs,
      ins_valid => constant4_outs_valid,
      ins_ready => constant4_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi6_outs,
      outs_valid => extsi6_outs_valid,
      outs_ready => extsi6_outs_ready
    );

  source5 : entity work.handshake_source_0(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source5_outs_valid,
      outs_ready => source5_outs_ready
    );

  constant12 : entity work.handshake_constant_6(arch)
    port map(
      ctrl_valid => source5_outs_valid,
      ctrl_ready => source5_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant12_outs,
      outs_valid => constant12_outs_valid,
      outs_ready => constant12_outs_ready
    );

  extsi7 : entity work.handshake_extsi_2(arch)
    port map(
      ins => constant12_outs,
      ins_valid => constant12_outs_valid,
      ins_ready => constant12_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi7_outs,
      outs_valid => extsi7_outs_valid,
      outs_ready => extsi7_outs_ready
    );

  addf1 : entity work.handshake_addf_1(arch)
    port map(
      lhs => fork8_outs_0,
      lhs_valid => fork8_outs_0_valid,
      lhs_ready => fork8_outs_0_ready,
      rhs => fork8_outs_1,
      rhs_valid => fork8_outs_1_valid,
      rhs_ready => fork8_outs_1_ready,
      clk => clk,
      rst => rst,
      result => addf1_result,
      result_valid => addf1_result_valid,
      result_ready => addf1_result_ready
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

  fork10 : entity work.handshake_fork_10(arch)
    port map(
      ins => addi0_result,
      ins_valid => addi0_result_valid,
      ins_ready => addi0_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork10_outs_0,
      outs(1) => fork10_outs_1,
      outs_valid(0) => fork10_outs_0_valid,
      outs_valid(1) => fork10_outs_1_valid,
      outs_ready(0) => fork10_outs_0_ready,
      outs_ready(1) => fork10_outs_1_ready
    );

  trunci2 : entity work.handshake_trunci_1(arch)
    port map(
      ins => fork10_outs_0,
      ins_valid => fork10_outs_0_valid,
      ins_ready => fork10_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => trunci2_outs,
      outs_valid => trunci2_outs_valid,
      outs_ready => trunci2_outs_ready
    );

  cmpi0 : entity work.handshake_cmpi_0(arch)
    port map(
      lhs => fork10_outs_1,
      lhs_valid => fork10_outs_1_valid,
      lhs_ready => fork10_outs_1_ready,
      rhs => extsi7_outs,
      rhs_valid => extsi7_outs_valid,
      rhs_ready => extsi7_outs_ready,
      clk => clk,
      rst => rst,
      result => cmpi0_result,
      result_valid => cmpi0_result_valid,
      result_ready => cmpi0_result_ready
    );

  fork11 : entity work.handshake_fork_6(arch)
    port map(
      ins => cmpi0_result,
      ins_valid => cmpi0_result_valid,
      ins_ready => cmpi0_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork11_outs_0,
      outs(1) => fork11_outs_1,
      outs(2) => fork11_outs_2,
      outs_valid(0) => fork11_outs_0_valid,
      outs_valid(1) => fork11_outs_1_valid,
      outs_valid(2) => fork11_outs_2_valid,
      outs_ready(0) => fork11_outs_0_ready,
      outs_ready(1) => fork11_outs_1_ready,
      outs_ready(2) => fork11_outs_2_ready
    );

  cond_br5 : entity work.handshake_cond_br_0(arch)
    port map(
      condition => fork11_outs_0,
      condition_valid => fork11_outs_0_valid,
      condition_ready => fork11_outs_0_ready,
      data => trunci2_outs,
      data_valid => trunci2_outs_valid,
      data_ready => trunci2_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br5_trueOut,
      trueOut_valid => cond_br5_trueOut_valid,
      trueOut_ready => cond_br5_trueOut_ready,
      falseOut => cond_br5_falseOut,
      falseOut_valid => cond_br5_falseOut_valid,
      falseOut_ready => cond_br5_falseOut_ready
    );

  sink2 : entity work.handshake_sink_0(arch)
    port map(
      ins => cond_br5_falseOut,
      ins_valid => cond_br5_falseOut_valid,
      ins_ready => cond_br5_falseOut_ready,
      clk => clk,
      rst => rst
    );

  buffer19 : entity work.handshake_buffer_2(arch)
    port map(
      ins => fork11_outs_1,
      ins_valid => fork11_outs_1_valid,
      ins_ready => fork11_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => buffer19_outs,
      outs_valid => buffer19_outs_valid,
      outs_ready => buffer19_outs_ready
    );

  cond_br6 : entity work.handshake_cond_br_1(arch)
    port map(
      condition => buffer19_outs,
      condition_valid => buffer19_outs_valid,
      condition_ready => buffer19_outs_ready,
      data => addf1_result,
      data_valid => addf1_result_valid,
      data_ready => addf1_result_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br6_trueOut,
      trueOut_valid => cond_br6_trueOut_valid,
      trueOut_ready => cond_br6_trueOut_ready,
      falseOut => cond_br6_falseOut,
      falseOut_valid => cond_br6_falseOut_valid,
      falseOut_ready => cond_br6_falseOut_ready
    );

  buffer18 : entity work.handshake_buffer_12(arch)
    port map(
      ins_valid => mux4_outs_valid,
      ins_ready => mux4_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer18_outs_valid,
      outs_ready => buffer18_outs_ready
    );

  cond_br7 : entity work.handshake_cond_br_2(arch)
    port map(
      condition => fork11_outs_2,
      condition_valid => fork11_outs_2_valid,
      condition_ready => fork11_outs_2_ready,
      data_valid => buffer18_outs_valid,
      data_ready => buffer18_outs_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br7_trueOut_valid,
      trueOut_ready => cond_br7_trueOut_ready,
      falseOut_valid => cond_br7_falseOut_valid,
      falseOut_ready => cond_br7_falseOut_ready
    );

  fork12 : entity work.handshake_fork_3(arch)
    port map(
      ins_valid => cond_br7_falseOut_valid,
      ins_ready => cond_br7_falseOut_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork12_outs_0_valid,
      outs_valid(1) => fork12_outs_1_valid,
      outs_ready(0) => fork12_outs_0_ready,
      outs_ready(1) => fork12_outs_1_ready
    );

end architecture;
