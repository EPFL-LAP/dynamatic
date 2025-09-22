library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity collision_donut is
  port (
    x_loadData : in std_logic_vector(31 downto 0);
    y_loadData : in std_logic_vector(31 downto 0);
    x_start_valid : in std_logic;
    y_start_valid : in std_logic;
    start_valid : in std_logic;
    clk : in std_logic;
    rst : in std_logic;
    out0_ready : in std_logic;
    x_end_ready : in std_logic;
    y_end_ready : in std_logic;
    end_ready : in std_logic;
    x_start_ready : out std_logic;
    y_start_ready : out std_logic;
    start_ready : out std_logic;
    out0 : out std_logic_vector(31 downto 0);
    out0_valid : out std_logic;
    x_end_valid : out std_logic;
    y_end_valid : out std_logic;
    end_valid : out std_logic;
    x_loadEn : out std_logic;
    x_loadAddr : out std_logic_vector(9 downto 0);
    x_storeEn : out std_logic;
    x_storeAddr : out std_logic_vector(9 downto 0);
    x_storeData : out std_logic_vector(31 downto 0);
    y_loadEn : out std_logic;
    y_loadAddr : out std_logic_vector(9 downto 0);
    y_storeEn : out std_logic;
    y_storeAddr : out std_logic_vector(9 downto 0);
    y_storeData : out std_logic_vector(31 downto 0)
  );
end entity;

architecture behavioral of collision_donut is

  signal fork0_outs_0_valid : std_logic;
  signal fork0_outs_0_ready : std_logic;
  signal fork0_outs_1_valid : std_logic;
  signal fork0_outs_1_ready : std_logic;
  signal fork0_outs_2_valid : std_logic;
  signal fork0_outs_2_ready : std_logic;
  signal mem_controller2_ldData_0 : std_logic_vector(31 downto 0);
  signal mem_controller2_ldData_0_valid : std_logic;
  signal mem_controller2_ldData_0_ready : std_logic;
  signal mem_controller2_memEnd_valid : std_logic;
  signal mem_controller2_memEnd_ready : std_logic;
  signal mem_controller2_loadEn : std_logic;
  signal mem_controller2_loadAddr : std_logic_vector(9 downto 0);
  signal mem_controller2_storeEn : std_logic;
  signal mem_controller2_storeAddr : std_logic_vector(9 downto 0);
  signal mem_controller2_storeData : std_logic_vector(31 downto 0);
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
  signal constant0_outs : std_logic_vector(0 downto 0);
  signal constant0_outs_valid : std_logic;
  signal constant0_outs_ready : std_logic;
  signal extsi7_outs : std_logic_vector(10 downto 0);
  signal extsi7_outs_valid : std_logic;
  signal extsi7_outs_ready : std_logic;
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
  signal trunci0_outs : std_logic_vector(9 downto 0);
  signal trunci0_outs_valid : std_logic;
  signal trunci0_outs_ready : std_logic;
  signal trunci1_outs : std_logic_vector(9 downto 0);
  signal trunci1_outs_valid : std_logic;
  signal trunci1_outs_ready : std_logic;
  signal control_merge0_outs_valid : std_logic;
  signal control_merge0_outs_ready : std_logic;
  signal control_merge0_index : std_logic_vector(0 downto 0);
  signal control_merge0_index_valid : std_logic;
  signal control_merge0_index_ready : std_logic;
  signal buffer2_outs_valid : std_logic;
  signal buffer2_outs_ready : std_logic;
  signal fork2_outs_0_valid : std_logic;
  signal fork2_outs_0_ready : std_logic;
  signal fork2_outs_1_valid : std_logic;
  signal fork2_outs_1_ready : std_logic;
  signal source0_outs_valid : std_logic;
  signal source0_outs_ready : std_logic;
  signal constant1_outs : std_logic_vector(3 downto 0);
  signal constant1_outs_valid : std_logic;
  signal constant1_outs_ready : std_logic;
  signal extsi1_outs : std_logic_vector(31 downto 0);
  signal extsi1_outs_valid : std_logic;
  signal extsi1_outs_ready : std_logic;
  signal constant8_outs : std_logic_vector(31 downto 0);
  signal constant8_outs_valid : std_logic;
  signal constant8_outs_ready : std_logic;
  signal load0_addrOut : std_logic_vector(9 downto 0);
  signal load0_addrOut_valid : std_logic;
  signal load0_addrOut_ready : std_logic;
  signal load0_dataOut : std_logic_vector(31 downto 0);
  signal load0_dataOut_valid : std_logic;
  signal load0_dataOut_ready : std_logic;
  signal fork3_outs_0 : std_logic_vector(31 downto 0);
  signal fork3_outs_0_valid : std_logic;
  signal fork3_outs_0_ready : std_logic;
  signal fork3_outs_1 : std_logic_vector(31 downto 0);
  signal fork3_outs_1_valid : std_logic;
  signal fork3_outs_1_ready : std_logic;
  signal load1_addrOut : std_logic_vector(9 downto 0);
  signal load1_addrOut_valid : std_logic;
  signal load1_addrOut_ready : std_logic;
  signal load1_dataOut : std_logic_vector(31 downto 0);
  signal load1_dataOut_valid : std_logic;
  signal load1_dataOut_ready : std_logic;
  signal fork4_outs_0 : std_logic_vector(31 downto 0);
  signal fork4_outs_0_valid : std_logic;
  signal fork4_outs_0_ready : std_logic;
  signal fork4_outs_1 : std_logic_vector(31 downto 0);
  signal fork4_outs_1_valid : std_logic;
  signal fork4_outs_1_ready : std_logic;
  signal muli0_result : std_logic_vector(31 downto 0);
  signal muli0_result_valid : std_logic;
  signal muli0_result_ready : std_logic;
  signal muli1_result : std_logic_vector(31 downto 0);
  signal muli1_result_valid : std_logic;
  signal muli1_result_ready : std_logic;
  signal addi0_result : std_logic_vector(31 downto 0);
  signal addi0_result_valid : std_logic;
  signal addi0_result_ready : std_logic;
  signal fork5_outs_0 : std_logic_vector(31 downto 0);
  signal fork5_outs_0_valid : std_logic;
  signal fork5_outs_0_ready : std_logic;
  signal fork5_outs_1 : std_logic_vector(31 downto 0);
  signal fork5_outs_1_valid : std_logic;
  signal fork5_outs_1_ready : std_logic;
  signal cmpi0_result : std_logic_vector(0 downto 0);
  signal cmpi0_result_valid : std_logic;
  signal cmpi0_result_ready : std_logic;
  signal fork6_outs_0 : std_logic_vector(0 downto 0);
  signal fork6_outs_0_valid : std_logic;
  signal fork6_outs_0_ready : std_logic;
  signal fork6_outs_1 : std_logic_vector(0 downto 0);
  signal fork6_outs_1_valid : std_logic;
  signal fork6_outs_1_ready : std_logic;
  signal fork6_outs_2 : std_logic_vector(0 downto 0);
  signal fork6_outs_2_valid : std_logic;
  signal fork6_outs_2_ready : std_logic;
  signal fork6_outs_3 : std_logic_vector(0 downto 0);
  signal fork6_outs_3_valid : std_logic;
  signal fork6_outs_3_ready : std_logic;
  signal buffer1_outs : std_logic_vector(10 downto 0);
  signal buffer1_outs_valid : std_logic;
  signal buffer1_outs_ready : std_logic;
  signal cond_br3_trueOut : std_logic_vector(10 downto 0);
  signal cond_br3_trueOut_valid : std_logic;
  signal cond_br3_trueOut_ready : std_logic;
  signal cond_br3_falseOut : std_logic_vector(10 downto 0);
  signal cond_br3_falseOut_valid : std_logic;
  signal cond_br3_falseOut_ready : std_logic;
  signal extsi8_outs : std_logic_vector(11 downto 0);
  signal extsi8_outs_valid : std_logic;
  signal extsi8_outs_ready : std_logic;
  signal cond_br4_trueOut : std_logic_vector(31 downto 0);
  signal cond_br4_trueOut_valid : std_logic;
  signal cond_br4_trueOut_ready : std_logic;
  signal cond_br4_falseOut : std_logic_vector(31 downto 0);
  signal cond_br4_falseOut_valid : std_logic;
  signal cond_br4_falseOut_ready : std_logic;
  signal cond_br5_trueOut_valid : std_logic;
  signal cond_br5_trueOut_ready : std_logic;
  signal cond_br5_falseOut_valid : std_logic;
  signal cond_br5_falseOut_ready : std_logic;
  signal cond_br6_trueOut : std_logic_vector(31 downto 0);
  signal cond_br6_trueOut_valid : std_logic;
  signal cond_br6_trueOut_ready : std_logic;
  signal cond_br6_falseOut : std_logic_vector(31 downto 0);
  signal cond_br6_falseOut_valid : std_logic;
  signal cond_br6_falseOut_ready : std_logic;
  signal buffer4_outs_valid : std_logic;
  signal buffer4_outs_ready : std_logic;
  signal fork7_outs_0_valid : std_logic;
  signal fork7_outs_0_ready : std_logic;
  signal fork7_outs_1_valid : std_logic;
  signal fork7_outs_1_ready : std_logic;
  signal source1_outs_valid : std_logic;
  signal source1_outs_ready : std_logic;
  signal constant2_outs : std_logic_vector(15 downto 0);
  signal constant2_outs_valid : std_logic;
  signal constant2_outs_ready : std_logic;
  signal extsi2_outs : std_logic_vector(31 downto 0);
  signal extsi2_outs_valid : std_logic;
  signal extsi2_outs_ready : std_logic;
  signal constant10_outs : std_logic_vector(31 downto 0);
  signal constant10_outs_valid : std_logic;
  signal constant10_outs_ready : std_logic;
  signal cmpi1_result : std_logic_vector(0 downto 0);
  signal cmpi1_result_valid : std_logic;
  signal cmpi1_result_ready : std_logic;
  signal buffer5_outs : std_logic_vector(0 downto 0);
  signal buffer5_outs_valid : std_logic;
  signal buffer5_outs_ready : std_logic;
  signal fork8_outs_0 : std_logic_vector(0 downto 0);
  signal fork8_outs_0_valid : std_logic;
  signal fork8_outs_0_ready : std_logic;
  signal fork8_outs_1 : std_logic_vector(0 downto 0);
  signal fork8_outs_1_valid : std_logic;
  signal fork8_outs_1_ready : std_logic;
  signal fork8_outs_2 : std_logic_vector(0 downto 0);
  signal fork8_outs_2_valid : std_logic;
  signal fork8_outs_2_ready : std_logic;
  signal buffer3_outs : std_logic_vector(10 downto 0);
  signal buffer3_outs_valid : std_logic;
  signal buffer3_outs_ready : std_logic;
  signal cond_br7_trueOut : std_logic_vector(10 downto 0);
  signal cond_br7_trueOut_valid : std_logic;
  signal cond_br7_trueOut_ready : std_logic;
  signal cond_br7_falseOut : std_logic_vector(10 downto 0);
  signal cond_br7_falseOut_valid : std_logic;
  signal cond_br7_falseOut_ready : std_logic;
  signal extsi9_outs : std_logic_vector(11 downto 0);
  signal extsi9_outs_valid : std_logic;
  signal extsi9_outs_ready : std_logic;
  signal cond_br8_trueOut : std_logic_vector(31 downto 0);
  signal cond_br8_trueOut_valid : std_logic;
  signal cond_br8_trueOut_ready : std_logic;
  signal cond_br8_falseOut : std_logic_vector(31 downto 0);
  signal cond_br8_falseOut_valid : std_logic;
  signal cond_br8_falseOut_ready : std_logic;
  signal cond_br9_trueOut_valid : std_logic;
  signal cond_br9_trueOut_ready : std_logic;
  signal cond_br9_falseOut_valid : std_logic;
  signal cond_br9_falseOut_ready : std_logic;
  signal extsi10_outs : std_logic_vector(11 downto 0);
  signal extsi10_outs_valid : std_logic;
  signal extsi10_outs_ready : std_logic;
  signal fork9_outs_0_valid : std_logic;
  signal fork9_outs_0_ready : std_logic;
  signal fork9_outs_1_valid : std_logic;
  signal fork9_outs_1_ready : std_logic;
  signal constant3_outs : std_logic_vector(0 downto 0);
  signal constant3_outs_valid : std_logic;
  signal constant3_outs_ready : std_logic;
  signal source2_outs_valid : std_logic;
  signal source2_outs_ready : std_logic;
  signal constant4_outs : std_logic_vector(1 downto 0);
  signal constant4_outs_valid : std_logic;
  signal constant4_outs_ready : std_logic;
  signal extsi11_outs : std_logic_vector(11 downto 0);
  signal extsi11_outs_valid : std_logic;
  signal extsi11_outs_ready : std_logic;
  signal source3_outs_valid : std_logic;
  signal source3_outs_ready : std_logic;
  signal constant5_outs : std_logic_vector(10 downto 0);
  signal constant5_outs_valid : std_logic;
  signal constant5_outs_ready : std_logic;
  signal extsi12_outs : std_logic_vector(11 downto 0);
  signal extsi12_outs_valid : std_logic;
  signal extsi12_outs_ready : std_logic;
  signal addi1_result : std_logic_vector(11 downto 0);
  signal addi1_result_valid : std_logic;
  signal addi1_result_ready : std_logic;
  signal fork10_outs_0 : std_logic_vector(11 downto 0);
  signal fork10_outs_0_valid : std_logic;
  signal fork10_outs_0_ready : std_logic;
  signal fork10_outs_1 : std_logic_vector(11 downto 0);
  signal fork10_outs_1_valid : std_logic;
  signal fork10_outs_1_ready : std_logic;
  signal cmpi2_result : std_logic_vector(0 downto 0);
  signal cmpi2_result_valid : std_logic;
  signal cmpi2_result_ready : std_logic;
  signal fork11_outs_0 : std_logic_vector(0 downto 0);
  signal fork11_outs_0_valid : std_logic;
  signal fork11_outs_0_ready : std_logic;
  signal fork11_outs_1 : std_logic_vector(0 downto 0);
  signal fork11_outs_1_valid : std_logic;
  signal fork11_outs_1_ready : std_logic;
  signal fork11_outs_2 : std_logic_vector(0 downto 0);
  signal fork11_outs_2_valid : std_logic;
  signal fork11_outs_2_ready : std_logic;
  signal cond_br10_trueOut : std_logic_vector(11 downto 0);
  signal cond_br10_trueOut_valid : std_logic;
  signal cond_br10_trueOut_ready : std_logic;
  signal cond_br10_falseOut : std_logic_vector(11 downto 0);
  signal cond_br10_falseOut_valid : std_logic;
  signal cond_br10_falseOut_ready : std_logic;
  signal trunci2_outs : std_logic_vector(10 downto 0);
  signal trunci2_outs_valid : std_logic;
  signal trunci2_outs_ready : std_logic;
  signal cond_br11_trueOut_valid : std_logic;
  signal cond_br11_trueOut_ready : std_logic;
  signal cond_br11_falseOut_valid : std_logic;
  signal cond_br11_falseOut_ready : std_logic;
  signal cond_br12_trueOut : std_logic_vector(0 downto 0);
  signal cond_br12_trueOut_valid : std_logic;
  signal cond_br12_trueOut_ready : std_logic;
  signal cond_br12_falseOut : std_logic_vector(0 downto 0);
  signal cond_br12_falseOut_valid : std_logic;
  signal cond_br12_falseOut_ready : std_logic;
  signal extsi13_outs : std_logic_vector(31 downto 0);
  signal extsi13_outs_valid : std_logic;
  signal extsi13_outs_ready : std_logic;
  signal mux1_outs : std_logic_vector(11 downto 0);
  signal mux1_outs_valid : std_logic;
  signal mux1_outs_ready : std_logic;
  signal mux2_outs : std_logic_vector(31 downto 0);
  signal mux2_outs_valid : std_logic;
  signal mux2_outs_ready : std_logic;
  signal control_merge3_outs_valid : std_logic;
  signal control_merge3_outs_ready : std_logic;
  signal control_merge3_index : std_logic_vector(0 downto 0);
  signal control_merge3_index_valid : std_logic;
  signal control_merge3_index_ready : std_logic;
  signal fork12_outs_0 : std_logic_vector(0 downto 0);
  signal fork12_outs_0_valid : std_logic;
  signal fork12_outs_0_ready : std_logic;
  signal fork12_outs_1 : std_logic_vector(0 downto 0);
  signal fork12_outs_1_valid : std_logic;
  signal fork12_outs_1_ready : std_logic;
  signal buffer6_outs : std_logic_vector(11 downto 0);
  signal buffer6_outs_valid : std_logic;
  signal buffer6_outs_ready : std_logic;
  signal mux3_outs : std_logic_vector(11 downto 0);
  signal mux3_outs_valid : std_logic;
  signal mux3_outs_ready : std_logic;
  signal extsi14_outs : std_logic_vector(31 downto 0);
  signal extsi14_outs_valid : std_logic;
  signal extsi14_outs_ready : std_logic;
  signal buffer7_outs : std_logic_vector(31 downto 0);
  signal buffer7_outs_valid : std_logic;
  signal buffer7_outs_ready : std_logic;
  signal mux4_outs : std_logic_vector(31 downto 0);
  signal mux4_outs_valid : std_logic;
  signal mux4_outs_ready : std_logic;
  signal control_merge4_outs_valid : std_logic;
  signal control_merge4_outs_ready : std_logic;
  signal control_merge4_index : std_logic_vector(0 downto 0);
  signal control_merge4_index_valid : std_logic;
  signal control_merge4_index_ready : std_logic;
  signal buffer8_outs : std_logic_vector(0 downto 0);
  signal buffer8_outs_valid : std_logic;
  signal buffer8_outs_ready : std_logic;
  signal fork13_outs_0 : std_logic_vector(0 downto 0);
  signal fork13_outs_0_valid : std_logic;
  signal fork13_outs_0_ready : std_logic;
  signal fork13_outs_1 : std_logic_vector(0 downto 0);
  signal fork13_outs_1_valid : std_logic;
  signal fork13_outs_1_ready : std_logic;
  signal fork14_outs_0_valid : std_logic;
  signal fork14_outs_0_ready : std_logic;
  signal fork14_outs_1_valid : std_logic;
  signal fork14_outs_1_ready : std_logic;
  signal source4_outs_valid : std_logic;
  signal source4_outs_ready : std_logic;
  signal constant15_outs : std_logic_vector(1 downto 0);
  signal constant15_outs_valid : std_logic;
  signal constant15_outs_ready : std_logic;
  signal extsi6_outs : std_logic_vector(31 downto 0);
  signal extsi6_outs_valid : std_logic;
  signal extsi6_outs_ready : std_logic;
  signal shli0_result : std_logic_vector(31 downto 0);
  signal shli0_result_valid : std_logic;
  signal shli0_result_ready : std_logic;
  signal andi0_result : std_logic_vector(31 downto 0);
  signal andi0_result_valid : std_logic;
  signal andi0_result_ready : std_logic;

begin

  out0 <= andi0_result;
  out0_valid <= andi0_result_valid;
  andi0_result_ready <= out0_ready;
  x_end_valid <= mem_controller3_memEnd_valid;
  mem_controller3_memEnd_ready <= x_end_ready;
  y_end_valid <= mem_controller2_memEnd_valid;
  mem_controller2_memEnd_ready <= y_end_ready;
  end_valid <= fork0_outs_1_valid;
  fork0_outs_1_ready <= end_ready;
  x_loadEn <= mem_controller3_loadEn;
  x_loadAddr <= mem_controller3_loadAddr;
  x_storeEn <= mem_controller3_storeEn;
  x_storeAddr <= mem_controller3_storeAddr;
  x_storeData <= mem_controller3_storeData;
  y_loadEn <= mem_controller2_loadEn;
  y_loadAddr <= mem_controller2_loadAddr;
  y_storeEn <= mem_controller2_storeEn;
  y_storeAddr <= mem_controller2_storeAddr;
  y_storeData <= mem_controller2_storeData;

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
      loadData => y_loadData,
      memStart_valid => y_start_valid,
      memStart_ready => y_start_ready,
      ldAddr(0) => load1_addrOut,
      ldAddr_valid(0) => load1_addrOut_valid,
      ldAddr_ready(0) => load1_addrOut_ready,
      ctrlEnd_valid => fork14_outs_1_valid,
      ctrlEnd_ready => fork14_outs_1_ready,
      clk => clk,
      rst => rst,
      ldData(0) => mem_controller2_ldData_0,
      ldData_valid(0) => mem_controller2_ldData_0_valid,
      ldData_ready(0) => mem_controller2_ldData_0_ready,
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
      loadData => x_loadData,
      memStart_valid => x_start_valid,
      memStart_ready => x_start_ready,
      ldAddr(0) => load0_addrOut,
      ldAddr_valid(0) => load0_addrOut_valid,
      ldAddr_ready(0) => load0_addrOut_ready,
      ctrlEnd_valid => fork14_outs_0_valid,
      ctrlEnd_ready => fork14_outs_0_ready,
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

  extsi7 : entity work.handshake_extsi_0(arch)
    port map(
      ins => constant0_outs,
      ins_valid => constant0_outs_valid,
      ins_ready => constant0_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi7_outs,
      outs_valid => extsi7_outs_valid,
      outs_ready => extsi7_outs_ready
    );

  mux0 : entity work.handshake_mux_0(arch)
    port map(
      index => control_merge0_index,
      index_valid => control_merge0_index_valid,
      index_ready => control_merge0_index_ready,
      ins(0) => extsi7_outs,
      ins(1) => trunci2_outs,
      ins_valid(0) => extsi7_outs_valid,
      ins_valid(1) => trunci2_outs_valid,
      ins_ready(0) => extsi7_outs_ready,
      ins_ready(1) => trunci2_outs_ready,
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
      outs_valid(0) => fork1_outs_0_valid,
      outs_valid(1) => fork1_outs_1_valid,
      outs_valid(2) => fork1_outs_2_valid,
      outs_ready(0) => fork1_outs_0_ready,
      outs_ready(1) => fork1_outs_1_ready,
      outs_ready(2) => fork1_outs_2_ready
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

  control_merge0 : entity work.handshake_control_merge_0(arch)
    port map(
      ins_valid(0) => fork0_outs_2_valid,
      ins_valid(1) => cond_br11_trueOut_valid,
      ins_ready(0) => fork0_outs_2_ready,
      ins_ready(1) => cond_br11_trueOut_ready,
      clk => clk,
      rst => rst,
      outs_valid => control_merge0_outs_valid,
      outs_ready => control_merge0_outs_ready,
      index => control_merge0_index,
      index_valid => control_merge0_index_valid,
      index_ready => control_merge0_index_ready
    );

  buffer2 : entity work.handshake_buffer_1(arch)
    port map(
      ins_valid => control_merge0_outs_valid,
      ins_ready => control_merge0_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer2_outs_valid,
      outs_ready => buffer2_outs_ready
    );

  fork2 : entity work.handshake_fork_2(arch)
    port map(
      ins_valid => buffer2_outs_valid,
      ins_ready => buffer2_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork2_outs_0_valid,
      outs_valid(1) => fork2_outs_1_valid,
      outs_ready(0) => fork2_outs_0_ready,
      outs_ready(1) => fork2_outs_1_ready
    );

  source0 : entity work.handshake_source_0(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source0_outs_valid,
      outs_ready => source0_outs_ready
    );

  constant1 : entity work.handshake_constant_1(arch)
    port map(
      ctrl_valid => source0_outs_valid,
      ctrl_ready => source0_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant1_outs,
      outs_valid => constant1_outs_valid,
      outs_ready => constant1_outs_ready
    );

  extsi1 : entity work.handshake_extsi_1(arch)
    port map(
      ins => constant1_outs,
      ins_valid => constant1_outs_valid,
      ins_ready => constant1_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi1_outs,
      outs_valid => extsi1_outs_valid,
      outs_ready => extsi1_outs_ready
    );

  constant8 : entity work.handshake_constant_2(arch)
    port map(
      ctrl_valid => fork2_outs_1_valid,
      ctrl_ready => fork2_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => constant8_outs,
      outs_valid => constant8_outs_valid,
      outs_ready => constant8_outs_ready
    );

  load0 : entity work.handshake_load_0(arch)
    port map(
      addrIn => trunci1_outs,
      addrIn_valid => trunci1_outs_valid,
      addrIn_ready => trunci1_outs_ready,
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

  fork3 : entity work.handshake_fork_3(arch)
    port map(
      ins => load0_dataOut,
      ins_valid => load0_dataOut_valid,
      ins_ready => load0_dataOut_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork3_outs_0,
      outs(1) => fork3_outs_1,
      outs_valid(0) => fork3_outs_0_valid,
      outs_valid(1) => fork3_outs_1_valid,
      outs_ready(0) => fork3_outs_0_ready,
      outs_ready(1) => fork3_outs_1_ready
    );

  load1 : entity work.handshake_load_0(arch)
    port map(
      addrIn => trunci0_outs,
      addrIn_valid => trunci0_outs_valid,
      addrIn_ready => trunci0_outs_ready,
      dataFromMem => mem_controller2_ldData_0,
      dataFromMem_valid => mem_controller2_ldData_0_valid,
      dataFromMem_ready => mem_controller2_ldData_0_ready,
      clk => clk,
      rst => rst,
      addrOut => load1_addrOut,
      addrOut_valid => load1_addrOut_valid,
      addrOut_ready => load1_addrOut_ready,
      dataOut => load1_dataOut,
      dataOut_valid => load1_dataOut_valid,
      dataOut_ready => load1_dataOut_ready
    );

  fork4 : entity work.handshake_fork_3(arch)
    port map(
      ins => load1_dataOut,
      ins_valid => load1_dataOut_valid,
      ins_ready => load1_dataOut_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork4_outs_0,
      outs(1) => fork4_outs_1,
      outs_valid(0) => fork4_outs_0_valid,
      outs_valid(1) => fork4_outs_1_valid,
      outs_ready(0) => fork4_outs_0_ready,
      outs_ready(1) => fork4_outs_1_ready
    );

  muli0 : entity work.handshake_muli_0(arch)
    port map(
      lhs => fork3_outs_0,
      lhs_valid => fork3_outs_0_valid,
      lhs_ready => fork3_outs_0_ready,
      rhs => fork3_outs_1,
      rhs_valid => fork3_outs_1_valid,
      rhs_ready => fork3_outs_1_ready,
      clk => clk,
      rst => rst,
      result => muli0_result,
      result_valid => muli0_result_valid,
      result_ready => muli0_result_ready
    );

  muli1 : entity work.handshake_muli_0(arch)
    port map(
      lhs => fork4_outs_0,
      lhs_valid => fork4_outs_0_valid,
      lhs_ready => fork4_outs_0_ready,
      rhs => fork4_outs_1,
      rhs_valid => fork4_outs_1_valid,
      rhs_ready => fork4_outs_1_ready,
      clk => clk,
      rst => rst,
      result => muli1_result,
      result_valid => muli1_result_valid,
      result_ready => muli1_result_ready
    );

  addi0 : entity work.handshake_addi_0(arch)
    port map(
      lhs => muli0_result,
      lhs_valid => muli0_result_valid,
      lhs_ready => muli0_result_ready,
      rhs => muli1_result,
      rhs_valid => muli1_result_valid,
      rhs_ready => muli1_result_ready,
      clk => clk,
      rst => rst,
      result => addi0_result,
      result_valid => addi0_result_valid,
      result_ready => addi0_result_ready
    );

  fork5 : entity work.handshake_fork_3(arch)
    port map(
      ins => addi0_result,
      ins_valid => addi0_result_valid,
      ins_ready => addi0_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork5_outs_0,
      outs(1) => fork5_outs_1,
      outs_valid(0) => fork5_outs_0_valid,
      outs_valid(1) => fork5_outs_1_valid,
      outs_ready(0) => fork5_outs_0_ready,
      outs_ready(1) => fork5_outs_1_ready
    );

  cmpi0 : entity work.handshake_cmpi_0(arch)
    port map(
      lhs => fork5_outs_1,
      lhs_valid => fork5_outs_1_valid,
      lhs_ready => fork5_outs_1_ready,
      rhs => extsi1_outs,
      rhs_valid => extsi1_outs_valid,
      rhs_ready => extsi1_outs_ready,
      clk => clk,
      rst => rst,
      result => cmpi0_result,
      result_valid => cmpi0_result_valid,
      result_ready => cmpi0_result_ready
    );

  fork6 : entity work.handshake_fork_4(arch)
    port map(
      ins => cmpi0_result,
      ins_valid => cmpi0_result_valid,
      ins_ready => cmpi0_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork6_outs_0,
      outs(1) => fork6_outs_1,
      outs(2) => fork6_outs_2,
      outs(3) => fork6_outs_3,
      outs_valid(0) => fork6_outs_0_valid,
      outs_valid(1) => fork6_outs_1_valid,
      outs_valid(2) => fork6_outs_2_valid,
      outs_valid(3) => fork6_outs_3_valid,
      outs_ready(0) => fork6_outs_0_ready,
      outs_ready(1) => fork6_outs_1_ready,
      outs_ready(2) => fork6_outs_2_ready,
      outs_ready(3) => fork6_outs_3_ready
    );

  buffer1 : entity work.handshake_buffer_2(arch)
    port map(
      ins => fork1_outs_2,
      ins_valid => fork1_outs_2_valid,
      ins_ready => fork1_outs_2_ready,
      clk => clk,
      rst => rst,
      outs => buffer1_outs,
      outs_valid => buffer1_outs_valid,
      outs_ready => buffer1_outs_ready
    );

  cond_br3 : entity work.handshake_cond_br_0(arch)
    port map(
      condition => fork6_outs_0,
      condition_valid => fork6_outs_0_valid,
      condition_ready => fork6_outs_0_ready,
      data => buffer1_outs,
      data_valid => buffer1_outs_valid,
      data_ready => buffer1_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br3_trueOut,
      trueOut_valid => cond_br3_trueOut_valid,
      trueOut_ready => cond_br3_trueOut_ready,
      falseOut => cond_br3_falseOut,
      falseOut_valid => cond_br3_falseOut_valid,
      falseOut_ready => cond_br3_falseOut_ready
    );

  extsi8 : entity work.handshake_extsi_2(arch)
    port map(
      ins => cond_br3_trueOut,
      ins_valid => cond_br3_trueOut_valid,
      ins_ready => cond_br3_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => extsi8_outs,
      outs_valid => extsi8_outs_valid,
      outs_ready => extsi8_outs_ready
    );

  cond_br4 : entity work.handshake_cond_br_1(arch)
    port map(
      condition => fork6_outs_3,
      condition_valid => fork6_outs_3_valid,
      condition_ready => fork6_outs_3_ready,
      data => constant8_outs,
      data_valid => constant8_outs_valid,
      data_ready => constant8_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br4_trueOut,
      trueOut_valid => cond_br4_trueOut_valid,
      trueOut_ready => cond_br4_trueOut_ready,
      falseOut => cond_br4_falseOut,
      falseOut_valid => cond_br4_falseOut_valid,
      falseOut_ready => cond_br4_falseOut_ready
    );

  sink0 : entity work.handshake_sink_0(arch)
    port map(
      ins => cond_br4_falseOut,
      ins_valid => cond_br4_falseOut_valid,
      ins_ready => cond_br4_falseOut_ready,
      clk => clk,
      rst => rst
    );

  cond_br5 : entity work.handshake_cond_br_2(arch)
    port map(
      condition => fork6_outs_2,
      condition_valid => fork6_outs_2_valid,
      condition_ready => fork6_outs_2_ready,
      data_valid => fork2_outs_0_valid,
      data_ready => fork2_outs_0_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br5_trueOut_valid,
      trueOut_ready => cond_br5_trueOut_ready,
      falseOut_valid => cond_br5_falseOut_valid,
      falseOut_ready => cond_br5_falseOut_ready
    );

  cond_br6 : entity work.handshake_cond_br_1(arch)
    port map(
      condition => fork6_outs_1,
      condition_valid => fork6_outs_1_valid,
      condition_ready => fork6_outs_1_ready,
      data => fork5_outs_0,
      data_valid => fork5_outs_0_valid,
      data_ready => fork5_outs_0_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br6_trueOut,
      trueOut_valid => cond_br6_trueOut_valid,
      trueOut_ready => cond_br6_trueOut_ready,
      falseOut => cond_br6_falseOut,
      falseOut_valid => cond_br6_falseOut_valid,
      falseOut_ready => cond_br6_falseOut_ready
    );

  sink1 : entity work.handshake_sink_0(arch)
    port map(
      ins => cond_br6_trueOut,
      ins_valid => cond_br6_trueOut_valid,
      ins_ready => cond_br6_trueOut_ready,
      clk => clk,
      rst => rst
    );

  buffer4 : entity work.handshake_buffer_1(arch)
    port map(
      ins_valid => cond_br5_falseOut_valid,
      ins_ready => cond_br5_falseOut_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer4_outs_valid,
      outs_ready => buffer4_outs_ready
    );

  fork7 : entity work.handshake_fork_2(arch)
    port map(
      ins_valid => buffer4_outs_valid,
      ins_ready => buffer4_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork7_outs_0_valid,
      outs_valid(1) => fork7_outs_1_valid,
      outs_ready(0) => fork7_outs_0_ready,
      outs_ready(1) => fork7_outs_1_ready
    );

  source1 : entity work.handshake_source_0(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source1_outs_valid,
      outs_ready => source1_outs_ready
    );

  constant2 : entity work.handshake_constant_3(arch)
    port map(
      ctrl_valid => source1_outs_valid,
      ctrl_ready => source1_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant2_outs,
      outs_valid => constant2_outs_valid,
      outs_ready => constant2_outs_ready
    );

  extsi2 : entity work.handshake_extsi_3(arch)
    port map(
      ins => constant2_outs,
      ins_valid => constant2_outs_valid,
      ins_ready => constant2_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi2_outs,
      outs_valid => extsi2_outs_valid,
      outs_ready => extsi2_outs_ready
    );

  constant10 : entity work.handshake_constant_4(arch)
    port map(
      ctrl_valid => fork7_outs_1_valid,
      ctrl_ready => fork7_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => constant10_outs,
      outs_valid => constant10_outs_valid,
      outs_ready => constant10_outs_ready
    );

  cmpi1 : entity work.handshake_cmpi_1(arch)
    port map(
      lhs => cond_br6_falseOut,
      lhs_valid => cond_br6_falseOut_valid,
      lhs_ready => cond_br6_falseOut_ready,
      rhs => extsi2_outs,
      rhs_valid => extsi2_outs_valid,
      rhs_ready => extsi2_outs_ready,
      clk => clk,
      rst => rst,
      result => cmpi1_result,
      result_valid => cmpi1_result_valid,
      result_ready => cmpi1_result_ready
    );

  buffer5 : entity work.handshake_buffer_3(arch)
    port map(
      ins => cmpi1_result,
      ins_valid => cmpi1_result_valid,
      ins_ready => cmpi1_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer5_outs,
      outs_valid => buffer5_outs_valid,
      outs_ready => buffer5_outs_ready
    );

  fork8 : entity work.handshake_fork_5(arch)
    port map(
      ins => buffer5_outs,
      ins_valid => buffer5_outs_valid,
      ins_ready => buffer5_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork8_outs_0,
      outs(1) => fork8_outs_1,
      outs(2) => fork8_outs_2,
      outs_valid(0) => fork8_outs_0_valid,
      outs_valid(1) => fork8_outs_1_valid,
      outs_valid(2) => fork8_outs_2_valid,
      outs_ready(0) => fork8_outs_0_ready,
      outs_ready(1) => fork8_outs_1_ready,
      outs_ready(2) => fork8_outs_2_ready
    );

  buffer3 : entity work.handshake_buffer_2(arch)
    port map(
      ins => cond_br3_falseOut,
      ins_valid => cond_br3_falseOut_valid,
      ins_ready => cond_br3_falseOut_ready,
      clk => clk,
      rst => rst,
      outs => buffer3_outs,
      outs_valid => buffer3_outs_valid,
      outs_ready => buffer3_outs_ready
    );

  cond_br7 : entity work.handshake_cond_br_0(arch)
    port map(
      condition => fork8_outs_0,
      condition_valid => fork8_outs_0_valid,
      condition_ready => fork8_outs_0_ready,
      data => buffer3_outs,
      data_valid => buffer3_outs_valid,
      data_ready => buffer3_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br7_trueOut,
      trueOut_valid => cond_br7_trueOut_valid,
      trueOut_ready => cond_br7_trueOut_ready,
      falseOut => cond_br7_falseOut,
      falseOut_valid => cond_br7_falseOut_valid,
      falseOut_ready => cond_br7_falseOut_ready
    );

  extsi9 : entity work.handshake_extsi_2(arch)
    port map(
      ins => cond_br7_trueOut,
      ins_valid => cond_br7_trueOut_valid,
      ins_ready => cond_br7_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => extsi9_outs,
      outs_valid => extsi9_outs_valid,
      outs_ready => extsi9_outs_ready
    );

  cond_br8 : entity work.handshake_cond_br_1(arch)
    port map(
      condition => fork8_outs_2,
      condition_valid => fork8_outs_2_valid,
      condition_ready => fork8_outs_2_ready,
      data => constant10_outs,
      data_valid => constant10_outs_valid,
      data_ready => constant10_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br8_trueOut,
      trueOut_valid => cond_br8_trueOut_valid,
      trueOut_ready => cond_br8_trueOut_ready,
      falseOut => cond_br8_falseOut,
      falseOut_valid => cond_br8_falseOut_valid,
      falseOut_ready => cond_br8_falseOut_ready
    );

  sink3 : entity work.handshake_sink_0(arch)
    port map(
      ins => cond_br8_falseOut,
      ins_valid => cond_br8_falseOut_valid,
      ins_ready => cond_br8_falseOut_ready,
      clk => clk,
      rst => rst
    );

  cond_br9 : entity work.handshake_cond_br_2(arch)
    port map(
      condition => fork8_outs_1,
      condition_valid => fork8_outs_1_valid,
      condition_ready => fork8_outs_1_ready,
      data_valid => fork7_outs_0_valid,
      data_ready => fork7_outs_0_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br9_trueOut_valid,
      trueOut_ready => cond_br9_trueOut_ready,
      falseOut_valid => cond_br9_falseOut_valid,
      falseOut_ready => cond_br9_falseOut_ready
    );

  extsi10 : entity work.handshake_extsi_2(arch)
    port map(
      ins => cond_br7_falseOut,
      ins_valid => cond_br7_falseOut_valid,
      ins_ready => cond_br7_falseOut_ready,
      clk => clk,
      rst => rst,
      outs => extsi10_outs,
      outs_valid => extsi10_outs_valid,
      outs_ready => extsi10_outs_ready
    );

  fork9 : entity work.handshake_fork_2(arch)
    port map(
      ins_valid => cond_br9_falseOut_valid,
      ins_ready => cond_br9_falseOut_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork9_outs_0_valid,
      outs_valid(1) => fork9_outs_1_valid,
      outs_ready(0) => fork9_outs_0_ready,
      outs_ready(1) => fork9_outs_1_ready
    );

  constant3 : entity work.handshake_constant_0(arch)
    port map(
      ctrl_valid => fork9_outs_0_valid,
      ctrl_ready => fork9_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => constant3_outs,
      outs_valid => constant3_outs_valid,
      outs_ready => constant3_outs_ready
    );

  source2 : entity work.handshake_source_0(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source2_outs_valid,
      outs_ready => source2_outs_ready
    );

  constant4 : entity work.handshake_constant_5(arch)
    port map(
      ctrl_valid => source2_outs_valid,
      ctrl_ready => source2_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant4_outs,
      outs_valid => constant4_outs_valid,
      outs_ready => constant4_outs_ready
    );

  extsi11 : entity work.handshake_extsi_4(arch)
    port map(
      ins => constant4_outs,
      ins_valid => constant4_outs_valid,
      ins_ready => constant4_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi11_outs,
      outs_valid => extsi11_outs_valid,
      outs_ready => extsi11_outs_ready
    );

  source3 : entity work.handshake_source_0(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source3_outs_valid,
      outs_ready => source3_outs_ready
    );

  constant5 : entity work.handshake_constant_6(arch)
    port map(
      ctrl_valid => source3_outs_valid,
      ctrl_ready => source3_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant5_outs,
      outs_valid => constant5_outs_valid,
      outs_ready => constant5_outs_ready
    );

  extsi12 : entity work.handshake_extsi_2(arch)
    port map(
      ins => constant5_outs,
      ins_valid => constant5_outs_valid,
      ins_ready => constant5_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi12_outs,
      outs_valid => extsi12_outs_valid,
      outs_ready => extsi12_outs_ready
    );

  addi1 : entity work.handshake_addi_1(arch)
    port map(
      lhs => extsi10_outs,
      lhs_valid => extsi10_outs_valid,
      lhs_ready => extsi10_outs_ready,
      rhs => extsi11_outs,
      rhs_valid => extsi11_outs_valid,
      rhs_ready => extsi11_outs_ready,
      clk => clk,
      rst => rst,
      result => addi1_result,
      result_valid => addi1_result_valid,
      result_ready => addi1_result_ready
    );

  fork10 : entity work.handshake_fork_6(arch)
    port map(
      ins => addi1_result,
      ins_valid => addi1_result_valid,
      ins_ready => addi1_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork10_outs_0,
      outs(1) => fork10_outs_1,
      outs_valid(0) => fork10_outs_0_valid,
      outs_valid(1) => fork10_outs_1_valid,
      outs_ready(0) => fork10_outs_0_ready,
      outs_ready(1) => fork10_outs_1_ready
    );

  cmpi2 : entity work.handshake_cmpi_2(arch)
    port map(
      lhs => fork10_outs_1,
      lhs_valid => fork10_outs_1_valid,
      lhs_ready => fork10_outs_1_ready,
      rhs => extsi12_outs,
      rhs_valid => extsi12_outs_valid,
      rhs_ready => extsi12_outs_ready,
      clk => clk,
      rst => rst,
      result => cmpi2_result,
      result_valid => cmpi2_result_valid,
      result_ready => cmpi2_result_ready
    );

  fork11 : entity work.handshake_fork_5(arch)
    port map(
      ins => cmpi2_result,
      ins_valid => cmpi2_result_valid,
      ins_ready => cmpi2_result_ready,
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

  cond_br10 : entity work.handshake_cond_br_3(arch)
    port map(
      condition => fork11_outs_0,
      condition_valid => fork11_outs_0_valid,
      condition_ready => fork11_outs_0_ready,
      data => fork10_outs_0,
      data_valid => fork10_outs_0_valid,
      data_ready => fork10_outs_0_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br10_trueOut,
      trueOut_valid => cond_br10_trueOut_valid,
      trueOut_ready => cond_br10_trueOut_ready,
      falseOut => cond_br10_falseOut,
      falseOut_valid => cond_br10_falseOut_valid,
      falseOut_ready => cond_br10_falseOut_ready
    );

  trunci2 : entity work.handshake_trunci_1(arch)
    port map(
      ins => cond_br10_trueOut,
      ins_valid => cond_br10_trueOut_valid,
      ins_ready => cond_br10_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => trunci2_outs,
      outs_valid => trunci2_outs_valid,
      outs_ready => trunci2_outs_ready
    );

  cond_br11 : entity work.handshake_cond_br_2(arch)
    port map(
      condition => fork11_outs_1,
      condition_valid => fork11_outs_1_valid,
      condition_ready => fork11_outs_1_ready,
      data_valid => fork9_outs_1_valid,
      data_ready => fork9_outs_1_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br11_trueOut_valid,
      trueOut_ready => cond_br11_trueOut_ready,
      falseOut_valid => cond_br11_falseOut_valid,
      falseOut_ready => cond_br11_falseOut_ready
    );

  cond_br12 : entity work.handshake_cond_br_4(arch)
    port map(
      condition => fork11_outs_2,
      condition_valid => fork11_outs_2_valid,
      condition_ready => fork11_outs_2_ready,
      data => constant3_outs,
      data_valid => constant3_outs_valid,
      data_ready => constant3_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br12_trueOut,
      trueOut_valid => cond_br12_trueOut_valid,
      trueOut_ready => cond_br12_trueOut_ready,
      falseOut => cond_br12_falseOut,
      falseOut_valid => cond_br12_falseOut_valid,
      falseOut_ready => cond_br12_falseOut_ready
    );

  sink5 : entity work.handshake_sink_1(arch)
    port map(
      ins => cond_br12_trueOut,
      ins_valid => cond_br12_trueOut_valid,
      ins_ready => cond_br12_trueOut_ready,
      clk => clk,
      rst => rst
    );

  extsi13 : entity work.handshake_extsi_5(arch)
    port map(
      ins => cond_br12_falseOut,
      ins_valid => cond_br12_falseOut_valid,
      ins_ready => cond_br12_falseOut_ready,
      clk => clk,
      rst => rst,
      outs => extsi13_outs,
      outs_valid => extsi13_outs_valid,
      outs_ready => extsi13_outs_ready
    );

  mux1 : entity work.handshake_mux_1(arch)
    port map(
      index => fork12_outs_0,
      index_valid => fork12_outs_0_valid,
      index_ready => fork12_outs_0_ready,
      ins(0) => extsi8_outs,
      ins(1) => cond_br10_falseOut,
      ins_valid(0) => extsi8_outs_valid,
      ins_valid(1) => cond_br10_falseOut_valid,
      ins_ready(0) => extsi8_outs_ready,
      ins_ready(1) => cond_br10_falseOut_ready,
      clk => clk,
      rst => rst,
      outs => mux1_outs,
      outs_valid => mux1_outs_valid,
      outs_ready => mux1_outs_ready
    );

  mux2 : entity work.handshake_mux_2(arch)
    port map(
      index => fork12_outs_1,
      index_valid => fork12_outs_1_valid,
      index_ready => fork12_outs_1_ready,
      ins(0) => cond_br4_trueOut,
      ins(1) => extsi13_outs,
      ins_valid(0) => cond_br4_trueOut_valid,
      ins_valid(1) => extsi13_outs_valid,
      ins_ready(0) => cond_br4_trueOut_ready,
      ins_ready(1) => extsi13_outs_ready,
      clk => clk,
      rst => rst,
      outs => mux2_outs,
      outs_valid => mux2_outs_valid,
      outs_ready => mux2_outs_ready
    );

  control_merge3 : entity work.handshake_control_merge_0(arch)
    port map(
      ins_valid(0) => cond_br5_trueOut_valid,
      ins_valid(1) => cond_br11_falseOut_valid,
      ins_ready(0) => cond_br5_trueOut_ready,
      ins_ready(1) => cond_br11_falseOut_ready,
      clk => clk,
      rst => rst,
      outs_valid => control_merge3_outs_valid,
      outs_ready => control_merge3_outs_ready,
      index => control_merge3_index,
      index_valid => control_merge3_index_valid,
      index_ready => control_merge3_index_ready
    );

  fork12 : entity work.handshake_fork_7(arch)
    port map(
      ins => control_merge3_index,
      ins_valid => control_merge3_index_valid,
      ins_ready => control_merge3_index_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork12_outs_0,
      outs(1) => fork12_outs_1,
      outs_valid(0) => fork12_outs_0_valid,
      outs_valid(1) => fork12_outs_1_valid,
      outs_ready(0) => fork12_outs_0_ready,
      outs_ready(1) => fork12_outs_1_ready
    );

  buffer6 : entity work.handshake_buffer_4(arch)
    port map(
      ins => mux1_outs,
      ins_valid => mux1_outs_valid,
      ins_ready => mux1_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer6_outs,
      outs_valid => buffer6_outs_valid,
      outs_ready => buffer6_outs_ready
    );

  mux3 : entity work.handshake_mux_1(arch)
    port map(
      index => fork13_outs_0,
      index_valid => fork13_outs_0_valid,
      index_ready => fork13_outs_0_ready,
      ins(0) => extsi9_outs,
      ins(1) => buffer6_outs,
      ins_valid(0) => extsi9_outs_valid,
      ins_valid(1) => buffer6_outs_valid,
      ins_ready(0) => extsi9_outs_ready,
      ins_ready(1) => buffer6_outs_ready,
      clk => clk,
      rst => rst,
      outs => mux3_outs,
      outs_valid => mux3_outs_valid,
      outs_ready => mux3_outs_ready
    );

  extsi14 : entity work.handshake_extsi_6(arch)
    port map(
      ins => mux3_outs,
      ins_valid => mux3_outs_valid,
      ins_ready => mux3_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi14_outs,
      outs_valid => extsi14_outs_valid,
      outs_ready => extsi14_outs_ready
    );

  buffer7 : entity work.handshake_buffer_5(arch)
    port map(
      ins => mux2_outs,
      ins_valid => mux2_outs_valid,
      ins_ready => mux2_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer7_outs,
      outs_valid => buffer7_outs_valid,
      outs_ready => buffer7_outs_ready
    );

  mux4 : entity work.handshake_mux_2(arch)
    port map(
      index => fork13_outs_1,
      index_valid => fork13_outs_1_valid,
      index_ready => fork13_outs_1_ready,
      ins(0) => cond_br8_trueOut,
      ins(1) => buffer7_outs,
      ins_valid(0) => cond_br8_trueOut_valid,
      ins_valid(1) => buffer7_outs_valid,
      ins_ready(0) => cond_br8_trueOut_ready,
      ins_ready(1) => buffer7_outs_ready,
      clk => clk,
      rst => rst,
      outs => mux4_outs,
      outs_valid => mux4_outs_valid,
      outs_ready => mux4_outs_ready
    );

  control_merge4 : entity work.handshake_control_merge_0(arch)
    port map(
      ins_valid(0) => cond_br9_trueOut_valid,
      ins_valid(1) => control_merge3_outs_valid,
      ins_ready(0) => cond_br9_trueOut_ready,
      ins_ready(1) => control_merge3_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => control_merge4_outs_valid,
      outs_ready => control_merge4_outs_ready,
      index => control_merge4_index,
      index_valid => control_merge4_index_valid,
      index_ready => control_merge4_index_ready
    );

  buffer8 : entity work.handshake_buffer_3(arch)
    port map(
      ins => control_merge4_index,
      ins_valid => control_merge4_index_valid,
      ins_ready => control_merge4_index_ready,
      clk => clk,
      rst => rst,
      outs => buffer8_outs,
      outs_valid => buffer8_outs_valid,
      outs_ready => buffer8_outs_ready
    );

  fork13 : entity work.handshake_fork_7(arch)
    port map(
      ins => buffer8_outs,
      ins_valid => buffer8_outs_valid,
      ins_ready => buffer8_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork13_outs_0,
      outs(1) => fork13_outs_1,
      outs_valid(0) => fork13_outs_0_valid,
      outs_valid(1) => fork13_outs_1_valid,
      outs_ready(0) => fork13_outs_0_ready,
      outs_ready(1) => fork13_outs_1_ready
    );

  fork14 : entity work.handshake_fork_2(arch)
    port map(
      ins_valid => control_merge4_outs_valid,
      ins_ready => control_merge4_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork14_outs_0_valid,
      outs_valid(1) => fork14_outs_1_valid,
      outs_ready(0) => fork14_outs_0_ready,
      outs_ready(1) => fork14_outs_1_ready
    );

  source4 : entity work.handshake_source_0(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source4_outs_valid,
      outs_ready => source4_outs_ready
    );

  constant15 : entity work.handshake_constant_5(arch)
    port map(
      ctrl_valid => source4_outs_valid,
      ctrl_ready => source4_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant15_outs,
      outs_valid => constant15_outs_valid,
      outs_ready => constant15_outs_ready
    );

  extsi6 : entity work.handshake_extsi_7(arch)
    port map(
      ins => constant15_outs,
      ins_valid => constant15_outs_valid,
      ins_ready => constant15_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi6_outs,
      outs_valid => extsi6_outs_valid,
      outs_ready => extsi6_outs_ready
    );

  shli0 : entity work.handshake_shli_0(arch)
    port map(
      lhs => extsi14_outs,
      lhs_valid => extsi14_outs_valid,
      lhs_ready => extsi14_outs_ready,
      rhs => extsi6_outs,
      rhs_valid => extsi6_outs_valid,
      rhs_ready => extsi6_outs_ready,
      clk => clk,
      rst => rst,
      result => shli0_result,
      result_valid => shli0_result_valid,
      result_ready => shli0_result_ready
    );

  andi0 : entity work.handshake_andi_0(arch)
    port map(
      lhs => shli0_result,
      lhs_valid => shli0_result_valid,
      lhs_ready => shli0_result_ready,
      rhs => mux4_outs,
      rhs_valid => mux4_outs_valid,
      rhs_ready => mux4_outs_ready,
      clk => clk,
      rst => rst,
      result => andi0_result,
      result_valid => andi0_result_valid,
      result_ready => andi0_result_ready
    );

end architecture;
