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
  signal buffer10_outs : std_logic_vector(0 downto 0);
  signal buffer10_outs_valid : std_logic;
  signal buffer10_outs_ready : std_logic;
  signal init0_outs : std_logic_vector(0 downto 0);
  signal init0_outs_valid : std_logic;
  signal init0_outs_ready : std_logic;
  signal buffer9_outs : std_logic_vector(0 downto 0);
  signal buffer9_outs_valid : std_logic;
  signal buffer9_outs_ready : std_logic;
  signal init1_outs : std_logic_vector(0 downto 0);
  signal init1_outs_valid : std_logic;
  signal init1_outs_ready : std_logic;
  signal mux0_outs : std_logic_vector(10 downto 0);
  signal mux0_outs_valid : std_logic;
  signal mux0_outs_ready : std_logic;
  signal buffer0_outs : std_logic_vector(10 downto 0);
  signal buffer0_outs_valid : std_logic;
  signal buffer0_outs_ready : std_logic;
  signal fork31_outs_0 : std_logic_vector(10 downto 0);
  signal fork31_outs_0_valid : std_logic;
  signal fork31_outs_0_ready : std_logic;
  signal fork31_outs_1 : std_logic_vector(10 downto 0);
  signal fork31_outs_1_valid : std_logic;
  signal fork31_outs_1_ready : std_logic;
  signal fork31_outs_2 : std_logic_vector(10 downto 0);
  signal fork31_outs_2_valid : std_logic;
  signal fork31_outs_2_ready : std_logic;
  signal fork31_outs_3 : std_logic_vector(10 downto 0);
  signal fork31_outs_3_valid : std_logic;
  signal fork31_outs_3_ready : std_logic;
  signal fork31_outs_4 : std_logic_vector(10 downto 0);
  signal fork31_outs_4_valid : std_logic;
  signal fork31_outs_4_ready : std_logic;
  signal trunci0_outs : std_logic_vector(9 downto 0);
  signal trunci0_outs_valid : std_logic;
  signal trunci0_outs_ready : std_logic;
  signal trunci1_outs : std_logic_vector(9 downto 0);
  signal trunci1_outs_valid : std_logic;
  signal trunci1_outs_ready : std_logic;
  signal mux7_outs_valid : std_logic;
  signal mux7_outs_ready : std_logic;
  signal buffer3_outs_valid : std_logic;
  signal buffer3_outs_ready : std_logic;
  signal buffer4_outs_valid : std_logic;
  signal buffer4_outs_ready : std_logic;
  signal fork32_outs_0_valid : std_logic;
  signal fork32_outs_0_ready : std_logic;
  signal fork32_outs_1_valid : std_logic;
  signal fork32_outs_1_ready : std_logic;
  signal fork32_outs_2_valid : std_logic;
  signal fork32_outs_2_ready : std_logic;
  signal fork32_outs_3_valid : std_logic;
  signal fork32_outs_3_ready : std_logic;
  signal fork32_outs_4_valid : std_logic;
  signal fork32_outs_4_ready : std_logic;
  signal fork32_outs_5_valid : std_logic;
  signal fork32_outs_5_ready : std_logic;
  signal fork32_outs_6_valid : std_logic;
  signal fork32_outs_6_ready : std_logic;
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
  signal buffer5_outs : std_logic_vector(0 downto 0);
  signal buffer5_outs_valid : std_logic;
  signal buffer5_outs_ready : std_logic;
  signal fork33_outs_0 : std_logic_vector(0 downto 0);
  signal fork33_outs_0_valid : std_logic;
  signal fork33_outs_0_ready : std_logic;
  signal fork33_outs_1 : std_logic_vector(0 downto 0);
  signal fork33_outs_1_valid : std_logic;
  signal fork33_outs_1_ready : std_logic;
  signal fork33_outs_2 : std_logic_vector(0 downto 0);
  signal fork33_outs_2_valid : std_logic;
  signal fork33_outs_2_ready : std_logic;
  signal fork33_outs_3 : std_logic_vector(0 downto 0);
  signal fork33_outs_3_valid : std_logic;
  signal fork33_outs_3_ready : std_logic;
  signal not0_outs : std_logic_vector(0 downto 0);
  signal not0_outs_valid : std_logic;
  signal not0_outs_ready : std_logic;
  signal fork2_outs_0 : std_logic_vector(0 downto 0);
  signal fork2_outs_0_valid : std_logic;
  signal fork2_outs_0_ready : std_logic;
  signal fork2_outs_1 : std_logic_vector(0 downto 0);
  signal fork2_outs_1_valid : std_logic;
  signal fork2_outs_1_ready : std_logic;
  signal andi1_result : std_logic_vector(0 downto 0);
  signal andi1_result_valid : std_logic;
  signal andi1_result_ready : std_logic;
  signal fork34_outs_0 : std_logic_vector(0 downto 0);
  signal fork34_outs_0_valid : std_logic;
  signal fork34_outs_0_ready : std_logic;
  signal fork34_outs_1 : std_logic_vector(0 downto 0);
  signal fork34_outs_1_valid : std_logic;
  signal fork34_outs_1_ready : std_logic;
  signal fork34_outs_2 : std_logic_vector(0 downto 0);
  signal fork34_outs_2_valid : std_logic;
  signal fork34_outs_2_ready : std_logic;
  signal andi2_result : std_logic_vector(0 downto 0);
  signal andi2_result_valid : std_logic;
  signal andi2_result_ready : std_logic;
  signal fork35_outs_0 : std_logic_vector(0 downto 0);
  signal fork35_outs_0_valid : std_logic;
  signal fork35_outs_0_ready : std_logic;
  signal fork35_outs_1 : std_logic_vector(0 downto 0);
  signal fork35_outs_1_valid : std_logic;
  signal fork35_outs_1_ready : std_logic;
  signal fork35_outs_2 : std_logic_vector(0 downto 0);
  signal fork35_outs_2_valid : std_logic;
  signal fork35_outs_2_ready : std_logic;
  signal buffer6_outs : std_logic_vector(11 downto 0);
  signal buffer6_outs_valid : std_logic;
  signal buffer6_outs_ready : std_logic;
  signal passer16_result : std_logic_vector(11 downto 0);
  signal passer16_result_valid : std_logic;
  signal passer16_result_ready : std_logic;
  signal extsi8_outs : std_logic_vector(11 downto 0);
  signal extsi8_outs_valid : std_logic;
  signal extsi8_outs_ready : std_logic;
  signal passer2_result : std_logic_vector(31 downto 0);
  signal passer2_result_valid : std_logic;
  signal passer2_result_ready : std_logic;
  signal passer3_result_valid : std_logic;
  signal passer3_result_ready : std_logic;
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
  signal buffer7_outs : std_logic_vector(0 downto 0);
  signal buffer7_outs_valid : std_logic;
  signal buffer7_outs_ready : std_logic;
  signal fork10_outs_0 : std_logic_vector(0 downto 0);
  signal fork10_outs_0_valid : std_logic;
  signal fork10_outs_0_ready : std_logic;
  signal fork10_outs_1 : std_logic_vector(0 downto 0);
  signal fork10_outs_1_valid : std_logic;
  signal fork10_outs_1_ready : std_logic;
  signal not1_outs : std_logic_vector(0 downto 0);
  signal not1_outs_valid : std_logic;
  signal not1_outs_ready : std_logic;
  signal andi3_result : std_logic_vector(0 downto 0);
  signal andi3_result_valid : std_logic;
  signal andi3_result_ready : std_logic;
  signal fork11_outs_0 : std_logic_vector(0 downto 0);
  signal fork11_outs_0_valid : std_logic;
  signal fork11_outs_0_ready : std_logic;
  signal fork11_outs_1 : std_logic_vector(0 downto 0);
  signal fork11_outs_1_valid : std_logic;
  signal fork11_outs_1_ready : std_logic;
  signal passer17_result : std_logic_vector(11 downto 0);
  signal passer17_result_valid : std_logic;
  signal passer17_result_ready : std_logic;
  signal buffer2_outs : std_logic_vector(10 downto 0);
  signal buffer2_outs_valid : std_logic;
  signal buffer2_outs_ready : std_logic;
  signal extsi9_outs : std_logic_vector(11 downto 0);
  signal extsi9_outs_valid : std_logic;
  signal extsi9_outs_ready : std_logic;
  signal passer8_result : std_logic_vector(31 downto 0);
  signal passer8_result_valid : std_logic;
  signal passer8_result_ready : std_logic;
  signal passer9_result_valid : std_logic;
  signal passer9_result_ready : std_logic;
  signal buffer1_outs : std_logic_vector(10 downto 0);
  signal buffer1_outs_valid : std_logic;
  signal buffer1_outs_ready : std_logic;
  signal extsi10_outs : std_logic_vector(11 downto 0);
  signal extsi10_outs_valid : std_logic;
  signal extsi10_outs_ready : std_logic;
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
  signal buffer8_outs : std_logic_vector(11 downto 0);
  signal buffer8_outs_valid : std_logic;
  signal buffer8_outs_ready : std_logic;
  signal fork28_outs_0 : std_logic_vector(11 downto 0);
  signal fork28_outs_0_valid : std_logic;
  signal fork28_outs_0_ready : std_logic;
  signal fork28_outs_1 : std_logic_vector(11 downto 0);
  signal fork28_outs_1_valid : std_logic;
  signal fork28_outs_1_ready : std_logic;
  signal fork28_outs_2 : std_logic_vector(11 downto 0);
  signal fork28_outs_2_valid : std_logic;
  signal fork28_outs_2_ready : std_logic;
  signal addi1_result : std_logic_vector(11 downto 0);
  signal addi1_result_valid : std_logic;
  signal addi1_result_ready : std_logic;
  signal cmpi2_result : std_logic_vector(0 downto 0);
  signal cmpi2_result_valid : std_logic;
  signal cmpi2_result_ready : std_logic;
  signal fork29_outs_0 : std_logic_vector(0 downto 0);
  signal fork29_outs_0_valid : std_logic;
  signal fork29_outs_0_ready : std_logic;
  signal fork29_outs_1 : std_logic_vector(0 downto 0);
  signal fork29_outs_1_valid : std_logic;
  signal fork29_outs_1_ready : std_logic;
  signal andi4_result : std_logic_vector(0 downto 0);
  signal andi4_result_valid : std_logic;
  signal andi4_result_ready : std_logic;
  signal fork30_outs_0 : std_logic_vector(0 downto 0);
  signal fork30_outs_0_valid : std_logic;
  signal fork30_outs_0_ready : std_logic;
  signal fork30_outs_1 : std_logic_vector(0 downto 0);
  signal fork30_outs_1_valid : std_logic;
  signal fork30_outs_1_ready : std_logic;
  signal fork30_outs_2 : std_logic_vector(0 downto 0);
  signal fork30_outs_2_valid : std_logic;
  signal fork30_outs_2_ready : std_logic;
  signal fork30_outs_3 : std_logic_vector(0 downto 0);
  signal fork30_outs_3_valid : std_logic;
  signal fork30_outs_3_ready : std_logic;
  signal not2_outs : std_logic_vector(0 downto 0);
  signal not2_outs_valid : std_logic;
  signal not2_outs_ready : std_logic;
  signal passer12_result : std_logic_vector(11 downto 0);
  signal passer12_result_valid : std_logic;
  signal passer12_result_ready : std_logic;
  signal passer18_result : std_logic_vector(10 downto 0);
  signal passer18_result_valid : std_logic;
  signal passer18_result_ready : std_logic;
  signal trunci2_outs : std_logic_vector(10 downto 0);
  signal trunci2_outs_valid : std_logic;
  signal trunci2_outs_ready : std_logic;
  signal passer13_result_valid : std_logic;
  signal passer13_result_ready : std_logic;
  signal passer14_result_valid : std_logic;
  signal passer14_result_ready : std_logic;
  signal passer19_result : std_logic_vector(31 downto 0);
  signal passer19_result_valid : std_logic;
  signal passer19_result_ready : std_logic;
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
  signal mux3_outs : std_logic_vector(11 downto 0);
  signal mux3_outs_valid : std_logic;
  signal mux3_outs_ready : std_logic;
  signal buffer12_outs : std_logic_vector(11 downto 0);
  signal buffer12_outs_valid : std_logic;
  signal buffer12_outs_ready : std_logic;
  signal extsi14_outs : std_logic_vector(31 downto 0);
  signal extsi14_outs_valid : std_logic;
  signal extsi14_outs_ready : std_logic;
  signal buffer11_outs : std_logic_vector(31 downto 0);
  signal buffer11_outs_valid : std_logic;
  signal buffer11_outs_ready : std_logic;
  signal mux4_outs : std_logic_vector(31 downto 0);
  signal mux4_outs_valid : std_logic;
  signal mux4_outs_ready : std_logic;
  signal control_merge4_outs_valid : std_logic;
  signal control_merge4_outs_ready : std_logic;
  signal control_merge4_index : std_logic_vector(0 downto 0);
  signal control_merge4_index_valid : std_logic;
  signal control_merge4_index_ready : std_logic;
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

  buffer10 : entity work.handshake_buffer_0(arch)
    port map(
      ins => fork30_outs_1,
      ins_valid => fork30_outs_1_valid,
      ins_ready => fork30_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => buffer10_outs,
      outs_valid => buffer10_outs_valid,
      outs_ready => buffer10_outs_ready
    );

  init0 : entity work.handshake_init_0(arch)
    port map(
      ins => buffer10_outs,
      ins_valid => buffer10_outs_valid,
      ins_ready => buffer10_outs_ready,
      clk => clk,
      rst => rst,
      outs => init0_outs,
      outs_valid => init0_outs_valid,
      outs_ready => init0_outs_ready
    );

  buffer9 : entity work.handshake_buffer_0(arch)
    port map(
      ins => fork30_outs_0,
      ins_valid => fork30_outs_0_valid,
      ins_ready => fork30_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => buffer9_outs,
      outs_valid => buffer9_outs_valid,
      outs_ready => buffer9_outs_ready
    );

  init1 : entity work.handshake_init_0(arch)
    port map(
      ins => buffer9_outs,
      ins_valid => buffer9_outs_valid,
      ins_ready => buffer9_outs_ready,
      clk => clk,
      rst => rst,
      outs => init1_outs,
      outs_valid => init1_outs_valid,
      outs_ready => init1_outs_ready
    );

  mux0 : entity work.handshake_mux_0(arch)
    port map(
      index => init0_outs,
      index_valid => init0_outs_valid,
      index_ready => init0_outs_ready,
      ins(0) => extsi7_outs,
      ins(1) => passer18_result,
      ins_valid(0) => extsi7_outs_valid,
      ins_valid(1) => passer18_result_valid,
      ins_ready(0) => extsi7_outs_ready,
      ins_ready(1) => passer18_result_ready,
      clk => clk,
      rst => rst,
      outs => mux0_outs,
      outs_valid => mux0_outs_valid,
      outs_ready => mux0_outs_ready
    );

  buffer0 : entity work.handshake_buffer_1(arch)
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

  fork31 : entity work.handshake_fork_1(arch)
    port map(
      ins => buffer0_outs,
      ins_valid => buffer0_outs_valid,
      ins_ready => buffer0_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork31_outs_0,
      outs(1) => fork31_outs_1,
      outs(2) => fork31_outs_2,
      outs(3) => fork31_outs_3,
      outs(4) => fork31_outs_4,
      outs_valid(0) => fork31_outs_0_valid,
      outs_valid(1) => fork31_outs_1_valid,
      outs_valid(2) => fork31_outs_2_valid,
      outs_valid(3) => fork31_outs_3_valid,
      outs_valid(4) => fork31_outs_4_valid,
      outs_ready(0) => fork31_outs_0_ready,
      outs_ready(1) => fork31_outs_1_ready,
      outs_ready(2) => fork31_outs_2_ready,
      outs_ready(3) => fork31_outs_3_ready,
      outs_ready(4) => fork31_outs_4_ready
    );

  trunci0 : entity work.handshake_trunci_0(arch)
    port map(
      ins => fork31_outs_0,
      ins_valid => fork31_outs_0_valid,
      ins_ready => fork31_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => trunci0_outs,
      outs_valid => trunci0_outs_valid,
      outs_ready => trunci0_outs_ready
    );

  trunci1 : entity work.handshake_trunci_0(arch)
    port map(
      ins => fork31_outs_1,
      ins_valid => fork31_outs_1_valid,
      ins_ready => fork31_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => trunci1_outs,
      outs_valid => trunci1_outs_valid,
      outs_ready => trunci1_outs_ready
    );

  mux7 : entity work.handshake_mux_1(arch)
    port map(
      index => init1_outs,
      index_valid => init1_outs_valid,
      index_ready => init1_outs_ready,
      ins_valid(0) => fork0_outs_2_valid,
      ins_valid(1) => passer13_result_valid,
      ins_ready(0) => fork0_outs_2_ready,
      ins_ready(1) => passer13_result_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux7_outs_valid,
      outs_ready => mux7_outs_ready
    );

  buffer3 : entity work.handshake_buffer_2(arch)
    port map(
      ins_valid => mux7_outs_valid,
      ins_ready => mux7_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer3_outs_valid,
      outs_ready => buffer3_outs_ready
    );

  buffer4 : entity work.handshake_buffer_3(arch)
    port map(
      ins_valid => buffer3_outs_valid,
      ins_ready => buffer3_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer4_outs_valid,
      outs_ready => buffer4_outs_ready
    );

  fork32 : entity work.handshake_fork_2(arch)
    port map(
      ins_valid => buffer4_outs_valid,
      ins_ready => buffer4_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork32_outs_0_valid,
      outs_valid(1) => fork32_outs_1_valid,
      outs_valid(2) => fork32_outs_2_valid,
      outs_valid(3) => fork32_outs_3_valid,
      outs_valid(4) => fork32_outs_4_valid,
      outs_valid(5) => fork32_outs_5_valid,
      outs_valid(6) => fork32_outs_6_valid,
      outs_ready(0) => fork32_outs_0_ready,
      outs_ready(1) => fork32_outs_1_ready,
      outs_ready(2) => fork32_outs_2_ready,
      outs_ready(3) => fork32_outs_3_ready,
      outs_ready(4) => fork32_outs_4_ready,
      outs_ready(5) => fork32_outs_5_ready,
      outs_ready(6) => fork32_outs_6_ready
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
      ctrl_valid => fork32_outs_0_valid,
      ctrl_ready => fork32_outs_0_ready,
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

  buffer5 : entity work.handshake_buffer_4(arch)
    port map(
      ins => cmpi0_result,
      ins_valid => cmpi0_result_valid,
      ins_ready => cmpi0_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer5_outs,
      outs_valid => buffer5_outs_valid,
      outs_ready => buffer5_outs_ready
    );

  fork33 : entity work.handshake_fork_4(arch)
    port map(
      ins => buffer5_outs,
      ins_valid => buffer5_outs_valid,
      ins_ready => buffer5_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork33_outs_0,
      outs(1) => fork33_outs_1,
      outs(2) => fork33_outs_2,
      outs(3) => fork33_outs_3,
      outs_valid(0) => fork33_outs_0_valid,
      outs_valid(1) => fork33_outs_1_valid,
      outs_valid(2) => fork33_outs_2_valid,
      outs_valid(3) => fork33_outs_3_valid,
      outs_ready(0) => fork33_outs_0_ready,
      outs_ready(1) => fork33_outs_1_ready,
      outs_ready(2) => fork33_outs_2_ready,
      outs_ready(3) => fork33_outs_3_ready
    );

  not0 : entity work.handshake_not_0(arch)
    port map(
      ins => fork33_outs_3,
      ins_valid => fork33_outs_3_valid,
      ins_ready => fork33_outs_3_ready,
      clk => clk,
      rst => rst,
      outs => not0_outs,
      outs_valid => not0_outs_valid,
      outs_ready => not0_outs_ready
    );

  fork2 : entity work.handshake_fork_5(arch)
    port map(
      ins => not0_outs,
      ins_valid => not0_outs_valid,
      ins_ready => not0_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork2_outs_0,
      outs(1) => fork2_outs_1,
      outs_valid(0) => fork2_outs_0_valid,
      outs_valid(1) => fork2_outs_1_valid,
      outs_ready(0) => fork2_outs_0_ready,
      outs_ready(1) => fork2_outs_1_ready
    );

  andi1 : entity work.handshake_andi_0(arch)
    port map(
      lhs => fork10_outs_1,
      lhs_valid => fork10_outs_1_valid,
      lhs_ready => fork10_outs_1_ready,
      rhs => fork2_outs_0,
      rhs_valid => fork2_outs_0_valid,
      rhs_ready => fork2_outs_0_ready,
      clk => clk,
      rst => rst,
      result => andi1_result,
      result_valid => andi1_result_valid,
      result_ready => andi1_result_ready
    );

  fork34 : entity work.handshake_fork_6(arch)
    port map(
      ins => andi1_result,
      ins_valid => andi1_result_valid,
      ins_ready => andi1_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork34_outs_0,
      outs(1) => fork34_outs_1,
      outs(2) => fork34_outs_2,
      outs_valid(0) => fork34_outs_0_valid,
      outs_valid(1) => fork34_outs_1_valid,
      outs_valid(2) => fork34_outs_2_valid,
      outs_ready(0) => fork34_outs_0_ready,
      outs_ready(1) => fork34_outs_1_ready,
      outs_ready(2) => fork34_outs_2_ready
    );

  andi2 : entity work.handshake_andi_0(arch)
    port map(
      lhs => not2_outs,
      lhs_valid => not2_outs_valid,
      lhs_ready => not2_outs_ready,
      rhs => fork11_outs_0,
      rhs_valid => fork11_outs_0_valid,
      rhs_ready => fork11_outs_0_ready,
      clk => clk,
      rst => rst,
      result => andi2_result,
      result_valid => andi2_result_valid,
      result_ready => andi2_result_ready
    );

  fork35 : entity work.handshake_fork_6(arch)
    port map(
      ins => andi2_result,
      ins_valid => andi2_result_valid,
      ins_ready => andi2_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork35_outs_0,
      outs(1) => fork35_outs_1,
      outs(2) => fork35_outs_2,
      outs_valid(0) => fork35_outs_0_valid,
      outs_valid(1) => fork35_outs_1_valid,
      outs_valid(2) => fork35_outs_2_valid,
      outs_ready(0) => fork35_outs_0_ready,
      outs_ready(1) => fork35_outs_1_ready,
      outs_ready(2) => fork35_outs_2_ready
    );

  buffer6 : entity work.handshake_buffer_5(arch)
    port map(
      ins => extsi8_outs,
      ins_valid => extsi8_outs_valid,
      ins_ready => extsi8_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer6_outs,
      outs_valid => buffer6_outs_valid,
      outs_ready => buffer6_outs_ready
    );

  passer16 : entity work.handshake_passer_0(arch)
    port map(
      data => buffer6_outs,
      data_valid => buffer6_outs_valid,
      data_ready => buffer6_outs_ready,
      ctrl => fork33_outs_2,
      ctrl_valid => fork33_outs_2_valid,
      ctrl_ready => fork33_outs_2_ready,
      clk => clk,
      rst => rst,
      result => passer16_result,
      result_valid => passer16_result_valid,
      result_ready => passer16_result_ready
    );

  extsi8 : entity work.handshake_extsi_2(arch)
    port map(
      ins => fork31_outs_2,
      ins_valid => fork31_outs_2_valid,
      ins_ready => fork31_outs_2_ready,
      clk => clk,
      rst => rst,
      outs => extsi8_outs,
      outs_valid => extsi8_outs_valid,
      outs_ready => extsi8_outs_ready
    );

  passer2 : entity work.handshake_passer_1(arch)
    port map(
      data => constant8_outs,
      data_valid => constant8_outs_valid,
      data_ready => constant8_outs_ready,
      ctrl => fork33_outs_1,
      ctrl_valid => fork33_outs_1_valid,
      ctrl_ready => fork33_outs_1_ready,
      clk => clk,
      rst => rst,
      result => passer2_result,
      result_valid => passer2_result_valid,
      result_ready => passer2_result_ready
    );

  passer3 : entity work.handshake_passer_2(arch)
    port map(
      data_valid => fork32_outs_1_valid,
      data_ready => fork32_outs_1_ready,
      ctrl => fork33_outs_0,
      ctrl_valid => fork33_outs_0_valid,
      ctrl_ready => fork33_outs_0_ready,
      clk => clk,
      rst => rst,
      result_valid => passer3_result_valid,
      result_ready => passer3_result_ready
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
      ctrl_valid => fork32_outs_2_valid,
      ctrl_ready => fork32_outs_2_ready,
      clk => clk,
      rst => rst,
      outs => constant10_outs,
      outs_valid => constant10_outs_valid,
      outs_ready => constant10_outs_ready
    );

  cmpi1 : entity work.handshake_cmpi_1(arch)
    port map(
      lhs => fork5_outs_0,
      lhs_valid => fork5_outs_0_valid,
      lhs_ready => fork5_outs_0_ready,
      rhs => extsi2_outs,
      rhs_valid => extsi2_outs_valid,
      rhs_ready => extsi2_outs_ready,
      clk => clk,
      rst => rst,
      result => cmpi1_result,
      result_valid => cmpi1_result_valid,
      result_ready => cmpi1_result_ready
    );

  buffer7 : entity work.handshake_buffer_4(arch)
    port map(
      ins => cmpi1_result,
      ins_valid => cmpi1_result_valid,
      ins_ready => cmpi1_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer7_outs,
      outs_valid => buffer7_outs_valid,
      outs_ready => buffer7_outs_ready
    );

  fork10 : entity work.handshake_fork_5(arch)
    port map(
      ins => buffer7_outs,
      ins_valid => buffer7_outs_valid,
      ins_ready => buffer7_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork10_outs_0,
      outs(1) => fork10_outs_1,
      outs_valid(0) => fork10_outs_0_valid,
      outs_valid(1) => fork10_outs_1_valid,
      outs_ready(0) => fork10_outs_0_ready,
      outs_ready(1) => fork10_outs_1_ready
    );

  not1 : entity work.handshake_not_0(arch)
    port map(
      ins => fork10_outs_0,
      ins_valid => fork10_outs_0_valid,
      ins_ready => fork10_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => not1_outs,
      outs_valid => not1_outs_valid,
      outs_ready => not1_outs_ready
    );

  andi3 : entity work.handshake_andi_0(arch)
    port map(
      lhs => fork2_outs_1,
      lhs_valid => fork2_outs_1_valid,
      lhs_ready => fork2_outs_1_ready,
      rhs => not1_outs,
      rhs_valid => not1_outs_valid,
      rhs_ready => not1_outs_ready,
      clk => clk,
      rst => rst,
      result => andi3_result,
      result_valid => andi3_result_valid,
      result_ready => andi3_result_ready
    );

  fork11 : entity work.handshake_fork_5(arch)
    port map(
      ins => andi3_result,
      ins_valid => andi3_result_valid,
      ins_ready => andi3_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork11_outs_0,
      outs(1) => fork11_outs_1,
      outs_valid(0) => fork11_outs_0_valid,
      outs_valid(1) => fork11_outs_1_valid,
      outs_ready(0) => fork11_outs_0_ready,
      outs_ready(1) => fork11_outs_1_ready
    );

  passer17 : entity work.handshake_passer_0(arch)
    port map(
      data => extsi9_outs,
      data_valid => extsi9_outs_valid,
      data_ready => extsi9_outs_ready,
      ctrl => fork34_outs_2,
      ctrl_valid => fork34_outs_2_valid,
      ctrl_ready => fork34_outs_2_ready,
      clk => clk,
      rst => rst,
      result => passer17_result,
      result_valid => passer17_result_valid,
      result_ready => passer17_result_ready
    );

  buffer2 : entity work.handshake_buffer_6(arch)
    port map(
      ins => fork31_outs_4,
      ins_valid => fork31_outs_4_valid,
      ins_ready => fork31_outs_4_ready,
      clk => clk,
      rst => rst,
      outs => buffer2_outs,
      outs_valid => buffer2_outs_valid,
      outs_ready => buffer2_outs_ready
    );

  extsi9 : entity work.handshake_extsi_2(arch)
    port map(
      ins => buffer2_outs,
      ins_valid => buffer2_outs_valid,
      ins_ready => buffer2_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi9_outs,
      outs_valid => extsi9_outs_valid,
      outs_ready => extsi9_outs_ready
    );

  passer8 : entity work.handshake_passer_1(arch)
    port map(
      data => constant10_outs,
      data_valid => constant10_outs_valid,
      data_ready => constant10_outs_ready,
      ctrl => fork34_outs_0,
      ctrl_valid => fork34_outs_0_valid,
      ctrl_ready => fork34_outs_0_ready,
      clk => clk,
      rst => rst,
      result => passer8_result,
      result_valid => passer8_result_valid,
      result_ready => passer8_result_ready
    );

  passer9 : entity work.handshake_passer_2(arch)
    port map(
      data_valid => fork32_outs_3_valid,
      data_ready => fork32_outs_3_ready,
      ctrl => fork34_outs_1,
      ctrl_valid => fork34_outs_1_valid,
      ctrl_ready => fork34_outs_1_ready,
      clk => clk,
      rst => rst,
      result_valid => passer9_result_valid,
      result_ready => passer9_result_ready
    );

  buffer1 : entity work.handshake_buffer_6(arch)
    port map(
      ins => fork31_outs_3,
      ins_valid => fork31_outs_3_valid,
      ins_ready => fork31_outs_3_ready,
      clk => clk,
      rst => rst,
      outs => buffer1_outs,
      outs_valid => buffer1_outs_valid,
      outs_ready => buffer1_outs_ready
    );

  extsi10 : entity work.handshake_extsi_2(arch)
    port map(
      ins => buffer1_outs,
      ins_valid => buffer1_outs_valid,
      ins_ready => buffer1_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi10_outs,
      outs_valid => extsi10_outs_valid,
      outs_ready => extsi10_outs_ready
    );

  constant3 : entity work.handshake_constant_0(arch)
    port map(
      ctrl_valid => fork32_outs_4_valid,
      ctrl_ready => fork32_outs_4_ready,
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

  buffer8 : entity work.handshake_buffer_5(arch)
    port map(
      ins => addi1_result,
      ins_valid => addi1_result_valid,
      ins_ready => addi1_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer8_outs,
      outs_valid => buffer8_outs_valid,
      outs_ready => buffer8_outs_ready
    );

  fork28 : entity work.handshake_fork_7(arch)
    port map(
      ins => buffer8_outs,
      ins_valid => buffer8_outs_valid,
      ins_ready => buffer8_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork28_outs_0,
      outs(1) => fork28_outs_1,
      outs(2) => fork28_outs_2,
      outs_valid(0) => fork28_outs_0_valid,
      outs_valid(1) => fork28_outs_1_valid,
      outs_valid(2) => fork28_outs_2_valid,
      outs_ready(0) => fork28_outs_0_ready,
      outs_ready(1) => fork28_outs_1_ready,
      outs_ready(2) => fork28_outs_2_ready
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

  cmpi2 : entity work.handshake_cmpi_2(arch)
    port map(
      lhs => fork28_outs_0,
      lhs_valid => fork28_outs_0_valid,
      lhs_ready => fork28_outs_0_ready,
      rhs => extsi12_outs,
      rhs_valid => extsi12_outs_valid,
      rhs_ready => extsi12_outs_ready,
      clk => clk,
      rst => rst,
      result => cmpi2_result,
      result_valid => cmpi2_result_valid,
      result_ready => cmpi2_result_ready
    );

  fork29 : entity work.handshake_fork_5(arch)
    port map(
      ins => cmpi2_result,
      ins_valid => cmpi2_result_valid,
      ins_ready => cmpi2_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork29_outs_0,
      outs(1) => fork29_outs_1,
      outs_valid(0) => fork29_outs_0_valid,
      outs_valid(1) => fork29_outs_1_valid,
      outs_ready(0) => fork29_outs_0_ready,
      outs_ready(1) => fork29_outs_1_ready
    );

  andi4 : entity work.handshake_andi_0(arch)
    port map(
      lhs => fork11_outs_1,
      lhs_valid => fork11_outs_1_valid,
      lhs_ready => fork11_outs_1_ready,
      rhs => fork29_outs_0,
      rhs_valid => fork29_outs_0_valid,
      rhs_ready => fork29_outs_0_ready,
      clk => clk,
      rst => rst,
      result => andi4_result,
      result_valid => andi4_result_valid,
      result_ready => andi4_result_ready
    );

  fork30 : entity work.handshake_fork_4(arch)
    port map(
      ins => andi4_result,
      ins_valid => andi4_result_valid,
      ins_ready => andi4_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork30_outs_0,
      outs(1) => fork30_outs_1,
      outs(2) => fork30_outs_2,
      outs(3) => fork30_outs_3,
      outs_valid(0) => fork30_outs_0_valid,
      outs_valid(1) => fork30_outs_1_valid,
      outs_valid(2) => fork30_outs_2_valid,
      outs_valid(3) => fork30_outs_3_valid,
      outs_ready(0) => fork30_outs_0_ready,
      outs_ready(1) => fork30_outs_1_ready,
      outs_ready(2) => fork30_outs_2_ready,
      outs_ready(3) => fork30_outs_3_ready
    );

  not2 : entity work.handshake_not_0(arch)
    port map(
      ins => fork29_outs_1,
      ins_valid => fork29_outs_1_valid,
      ins_ready => fork29_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => not2_outs,
      outs_valid => not2_outs_valid,
      outs_ready => not2_outs_ready
    );

  passer12 : entity work.handshake_passer_0(arch)
    port map(
      data => fork28_outs_1,
      data_valid => fork28_outs_1_valid,
      data_ready => fork28_outs_1_ready,
      ctrl => fork35_outs_0,
      ctrl_valid => fork35_outs_0_valid,
      ctrl_ready => fork35_outs_0_ready,
      clk => clk,
      rst => rst,
      result => passer12_result,
      result_valid => passer12_result_valid,
      result_ready => passer12_result_ready
    );

  passer18 : entity work.handshake_passer_3(arch)
    port map(
      data => trunci2_outs,
      data_valid => trunci2_outs_valid,
      data_ready => trunci2_outs_ready,
      ctrl => fork30_outs_2,
      ctrl_valid => fork30_outs_2_valid,
      ctrl_ready => fork30_outs_2_ready,
      clk => clk,
      rst => rst,
      result => passer18_result,
      result_valid => passer18_result_valid,
      result_ready => passer18_result_ready
    );

  trunci2 : entity work.handshake_trunci_1(arch)
    port map(
      ins => fork28_outs_2,
      ins_valid => fork28_outs_2_valid,
      ins_ready => fork28_outs_2_ready,
      clk => clk,
      rst => rst,
      outs => trunci2_outs,
      outs_valid => trunci2_outs_valid,
      outs_ready => trunci2_outs_ready
    );

  passer13 : entity work.handshake_passer_2(arch)
    port map(
      data_valid => fork32_outs_6_valid,
      data_ready => fork32_outs_6_ready,
      ctrl => fork30_outs_3,
      ctrl_valid => fork30_outs_3_valid,
      ctrl_ready => fork30_outs_3_ready,
      clk => clk,
      rst => rst,
      result_valid => passer13_result_valid,
      result_ready => passer13_result_ready
    );

  passer14 : entity work.handshake_passer_2(arch)
    port map(
      data_valid => fork32_outs_5_valid,
      data_ready => fork32_outs_5_ready,
      ctrl => fork35_outs_2,
      ctrl_valid => fork35_outs_2_valid,
      ctrl_ready => fork35_outs_2_ready,
      clk => clk,
      rst => rst,
      result_valid => passer14_result_valid,
      result_ready => passer14_result_ready
    );

  passer19 : entity work.handshake_passer_1(arch)
    port map(
      data => extsi13_outs,
      data_valid => extsi13_outs_valid,
      data_ready => extsi13_outs_ready,
      ctrl => fork35_outs_1,
      ctrl_valid => fork35_outs_1_valid,
      ctrl_ready => fork35_outs_1_ready,
      clk => clk,
      rst => rst,
      result => passer19_result,
      result_valid => passer19_result_valid,
      result_ready => passer19_result_ready
    );

  extsi13 : entity work.handshake_extsi_5(arch)
    port map(
      ins => constant3_outs,
      ins_valid => constant3_outs_valid,
      ins_ready => constant3_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi13_outs,
      outs_valid => extsi13_outs_valid,
      outs_ready => extsi13_outs_ready
    );

  mux1 : entity work.handshake_mux_2(arch)
    port map(
      index => fork12_outs_0,
      index_valid => fork12_outs_0_valid,
      index_ready => fork12_outs_0_ready,
      ins(0) => passer16_result,
      ins(1) => passer12_result,
      ins_valid(0) => passer16_result_valid,
      ins_valid(1) => passer12_result_valid,
      ins_ready(0) => passer16_result_ready,
      ins_ready(1) => passer12_result_ready,
      clk => clk,
      rst => rst,
      outs => mux1_outs,
      outs_valid => mux1_outs_valid,
      outs_ready => mux1_outs_ready
    );

  mux2 : entity work.handshake_mux_3(arch)
    port map(
      index => fork12_outs_1,
      index_valid => fork12_outs_1_valid,
      index_ready => fork12_outs_1_ready,
      ins(0) => passer2_result,
      ins(1) => passer19_result,
      ins_valid(0) => passer2_result_valid,
      ins_valid(1) => passer19_result_valid,
      ins_ready(0) => passer2_result_ready,
      ins_ready(1) => passer19_result_ready,
      clk => clk,
      rst => rst,
      outs => mux2_outs,
      outs_valid => mux2_outs_valid,
      outs_ready => mux2_outs_ready
    );

  control_merge3 : entity work.handshake_control_merge_0(arch)
    port map(
      ins_valid(0) => passer3_result_valid,
      ins_valid(1) => passer14_result_valid,
      ins_ready(0) => passer3_result_ready,
      ins_ready(1) => passer14_result_ready,
      clk => clk,
      rst => rst,
      outs_valid => control_merge3_outs_valid,
      outs_ready => control_merge3_outs_ready,
      index => control_merge3_index,
      index_valid => control_merge3_index_valid,
      index_ready => control_merge3_index_ready
    );

  fork12 : entity work.handshake_fork_5(arch)
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

  mux3 : entity work.handshake_mux_2(arch)
    port map(
      index => fork13_outs_0,
      index_valid => fork13_outs_0_valid,
      index_ready => fork13_outs_0_ready,
      ins(0) => passer17_result,
      ins(1) => mux1_outs,
      ins_valid(0) => passer17_result_valid,
      ins_valid(1) => mux1_outs_valid,
      ins_ready(0) => passer17_result_ready,
      ins_ready(1) => mux1_outs_ready,
      clk => clk,
      rst => rst,
      outs => mux3_outs,
      outs_valid => mux3_outs_valid,
      outs_ready => mux3_outs_ready
    );

  buffer12 : entity work.handshake_buffer_5(arch)
    port map(
      ins => mux3_outs,
      ins_valid => mux3_outs_valid,
      ins_ready => mux3_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer12_outs,
      outs_valid => buffer12_outs_valid,
      outs_ready => buffer12_outs_ready
    );

  extsi14 : entity work.handshake_extsi_6(arch)
    port map(
      ins => buffer12_outs,
      ins_valid => buffer12_outs_valid,
      ins_ready => buffer12_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi14_outs,
      outs_valid => extsi14_outs_valid,
      outs_ready => extsi14_outs_ready
    );

  buffer11 : entity work.handshake_buffer_7(arch)
    port map(
      ins => mux2_outs,
      ins_valid => mux2_outs_valid,
      ins_ready => mux2_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer11_outs,
      outs_valid => buffer11_outs_valid,
      outs_ready => buffer11_outs_ready
    );

  mux4 : entity work.handshake_mux_3(arch)
    port map(
      index => fork13_outs_1,
      index_valid => fork13_outs_1_valid,
      index_ready => fork13_outs_1_ready,
      ins(0) => passer8_result,
      ins(1) => buffer11_outs,
      ins_valid(0) => passer8_result_valid,
      ins_valid(1) => buffer11_outs_valid,
      ins_ready(0) => passer8_result_ready,
      ins_ready(1) => buffer11_outs_ready,
      clk => clk,
      rst => rst,
      outs => mux4_outs,
      outs_valid => mux4_outs_valid,
      outs_ready => mux4_outs_ready
    );

  control_merge4 : entity work.handshake_control_merge_0(arch)
    port map(
      ins_valid(0) => passer9_result_valid,
      ins_valid(1) => control_merge3_outs_valid,
      ins_ready(0) => passer9_result_ready,
      ins_ready(1) => control_merge3_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => control_merge4_outs_valid,
      outs_ready => control_merge4_outs_ready,
      index => control_merge4_index,
      index_valid => control_merge4_index_valid,
      index_ready => control_merge4_index_ready
    );

  fork13 : entity work.handshake_fork_5(arch)
    port map(
      ins => control_merge4_index,
      ins_valid => control_merge4_index_valid,
      ins_ready => control_merge4_index_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork13_outs_0,
      outs(1) => fork13_outs_1,
      outs_valid(0) => fork13_outs_0_valid,
      outs_valid(1) => fork13_outs_1_valid,
      outs_ready(0) => fork13_outs_0_ready,
      outs_ready(1) => fork13_outs_1_ready
    );

  fork14 : entity work.handshake_fork_8(arch)
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

  andi0 : entity work.handshake_andi_1(arch)
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
