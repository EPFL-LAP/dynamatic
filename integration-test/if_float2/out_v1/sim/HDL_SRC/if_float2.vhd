library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity if_float2 is
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

architecture behavioral of if_float2 is

  signal non_spec0_dataOut : std_logic_vector(31 downto 0);
  signal non_spec0_dataOut_valid : std_logic;
  signal non_spec0_dataOut_ready : std_logic;
  signal non_spec0_dataOut_spec : std_logic_vector(0 downto 0);
  signal fork0_outs_0_valid : std_logic;
  signal fork0_outs_0_ready : std_logic;
  signal fork0_outs_1_valid : std_logic;
  signal fork0_outs_1_ready : std_logic;
  signal fork0_outs_2_valid : std_logic;
  signal fork0_outs_2_ready : std_logic;
  signal non_spec1_dataOut_valid : std_logic;
  signal non_spec1_dataOut_ready : std_logic;
  signal non_spec1_dataOut_spec : std_logic_vector(0 downto 0);
  signal buffer17_outs : std_logic_vector(0 downto 0);
  signal buffer17_outs_valid : std_logic;
  signal buffer17_outs_ready : std_logic;
  signal buffer18_outs_valid : std_logic;
  signal buffer18_outs_ready : std_logic;
  signal buffer18_outs_spec : std_logic_vector(0 downto 0);
  signal spec_commit0_outs_valid : std_logic;
  signal spec_commit0_outs_ready : std_logic;
  signal buffer19_outs : std_logic_vector(0 downto 0);
  signal buffer19_outs_valid : std_logic;
  signal buffer19_outs_ready : std_logic;
  signal buffer20_outs : std_logic_vector(31 downto 0);
  signal buffer20_outs_valid : std_logic;
  signal buffer20_outs_ready : std_logic;
  signal buffer20_outs_spec : std_logic_vector(0 downto 0);
  signal spec_commit1_outs : std_logic_vector(31 downto 0);
  signal spec_commit1_outs_valid : std_logic;
  signal spec_commit1_outs_ready : std_logic;
  signal mem_controller2_memEnd_valid : std_logic;
  signal mem_controller2_memEnd_ready : std_logic;
  signal mem_controller2_loadEn : std_logic;
  signal mem_controller2_loadAddr : std_logic_vector(6 downto 0);
  signal mem_controller2_storeEn : std_logic;
  signal mem_controller2_storeAddr : std_logic_vector(6 downto 0);
  signal mem_controller2_storeData : std_logic_vector(31 downto 0);
  signal buffer21_outs : std_logic_vector(0 downto 0);
  signal buffer21_outs_valid : std_logic;
  signal buffer21_outs_ready : std_logic;
  signal buffer22_outs_valid : std_logic;
  signal buffer22_outs_ready : std_logic;
  signal buffer22_outs_spec : std_logic_vector(0 downto 0);
  signal spec_commit2_outs_valid : std_logic;
  signal spec_commit2_outs_ready : std_logic;
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
  signal non_spec2_dataOut : std_logic_vector(7 downto 0);
  signal non_spec2_dataOut_valid : std_logic;
  signal non_spec2_dataOut_ready : std_logic;
  signal non_spec2_dataOut_spec : std_logic_vector(0 downto 0);
  signal mux0_outs : std_logic_vector(7 downto 0);
  signal mux0_outs_valid : std_logic;
  signal mux0_outs_ready : std_logic;
  signal mux0_outs_spec : std_logic_vector(0 downto 0);
  signal buffer0_outs : std_logic_vector(7 downto 0);
  signal buffer0_outs_valid : std_logic;
  signal buffer0_outs_ready : std_logic;
  signal buffer0_outs_spec : std_logic_vector(0 downto 0);
  signal fork1_outs_0 : std_logic_vector(7 downto 0);
  signal fork1_outs_0_valid : std_logic;
  signal fork1_outs_0_ready : std_logic;
  signal fork1_outs_0_spec : std_logic_vector(0 downto 0);
  signal fork1_outs_1 : std_logic_vector(7 downto 0);
  signal fork1_outs_1_valid : std_logic;
  signal fork1_outs_1_ready : std_logic;
  signal fork1_outs_1_spec : std_logic_vector(0 downto 0);
  signal trunci0_outs : std_logic_vector(6 downto 0);
  signal trunci0_outs_valid : std_logic;
  signal trunci0_outs_ready : std_logic;
  signal trunci0_outs_spec : std_logic_vector(0 downto 0);
  signal mux1_outs : std_logic_vector(31 downto 0);
  signal mux1_outs_valid : std_logic;
  signal mux1_outs_ready : std_logic;
  signal mux1_outs_spec : std_logic_vector(0 downto 0);
  signal buffer2_outs : std_logic_vector(31 downto 0);
  signal buffer2_outs_valid : std_logic;
  signal buffer2_outs_ready : std_logic;
  signal buffer2_outs_spec : std_logic_vector(0 downto 0);
  signal fork2_outs_0 : std_logic_vector(31 downto 0);
  signal fork2_outs_0_valid : std_logic;
  signal fork2_outs_0_ready : std_logic;
  signal fork2_outs_0_spec : std_logic_vector(0 downto 0);
  signal fork2_outs_1 : std_logic_vector(31 downto 0);
  signal fork2_outs_1_valid : std_logic;
  signal fork2_outs_1_ready : std_logic;
  signal fork2_outs_1_spec : std_logic_vector(0 downto 0);
  signal fork2_outs_2 : std_logic_vector(31 downto 0);
  signal fork2_outs_2_valid : std_logic;
  signal fork2_outs_2_ready : std_logic;
  signal fork2_outs_2_spec : std_logic_vector(0 downto 0);
  signal control_merge0_outs_valid : std_logic;
  signal control_merge0_outs_ready : std_logic;
  signal control_merge0_outs_spec : std_logic_vector(0 downto 0);
  signal control_merge0_index : std_logic_vector(0 downto 0);
  signal control_merge0_index_valid : std_logic;
  signal control_merge0_index_ready : std_logic;
  signal control_merge0_index_spec : std_logic_vector(0 downto 0);
  signal fork4_outs_0_valid : std_logic;
  signal fork4_outs_0_ready : std_logic;
  signal fork4_outs_0_spec : std_logic_vector(0 downto 0);
  signal fork4_outs_1_valid : std_logic;
  signal fork4_outs_1_ready : std_logic;
  signal fork4_outs_1_spec : std_logic_vector(0 downto 0);
  signal fork3_outs_0 : std_logic_vector(0 downto 0);
  signal fork3_outs_0_valid : std_logic;
  signal fork3_outs_0_ready : std_logic;
  signal fork3_outs_0_spec : std_logic_vector(0 downto 0);
  signal fork3_outs_1 : std_logic_vector(0 downto 0);
  signal fork3_outs_1_valid : std_logic;
  signal fork3_outs_1_ready : std_logic;
  signal fork3_outs_1_spec : std_logic_vector(0 downto 0);
  signal source0_outs_valid : std_logic;
  signal source0_outs_ready : std_logic;
  signal source0_outs_spec : std_logic_vector(0 downto 0);
  signal constant7_outs : std_logic_vector(31 downto 0);
  signal constant7_outs_valid : std_logic;
  signal constant7_outs_ready : std_logic;
  signal constant7_outs_spec : std_logic_vector(0 downto 0);
  signal source1_outs_valid : std_logic;
  signal source1_outs_ready : std_logic;
  signal source1_outs_spec : std_logic_vector(0 downto 0);
  signal constant8_outs : std_logic_vector(31 downto 0);
  signal constant8_outs_valid : std_logic;
  signal constant8_outs_ready : std_logic;
  signal constant8_outs_spec : std_logic_vector(0 downto 0);
  signal buffer1_outs : std_logic_vector(6 downto 0);
  signal buffer1_outs_valid : std_logic;
  signal buffer1_outs_ready : std_logic;
  signal buffer1_outs_spec : std_logic_vector(0 downto 0);
  signal load0_addrOut : std_logic_vector(6 downto 0);
  signal load0_addrOut_valid : std_logic;
  signal load0_addrOut_ready : std_logic;
  signal load0_dataOut : std_logic_vector(31 downto 0);
  signal load0_dataOut_valid : std_logic;
  signal load0_dataOut_ready : std_logic;
  signal load0_dataOut_spec : std_logic_vector(0 downto 0);
  signal buffer3_outs : std_logic_vector(31 downto 0);
  signal buffer3_outs_valid : std_logic;
  signal buffer3_outs_ready : std_logic;
  signal buffer3_outs_spec : std_logic_vector(0 downto 0);
  signal mulf0_result : std_logic_vector(31 downto 0);
  signal mulf0_result_valid : std_logic;
  signal mulf0_result_ready : std_logic;
  signal mulf0_result_spec : std_logic_vector(0 downto 0);
  signal mulf1_result : std_logic_vector(31 downto 0);
  signal mulf1_result_valid : std_logic;
  signal mulf1_result_ready : std_logic;
  signal mulf1_result_spec : std_logic_vector(0 downto 0);
  signal buffer4_outs : std_logic_vector(31 downto 0);
  signal buffer4_outs_valid : std_logic;
  signal buffer4_outs_ready : std_logic;
  signal buffer4_outs_spec : std_logic_vector(0 downto 0);
  signal addf0_result : std_logic_vector(31 downto 0);
  signal addf0_result_valid : std_logic;
  signal addf0_result_ready : std_logic;
  signal addf0_result_spec : std_logic_vector(0 downto 0);
  signal cmpf0_result : std_logic_vector(0 downto 0);
  signal cmpf0_result_valid : std_logic;
  signal cmpf0_result_ready : std_logic;
  signal cmpf0_result_spec : std_logic_vector(0 downto 0);
  signal speculator0_outs : std_logic_vector(0 downto 0);
  signal speculator0_outs_valid : std_logic;
  signal speculator0_outs_ready : std_logic;
  signal speculator0_outs_spec : std_logic_vector(0 downto 0);
  signal speculator0_ctrl_save : std_logic_vector(0 downto 0);
  signal speculator0_ctrl_save_valid : std_logic;
  signal speculator0_ctrl_save_ready : std_logic;
  signal speculator0_ctrl_commit : std_logic_vector(0 downto 0);
  signal speculator0_ctrl_commit_valid : std_logic;
  signal speculator0_ctrl_commit_ready : std_logic;
  signal speculator0_ctrl_sc_save : std_logic_vector(2 downto 0);
  signal speculator0_ctrl_sc_save_valid : std_logic;
  signal speculator0_ctrl_sc_save_ready : std_logic;
  signal speculator0_ctrl_sc_commit : std_logic_vector(2 downto 0);
  signal speculator0_ctrl_sc_commit_valid : std_logic;
  signal speculator0_ctrl_sc_commit_ready : std_logic;
  signal speculator0_ctrl_sc_branch : std_logic_vector(0 downto 0);
  signal speculator0_ctrl_sc_branch_valid : std_logic;
  signal speculator0_ctrl_sc_branch_ready : std_logic;
  signal cond_br9_trueOut : std_logic_vector(2 downto 0);
  signal cond_br9_trueOut_valid : std_logic;
  signal cond_br9_trueOut_ready : std_logic;
  signal cond_br9_falseOut : std_logic_vector(2 downto 0);
  signal cond_br9_falseOut_valid : std_logic;
  signal cond_br9_falseOut_ready : std_logic;
  signal merge0_outs : std_logic_vector(2 downto 0);
  signal merge0_outs_valid : std_logic;
  signal merge0_outs_ready : std_logic;
  signal fork8_outs_0 : std_logic_vector(2 downto 0);
  signal fork8_outs_0_valid : std_logic;
  signal fork8_outs_0_ready : std_logic;
  signal fork8_outs_1 : std_logic_vector(2 downto 0);
  signal fork8_outs_1_valid : std_logic;
  signal fork8_outs_1_ready : std_logic;
  signal fork8_outs_2 : std_logic_vector(2 downto 0);
  signal fork8_outs_2_valid : std_logic;
  signal fork8_outs_2_ready : std_logic;
  signal fork12_outs_0 : std_logic_vector(0 downto 0);
  signal fork12_outs_0_valid : std_logic;
  signal fork12_outs_0_ready : std_logic;
  signal fork12_outs_1 : std_logic_vector(0 downto 0);
  signal fork12_outs_1_valid : std_logic;
  signal fork12_outs_1_ready : std_logic;
  signal buffer23_outs : std_logic_vector(0 downto 0);
  signal buffer23_outs_valid : std_logic;
  signal buffer23_outs_ready : std_logic;
  signal cond_br0_trueOut : std_logic_vector(0 downto 0);
  signal cond_br0_trueOut_valid : std_logic;
  signal cond_br0_trueOut_ready : std_logic;
  signal cond_br0_falseOut : std_logic_vector(0 downto 0);
  signal cond_br0_falseOut_valid : std_logic;
  signal cond_br0_falseOut_ready : std_logic;
  signal fork14_outs_0 : std_logic_vector(0 downto 0);
  signal fork14_outs_0_valid : std_logic;
  signal fork14_outs_0_ready : std_logic;
  signal fork14_outs_1 : std_logic_vector(0 downto 0);
  signal fork14_outs_1_valid : std_logic;
  signal fork14_outs_1_ready : std_logic;
  signal fork14_outs_2 : std_logic_vector(0 downto 0);
  signal fork14_outs_2_valid : std_logic;
  signal fork14_outs_2_ready : std_logic;
  signal speculating_branch0_trueOut : std_logic_vector(0 downto 0);
  signal speculating_branch0_trueOut_valid : std_logic;
  signal speculating_branch0_trueOut_ready : std_logic;
  signal speculating_branch0_falseOut : std_logic_vector(0 downto 0);
  signal speculating_branch0_falseOut_valid : std_logic;
  signal speculating_branch0_falseOut_ready : std_logic;
  signal buffer5_outs : std_logic_vector(0 downto 0);
  signal buffer5_outs_valid : std_logic;
  signal buffer5_outs_ready : std_logic;
  signal buffer24_outs : std_logic_vector(0 downto 0);
  signal buffer24_outs_valid : std_logic;
  signal buffer24_outs_ready : std_logic;
  signal cond_br1_trueOut : std_logic_vector(0 downto 0);
  signal cond_br1_trueOut_valid : std_logic;
  signal cond_br1_trueOut_ready : std_logic;
  signal cond_br1_falseOut : std_logic_vector(0 downto 0);
  signal cond_br1_falseOut_valid : std_logic;
  signal cond_br1_falseOut_ready : std_logic;
  signal fork15_outs_0 : std_logic_vector(0 downto 0);
  signal fork15_outs_0_valid : std_logic;
  signal fork15_outs_0_ready : std_logic;
  signal fork15_outs_1 : std_logic_vector(0 downto 0);
  signal fork15_outs_1_valid : std_logic;
  signal fork15_outs_1_ready : std_logic;
  signal fork15_outs_2 : std_logic_vector(0 downto 0);
  signal fork15_outs_2_valid : std_logic;
  signal fork15_outs_2_ready : std_logic;
  signal speculating_branch1_trueOut : std_logic_vector(0 downto 0);
  signal speculating_branch1_trueOut_valid : std_logic;
  signal speculating_branch1_trueOut_ready : std_logic;
  signal speculating_branch1_falseOut : std_logic_vector(0 downto 0);
  signal speculating_branch1_falseOut_valid : std_logic;
  signal speculating_branch1_falseOut_ready : std_logic;
  signal fork16_outs_0 : std_logic_vector(0 downto 0);
  signal fork16_outs_0_valid : std_logic;
  signal fork16_outs_0_ready : std_logic;
  signal fork16_outs_1 : std_logic_vector(0 downto 0);
  signal fork16_outs_1_valid : std_logic;
  signal fork16_outs_1_ready : std_logic;
  signal buffer25_outs : std_logic_vector(0 downto 0);
  signal buffer25_outs_valid : std_logic;
  signal buffer25_outs_ready : std_logic;
  signal cond_br8_trueOut : std_logic_vector(0 downto 0);
  signal cond_br8_trueOut_valid : std_logic;
  signal cond_br8_trueOut_ready : std_logic;
  signal cond_br8_falseOut : std_logic_vector(0 downto 0);
  signal cond_br8_falseOut_valid : std_logic;
  signal cond_br8_falseOut_ready : std_logic;
  signal fork17_outs_0 : std_logic_vector(0 downto 0);
  signal fork17_outs_0_valid : std_logic;
  signal fork17_outs_0_ready : std_logic;
  signal fork17_outs_0_spec : std_logic_vector(0 downto 0);
  signal fork17_outs_1 : std_logic_vector(0 downto 0);
  signal fork17_outs_1_valid : std_logic;
  signal fork17_outs_1_ready : std_logic;
  signal fork17_outs_1_spec : std_logic_vector(0 downto 0);
  signal fork17_outs_2 : std_logic_vector(0 downto 0);
  signal fork17_outs_2_valid : std_logic;
  signal fork17_outs_2_ready : std_logic;
  signal fork17_outs_2_spec : std_logic_vector(0 downto 0);
  signal fork17_outs_3 : std_logic_vector(0 downto 0);
  signal fork17_outs_3_valid : std_logic;
  signal fork17_outs_3_ready : std_logic;
  signal fork17_outs_3_spec : std_logic_vector(0 downto 0);
  signal fork17_outs_4 : std_logic_vector(0 downto 0);
  signal fork17_outs_4_valid : std_logic;
  signal fork17_outs_4_ready : std_logic;
  signal fork17_outs_4_spec : std_logic_vector(0 downto 0);
  signal fork17_outs_5 : std_logic_vector(0 downto 0);
  signal fork17_outs_5_valid : std_logic;
  signal fork17_outs_5_ready : std_logic;
  signal fork17_outs_5_spec : std_logic_vector(0 downto 0);
  signal fork17_outs_6 : std_logic_vector(0 downto 0);
  signal fork17_outs_6_valid : std_logic;
  signal fork17_outs_6_ready : std_logic;
  signal fork17_outs_6_spec : std_logic_vector(0 downto 0);
  signal fork17_outs_7 : std_logic_vector(0 downto 0);
  signal fork17_outs_7_valid : std_logic;
  signal fork17_outs_7_ready : std_logic;
  signal fork17_outs_7_spec : std_logic_vector(0 downto 0);
  signal spec_save_commit0_outs : std_logic_vector(7 downto 0);
  signal spec_save_commit0_outs_valid : std_logic;
  signal spec_save_commit0_outs_ready : std_logic;
  signal spec_save_commit0_outs_spec : std_logic_vector(0 downto 0);
  signal cond_br2_trueOut : std_logic_vector(7 downto 0);
  signal cond_br2_trueOut_valid : std_logic;
  signal cond_br2_trueOut_ready : std_logic;
  signal cond_br2_trueOut_spec : std_logic_vector(0 downto 0);
  signal cond_br2_falseOut : std_logic_vector(7 downto 0);
  signal cond_br2_falseOut_valid : std_logic;
  signal cond_br2_falseOut_ready : std_logic;
  signal cond_br2_falseOut_spec : std_logic_vector(0 downto 0);
  signal spec_save_commit1_outs : std_logic_vector(31 downto 0);
  signal spec_save_commit1_outs_valid : std_logic;
  signal spec_save_commit1_outs_ready : std_logic;
  signal spec_save_commit1_outs_spec : std_logic_vector(0 downto 0);
  signal cond_br3_trueOut : std_logic_vector(31 downto 0);
  signal cond_br3_trueOut_valid : std_logic;
  signal cond_br3_trueOut_ready : std_logic;
  signal cond_br3_trueOut_spec : std_logic_vector(0 downto 0);
  signal cond_br3_falseOut : std_logic_vector(31 downto 0);
  signal cond_br3_falseOut_valid : std_logic;
  signal cond_br3_falseOut_ready : std_logic;
  signal cond_br3_falseOut_spec : std_logic_vector(0 downto 0);
  signal spec_save_commit2_outs_valid : std_logic;
  signal spec_save_commit2_outs_ready : std_logic;
  signal spec_save_commit2_outs_spec : std_logic_vector(0 downto 0);
  signal cond_br4_trueOut_valid : std_logic;
  signal cond_br4_trueOut_ready : std_logic;
  signal cond_br4_trueOut_spec : std_logic_vector(0 downto 0);
  signal cond_br4_falseOut_valid : std_logic;
  signal cond_br4_falseOut_ready : std_logic;
  signal cond_br4_falseOut_spec : std_logic_vector(0 downto 0);
  signal source2_outs_valid : std_logic;
  signal source2_outs_ready : std_logic;
  signal source2_outs_spec : std_logic_vector(0 downto 0);
  signal constant9_outs : std_logic_vector(31 downto 0);
  signal constant9_outs_valid : std_logic;
  signal constant9_outs_ready : std_logic;
  signal constant9_outs_spec : std_logic_vector(0 downto 0);
  signal addf1_result : std_logic_vector(31 downto 0);
  signal addf1_result_valid : std_logic;
  signal addf1_result_ready : std_logic;
  signal addf1_result_spec : std_logic_vector(0 downto 0);
  signal fork5_outs_0 : std_logic_vector(7 downto 0);
  signal fork5_outs_0_valid : std_logic;
  signal fork5_outs_0_ready : std_logic;
  signal fork5_outs_0_spec : std_logic_vector(0 downto 0);
  signal fork5_outs_1 : std_logic_vector(7 downto 0);
  signal fork5_outs_1_valid : std_logic;
  signal fork5_outs_1_ready : std_logic;
  signal fork5_outs_1_spec : std_logic_vector(0 downto 0);
  signal buffer9_outs : std_logic_vector(7 downto 0);
  signal buffer9_outs_valid : std_logic;
  signal buffer9_outs_ready : std_logic;
  signal buffer9_outs_spec : std_logic_vector(0 downto 0);
  signal trunci1_outs : std_logic_vector(6 downto 0);
  signal trunci1_outs_valid : std_logic;
  signal trunci1_outs_ready : std_logic;
  signal trunci1_outs_spec : std_logic_vector(0 downto 0);
  signal fork6_outs_0 : std_logic_vector(31 downto 0);
  signal fork6_outs_0_valid : std_logic;
  signal fork6_outs_0_ready : std_logic;
  signal fork6_outs_0_spec : std_logic_vector(0 downto 0);
  signal fork6_outs_1 : std_logic_vector(31 downto 0);
  signal fork6_outs_1_valid : std_logic;
  signal fork6_outs_1_ready : std_logic;
  signal fork6_outs_1_spec : std_logic_vector(0 downto 0);
  signal fork7_outs_0_valid : std_logic;
  signal fork7_outs_0_ready : std_logic;
  signal fork7_outs_0_spec : std_logic_vector(0 downto 0);
  signal fork7_outs_1_valid : std_logic;
  signal fork7_outs_1_ready : std_logic;
  signal fork7_outs_1_spec : std_logic_vector(0 downto 0);
  signal buffer11_outs_valid : std_logic;
  signal buffer11_outs_ready : std_logic;
  signal buffer11_outs_spec : std_logic_vector(0 downto 0);
  signal constant3_outs : std_logic_vector(1 downto 0);
  signal constant3_outs_valid : std_logic;
  signal constant3_outs_ready : std_logic;
  signal constant3_outs_spec : std_logic_vector(0 downto 0);
  signal extsi1_outs : std_logic_vector(31 downto 0);
  signal extsi1_outs_valid : std_logic;
  signal extsi1_outs_ready : std_logic;
  signal extsi1_outs_spec : std_logic_vector(0 downto 0);
  signal source3_outs_valid : std_logic;
  signal source3_outs_ready : std_logic;
  signal source3_outs_spec : std_logic_vector(0 downto 0);
  signal constant10_outs : std_logic_vector(31 downto 0);
  signal constant10_outs_valid : std_logic;
  signal constant10_outs_ready : std_logic;
  signal constant10_outs_spec : std_logic_vector(0 downto 0);
  signal buffer26_outs : std_logic_vector(0 downto 0);
  signal buffer26_outs_valid : std_logic;
  signal buffer26_outs_ready : std_logic;
  signal buffer27_outs : std_logic_vector(6 downto 0);
  signal buffer27_outs_valid : std_logic;
  signal buffer27_outs_ready : std_logic;
  signal buffer27_outs_spec : std_logic_vector(0 downto 0);
  signal spec_commit3_outs : std_logic_vector(6 downto 0);
  signal spec_commit3_outs_valid : std_logic;
  signal spec_commit3_outs_ready : std_logic;
  signal buffer10_outs : std_logic_vector(31 downto 0);
  signal buffer10_outs_valid : std_logic;
  signal buffer10_outs_ready : std_logic;
  signal buffer10_outs_spec : std_logic_vector(0 downto 0);
  signal buffer28_outs : std_logic_vector(0 downto 0);
  signal buffer28_outs_valid : std_logic;
  signal buffer28_outs_ready : std_logic;
  signal buffer29_outs : std_logic_vector(31 downto 0);
  signal buffer29_outs_valid : std_logic;
  signal buffer29_outs_ready : std_logic;
  signal buffer29_outs_spec : std_logic_vector(0 downto 0);
  signal spec_commit4_outs : std_logic_vector(31 downto 0);
  signal spec_commit4_outs_valid : std_logic;
  signal spec_commit4_outs_ready : std_logic;
  signal store0_addrOut : std_logic_vector(6 downto 0);
  signal store0_addrOut_valid : std_logic;
  signal store0_addrOut_ready : std_logic;
  signal store0_dataToMem : std_logic_vector(31 downto 0);
  signal store0_dataToMem_valid : std_logic;
  signal store0_dataToMem_ready : std_logic;
  signal addf2_result : std_logic_vector(31 downto 0);
  signal addf2_result_valid : std_logic;
  signal addf2_result_ready : std_logic;
  signal addf2_result_spec : std_logic_vector(0 downto 0);
  signal buffer6_outs : std_logic_vector(0 downto 0);
  signal buffer6_outs_valid : std_logic;
  signal buffer6_outs_ready : std_logic;
  signal buffer6_outs_spec : std_logic_vector(0 downto 0);
  signal mux2_outs : std_logic_vector(31 downto 0);
  signal mux2_outs_valid : std_logic;
  signal mux2_outs_ready : std_logic;
  signal mux2_outs_spec : std_logic_vector(0 downto 0);
  signal buffer7_outs : std_logic_vector(7 downto 0);
  signal buffer7_outs_valid : std_logic;
  signal buffer7_outs_ready : std_logic;
  signal buffer7_outs_spec : std_logic_vector(0 downto 0);
  signal mux3_outs : std_logic_vector(7 downto 0);
  signal mux3_outs_valid : std_logic;
  signal mux3_outs_ready : std_logic;
  signal mux3_outs_spec : std_logic_vector(0 downto 0);
  signal buffer13_outs : std_logic_vector(7 downto 0);
  signal buffer13_outs_valid : std_logic;
  signal buffer13_outs_ready : std_logic;
  signal buffer13_outs_spec : std_logic_vector(0 downto 0);
  signal extsi5_outs : std_logic_vector(8 downto 0);
  signal extsi5_outs_valid : std_logic;
  signal extsi5_outs_ready : std_logic;
  signal extsi5_outs_spec : std_logic_vector(0 downto 0);
  signal buffer8_outs_valid : std_logic;
  signal buffer8_outs_ready : std_logic;
  signal buffer8_outs_spec : std_logic_vector(0 downto 0);
  signal mux4_outs_valid : std_logic;
  signal mux4_outs_ready : std_logic;
  signal mux4_outs_spec : std_logic_vector(0 downto 0);
  signal source4_outs_valid : std_logic;
  signal source4_outs_ready : std_logic;
  signal source4_outs_spec : std_logic_vector(0 downto 0);
  signal constant11_outs : std_logic_vector(31 downto 0);
  signal constant11_outs_valid : std_logic;
  signal constant11_outs_ready : std_logic;
  signal constant11_outs_spec : std_logic_vector(0 downto 0);
  signal source5_outs_valid : std_logic;
  signal source5_outs_ready : std_logic;
  signal source5_outs_spec : std_logic_vector(0 downto 0);
  signal constant4_outs : std_logic_vector(1 downto 0);
  signal constant4_outs_valid : std_logic;
  signal constant4_outs_ready : std_logic;
  signal constant4_outs_spec : std_logic_vector(0 downto 0);
  signal extsi6_outs : std_logic_vector(8 downto 0);
  signal extsi6_outs_valid : std_logic;
  signal extsi6_outs_ready : std_logic;
  signal extsi6_outs_spec : std_logic_vector(0 downto 0);
  signal source6_outs_valid : std_logic;
  signal source6_outs_ready : std_logic;
  signal source6_outs_spec : std_logic_vector(0 downto 0);
  signal constant5_outs : std_logic_vector(7 downto 0);
  signal constant5_outs_valid : std_logic;
  signal constant5_outs_ready : std_logic;
  signal constant5_outs_spec : std_logic_vector(0 downto 0);
  signal extsi7_outs : std_logic_vector(8 downto 0);
  signal extsi7_outs_valid : std_logic;
  signal extsi7_outs_ready : std_logic;
  signal extsi7_outs_spec : std_logic_vector(0 downto 0);
  signal buffer12_outs : std_logic_vector(31 downto 0);
  signal buffer12_outs_valid : std_logic;
  signal buffer12_outs_ready : std_logic;
  signal buffer12_outs_spec : std_logic_vector(0 downto 0);
  signal divf0_result : std_logic_vector(31 downto 0);
  signal divf0_result_valid : std_logic;
  signal divf0_result_ready : std_logic;
  signal divf0_result_spec : std_logic_vector(0 downto 0);
  signal buffer14_outs : std_logic_vector(8 downto 0);
  signal buffer14_outs_valid : std_logic;
  signal buffer14_outs_ready : std_logic;
  signal buffer14_outs_spec : std_logic_vector(0 downto 0);
  signal addi0_result : std_logic_vector(8 downto 0);
  signal addi0_result_valid : std_logic;
  signal addi0_result_ready : std_logic;
  signal addi0_result_spec : std_logic_vector(0 downto 0);
  signal fork9_outs_0 : std_logic_vector(8 downto 0);
  signal fork9_outs_0_valid : std_logic;
  signal fork9_outs_0_ready : std_logic;
  signal fork9_outs_0_spec : std_logic_vector(0 downto 0);
  signal fork9_outs_1 : std_logic_vector(8 downto 0);
  signal fork9_outs_1_valid : std_logic;
  signal fork9_outs_1_ready : std_logic;
  signal fork9_outs_1_spec : std_logic_vector(0 downto 0);
  signal trunci2_outs : std_logic_vector(7 downto 0);
  signal trunci2_outs_valid : std_logic;
  signal trunci2_outs_ready : std_logic;
  signal trunci2_outs_spec : std_logic_vector(0 downto 0);
  signal cmpi0_result : std_logic_vector(0 downto 0);
  signal cmpi0_result_valid : std_logic;
  signal cmpi0_result_ready : std_logic;
  signal cmpi0_result_spec : std_logic_vector(0 downto 0);
  signal fork18_outs_0 : std_logic_vector(0 downto 0);
  signal fork18_outs_0_valid : std_logic;
  signal fork18_outs_0_ready : std_logic;
  signal fork18_outs_0_spec : std_logic_vector(0 downto 0);
  signal fork18_outs_1 : std_logic_vector(0 downto 0);
  signal fork18_outs_1_valid : std_logic;
  signal fork18_outs_1_ready : std_logic;
  signal fork18_outs_1_spec : std_logic_vector(0 downto 0);
  signal fork18_outs_2 : std_logic_vector(0 downto 0);
  signal fork18_outs_2_valid : std_logic;
  signal fork18_outs_2_ready : std_logic;
  signal fork18_outs_2_spec : std_logic_vector(0 downto 0);
  signal fork18_outs_3 : std_logic_vector(0 downto 0);
  signal fork18_outs_3_valid : std_logic;
  signal fork18_outs_3_ready : std_logic;
  signal fork18_outs_3_spec : std_logic_vector(0 downto 0);
  signal fork18_outs_4 : std_logic_vector(0 downto 0);
  signal fork18_outs_4_valid : std_logic;
  signal fork18_outs_4_ready : std_logic;
  signal fork18_outs_4_spec : std_logic_vector(0 downto 0);
  signal cond_br5_trueOut : std_logic_vector(7 downto 0);
  signal cond_br5_trueOut_valid : std_logic;
  signal cond_br5_trueOut_ready : std_logic;
  signal cond_br5_trueOut_spec : std_logic_vector(0 downto 0);
  signal cond_br5_falseOut : std_logic_vector(7 downto 0);
  signal cond_br5_falseOut_valid : std_logic;
  signal cond_br5_falseOut_ready : std_logic;
  signal cond_br5_falseOut_spec : std_logic_vector(0 downto 0);
  signal cond_br6_trueOut : std_logic_vector(31 downto 0);
  signal cond_br6_trueOut_valid : std_logic;
  signal cond_br6_trueOut_ready : std_logic;
  signal cond_br6_trueOut_spec : std_logic_vector(0 downto 0);
  signal cond_br6_falseOut : std_logic_vector(31 downto 0);
  signal cond_br6_falseOut_valid : std_logic;
  signal cond_br6_falseOut_ready : std_logic;
  signal cond_br6_falseOut_spec : std_logic_vector(0 downto 0);
  signal buffer15_outs_valid : std_logic;
  signal buffer15_outs_ready : std_logic;
  signal buffer15_outs_spec : std_logic_vector(0 downto 0);
  signal buffer16_outs_valid : std_logic;
  signal buffer16_outs_ready : std_logic;
  signal buffer16_outs_spec : std_logic_vector(0 downto 0);
  signal cond_br7_trueOut_valid : std_logic;
  signal cond_br7_trueOut_ready : std_logic;
  signal cond_br7_trueOut_spec : std_logic_vector(0 downto 0);
  signal cond_br7_falseOut_valid : std_logic;
  signal cond_br7_falseOut_ready : std_logic;
  signal cond_br7_falseOut_spec : std_logic_vector(0 downto 0);
  signal fork11_outs_0_valid : std_logic;
  signal fork11_outs_0_ready : std_logic;
  signal fork11_outs_0_spec : std_logic_vector(0 downto 0);
  signal fork11_outs_1_valid : std_logic;
  signal fork11_outs_1_ready : std_logic;
  signal fork11_outs_1_spec : std_logic_vector(0 downto 0);
  signal buffer30_outs : std_logic_vector(0 downto 0);
  signal buffer30_outs_valid : std_logic;
  signal buffer30_outs_ready : std_logic;
  signal buffer31_outs : std_logic_vector(31 downto 0);
  signal buffer31_outs_valid : std_logic;
  signal buffer31_outs_ready : std_logic;
  signal buffer31_outs_spec : std_logic_vector(0 downto 0);
  signal spec_commit5_outs : std_logic_vector(31 downto 0);
  signal spec_commit5_outs_valid : std_logic;
  signal spec_commit5_outs_ready : std_logic;

begin

  out0 <= spec_commit5_outs;
  out0_valid <= spec_commit5_outs_valid;
  spec_commit5_outs_ready <= out0_ready;
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

  non_spec0 : entity work.handshake_non_spec_0(arch)
    port map(
      dataIn => x0,
      dataIn_valid => x0_valid,
      dataIn_ready => x0_ready,
      clk => clk,
      rst => rst,
      dataOut => non_spec0_dataOut,
      dataOut_valid => non_spec0_dataOut_valid,
      dataOut_ready => non_spec0_dataOut_ready,
      dataOut_spec => non_spec0_dataOut_spec
    );

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

  non_spec1 : entity work.handshake_non_spec_1(arch)
    port map(
      dataIn_valid => fork0_outs_2_valid,
      dataIn_ready => fork0_outs_2_ready,
      clk => clk,
      rst => rst,
      dataOut_valid => non_spec1_dataOut_valid,
      dataOut_ready => non_spec1_dataOut_ready,
      dataOut_spec => non_spec1_dataOut_spec
    );

  buffer17 : entity work.handshake_buffer_0(arch)
    port map(
      ins => fork14_outs_0,
      ins_valid => fork14_outs_0_valid,
      ins_ready => fork14_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => buffer17_outs,
      outs_valid => buffer17_outs_valid,
      outs_ready => buffer17_outs_ready
    );

  buffer18 : entity work.handshake_buffer_1(arch)
    port map(
      ins_valid => fork11_outs_1_valid,
      ins_ready => fork11_outs_1_ready,
      ins_spec => fork11_outs_1_spec,
      clk => clk,
      rst => rst,
      outs_valid => buffer18_outs_valid,
      outs_ready => buffer18_outs_ready,
      outs_spec => buffer18_outs_spec
    );

  spec_commit0 : entity work.handshake_spec_commit_0(arch)
    port map(
      ins_valid => buffer18_outs_valid,
      ins_ready => buffer18_outs_ready,
      ins_spec => buffer18_outs_spec,
      ctrl => buffer17_outs,
      ctrl_valid => buffer17_outs_valid,
      ctrl_ready => buffer17_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => spec_commit0_outs_valid,
      outs_ready => spec_commit0_outs_ready
    );

  buffer19 : entity work.handshake_buffer_0(arch)
    port map(
      ins => fork15_outs_1,
      ins_valid => fork15_outs_1_valid,
      ins_ready => fork15_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => buffer19_outs,
      outs_valid => buffer19_outs_valid,
      outs_ready => buffer19_outs_ready
    );

  buffer20 : entity work.handshake_buffer_2(arch)
    port map(
      ins => extsi1_outs,
      ins_valid => extsi1_outs_valid,
      ins_ready => extsi1_outs_ready,
      ins_spec => extsi1_outs_spec,
      clk => clk,
      rst => rst,
      outs => buffer20_outs,
      outs_valid => buffer20_outs_valid,
      outs_ready => buffer20_outs_ready,
      outs_spec => buffer20_outs_spec
    );

  spec_commit1 : entity work.handshake_spec_commit_1(arch)
    port map(
      ins => buffer20_outs,
      ins_valid => buffer20_outs_valid,
      ins_ready => buffer20_outs_ready,
      ins_spec => buffer20_outs_spec,
      ctrl => buffer19_outs,
      ctrl_valid => buffer19_outs_valid,
      ctrl_ready => buffer19_outs_ready,
      clk => clk,
      rst => rst,
      outs => spec_commit1_outs,
      outs_valid => spec_commit1_outs_valid,
      outs_ready => spec_commit1_outs_ready
    );

  mem_controller2 : entity work.handshake_mem_controller_0(arch)
    port map(
      loadData => minus_trace_loadData,
      memStart_valid => minus_trace_start_valid,
      memStart_ready => minus_trace_start_ready,
      ctrl(0) => spec_commit1_outs,
      ctrl_valid(0) => spec_commit1_outs_valid,
      ctrl_ready(0) => spec_commit1_outs_ready,
      stAddr(0) => store0_addrOut,
      stAddr_valid(0) => store0_addrOut_valid,
      stAddr_ready(0) => store0_addrOut_ready,
      stData(0) => store0_dataToMem,
      stData_valid(0) => store0_dataToMem_valid,
      stData_ready(0) => store0_dataToMem_ready,
      ctrlEnd_valid => spec_commit0_outs_valid,
      ctrlEnd_ready => spec_commit0_outs_ready,
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

  buffer21 : entity work.handshake_buffer_0(arch)
    port map(
      ins => fork14_outs_1,
      ins_valid => fork14_outs_1_valid,
      ins_ready => fork14_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => buffer21_outs,
      outs_valid => buffer21_outs_valid,
      outs_ready => buffer21_outs_ready
    );

  buffer22 : entity work.handshake_buffer_1(arch)
    port map(
      ins_valid => fork11_outs_0_valid,
      ins_ready => fork11_outs_0_ready,
      ins_spec => fork11_outs_0_spec,
      clk => clk,
      rst => rst,
      outs_valid => buffer22_outs_valid,
      outs_ready => buffer22_outs_ready,
      outs_spec => buffer22_outs_spec
    );

  spec_commit2 : entity work.handshake_spec_commit_0(arch)
    port map(
      ins_valid => buffer22_outs_valid,
      ins_ready => buffer22_outs_ready,
      ins_spec => buffer22_outs_spec,
      ctrl => buffer21_outs,
      ctrl_valid => buffer21_outs_valid,
      ctrl_ready => buffer21_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => spec_commit2_outs_valid,
      outs_ready => spec_commit2_outs_ready
    );

  mem_controller3 : entity work.handshake_mem_controller_1(arch)
    port map(
      loadData => a_loadData,
      memStart_valid => a_start_valid,
      memStart_ready => a_start_ready,
      ldAddr(0) => load0_addrOut,
      ldAddr_valid(0) => load0_addrOut_valid,
      ldAddr_ready(0) => load0_addrOut_ready,
      ctrlEnd_valid => spec_commit2_outs_valid,
      ctrlEnd_ready => spec_commit2_outs_ready,
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

  non_spec2 : entity work.handshake_non_spec_2(arch)
    port map(
      dataIn => extsi4_outs,
      dataIn_valid => extsi4_outs_valid,
      dataIn_ready => extsi4_outs_ready,
      clk => clk,
      rst => rst,
      dataOut => non_spec2_dataOut,
      dataOut_valid => non_spec2_dataOut_valid,
      dataOut_ready => non_spec2_dataOut_ready,
      dataOut_spec => non_spec2_dataOut_spec
    );

  mux0 : entity work.handshake_mux_0(arch)
    port map(
      index => fork3_outs_0,
      index_valid => fork3_outs_0_valid,
      index_ready => fork3_outs_0_ready,
      index_spec => fork3_outs_0_spec,
      ins(0) => non_spec2_dataOut,
      ins(1) => cond_br5_trueOut,
      ins_valid(0) => non_spec2_dataOut_valid,
      ins_valid(1) => cond_br5_trueOut_valid,
      ins_ready(0) => non_spec2_dataOut_ready,
      ins_ready(1) => cond_br5_trueOut_ready,
      ins_0_spec => non_spec2_dataOut_spec,
      ins_1_spec => cond_br5_trueOut_spec,
      clk => clk,
      rst => rst,
      outs => mux0_outs,
      outs_valid => mux0_outs_valid,
      outs_ready => mux0_outs_ready,
      outs_spec => mux0_outs_spec
    );

  buffer0 : entity work.handshake_buffer_3(arch)
    port map(
      ins => mux0_outs,
      ins_valid => mux0_outs_valid,
      ins_ready => mux0_outs_ready,
      ins_spec => mux0_outs_spec,
      clk => clk,
      rst => rst,
      outs => buffer0_outs,
      outs_valid => buffer0_outs_valid,
      outs_ready => buffer0_outs_ready,
      outs_spec => buffer0_outs_spec
    );

  fork1 : entity work.handshake_fork_1(arch)
    port map(
      ins => buffer0_outs,
      ins_valid => buffer0_outs_valid,
      ins_ready => buffer0_outs_ready,
      ins_spec => buffer0_outs_spec,
      clk => clk,
      rst => rst,
      outs(0) => fork1_outs_0,
      outs(1) => fork1_outs_1,
      outs_valid(0) => fork1_outs_0_valid,
      outs_valid(1) => fork1_outs_1_valid,
      outs_ready(0) => fork1_outs_0_ready,
      outs_ready(1) => fork1_outs_1_ready,
      outs_0_spec => fork1_outs_0_spec,
      outs_1_spec => fork1_outs_1_spec
    );

  trunci0 : entity work.handshake_trunci_0(arch)
    port map(
      ins => fork1_outs_0,
      ins_valid => fork1_outs_0_valid,
      ins_ready => fork1_outs_0_ready,
      ins_spec => fork1_outs_0_spec,
      clk => clk,
      rst => rst,
      outs => trunci0_outs,
      outs_valid => trunci0_outs_valid,
      outs_ready => trunci0_outs_ready,
      outs_spec => trunci0_outs_spec
    );

  mux1 : entity work.handshake_mux_1(arch)
    port map(
      index => fork3_outs_1,
      index_valid => fork3_outs_1_valid,
      index_ready => fork3_outs_1_ready,
      index_spec => fork3_outs_1_spec,
      ins(0) => non_spec0_dataOut,
      ins(1) => cond_br6_trueOut,
      ins_valid(0) => non_spec0_dataOut_valid,
      ins_valid(1) => cond_br6_trueOut_valid,
      ins_ready(0) => non_spec0_dataOut_ready,
      ins_ready(1) => cond_br6_trueOut_ready,
      ins_0_spec => non_spec0_dataOut_spec,
      ins_1_spec => cond_br6_trueOut_spec,
      clk => clk,
      rst => rst,
      outs => mux1_outs,
      outs_valid => mux1_outs_valid,
      outs_ready => mux1_outs_ready,
      outs_spec => mux1_outs_spec
    );

  buffer2 : entity work.handshake_buffer_4(arch)
    port map(
      ins => mux1_outs,
      ins_valid => mux1_outs_valid,
      ins_ready => mux1_outs_ready,
      ins_spec => mux1_outs_spec,
      clk => clk,
      rst => rst,
      outs => buffer2_outs,
      outs_valid => buffer2_outs_valid,
      outs_ready => buffer2_outs_ready,
      outs_spec => buffer2_outs_spec
    );

  fork2 : entity work.handshake_fork_2(arch)
    port map(
      ins => buffer2_outs,
      ins_valid => buffer2_outs_valid,
      ins_ready => buffer2_outs_ready,
      ins_spec => buffer2_outs_spec,
      clk => clk,
      rst => rst,
      outs(0) => fork2_outs_0,
      outs(1) => fork2_outs_1,
      outs(2) => fork2_outs_2,
      outs_valid(0) => fork2_outs_0_valid,
      outs_valid(1) => fork2_outs_1_valid,
      outs_valid(2) => fork2_outs_2_valid,
      outs_ready(0) => fork2_outs_0_ready,
      outs_ready(1) => fork2_outs_1_ready,
      outs_ready(2) => fork2_outs_2_ready,
      outs_0_spec => fork2_outs_0_spec,
      outs_1_spec => fork2_outs_1_spec,
      outs_2_spec => fork2_outs_2_spec
    );

  control_merge0 : entity work.handshake_control_merge_0(arch)
    port map(
      ins_valid(0) => non_spec1_dataOut_valid,
      ins_valid(1) => cond_br7_trueOut_valid,
      ins_ready(0) => non_spec1_dataOut_ready,
      ins_ready(1) => cond_br7_trueOut_ready,
      ins_0_spec => non_spec1_dataOut_spec,
      ins_1_spec => cond_br7_trueOut_spec,
      clk => clk,
      rst => rst,
      outs_valid => control_merge0_outs_valid,
      outs_ready => control_merge0_outs_ready,
      outs_spec => control_merge0_outs_spec,
      index => control_merge0_index,
      index_valid => control_merge0_index_valid,
      index_ready => control_merge0_index_ready,
      index_spec => control_merge0_index_spec
    );

  fork4 : entity work.handshake_fork_3(arch)
    port map(
      ins_valid => control_merge0_outs_valid,
      ins_ready => control_merge0_outs_ready,
      ins_spec => control_merge0_outs_spec,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork4_outs_0_valid,
      outs_valid(1) => fork4_outs_1_valid,
      outs_ready(0) => fork4_outs_0_ready,
      outs_ready(1) => fork4_outs_1_ready,
      outs_0_spec => fork4_outs_0_spec,
      outs_1_spec => fork4_outs_1_spec
    );

  fork3 : entity work.handshake_fork_4(arch)
    port map(
      ins => control_merge0_index,
      ins_valid => control_merge0_index_valid,
      ins_ready => control_merge0_index_ready,
      ins_spec => control_merge0_index_spec,
      clk => clk,
      rst => rst,
      outs(0) => fork3_outs_0,
      outs(1) => fork3_outs_1,
      outs_valid(0) => fork3_outs_0_valid,
      outs_valid(1) => fork3_outs_1_valid,
      outs_ready(0) => fork3_outs_0_ready,
      outs_ready(1) => fork3_outs_1_ready,
      outs_0_spec => fork3_outs_0_spec,
      outs_1_spec => fork3_outs_1_spec
    );

  source0 : entity work.handshake_source_0(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source0_outs_valid,
      outs_ready => source0_outs_ready,
      outs_spec => source0_outs_spec
    );

  constant7 : entity work.handshake_constant_1(arch)
    port map(
      ctrl_valid => source0_outs_valid,
      ctrl_ready => source0_outs_ready,
      ctrl_spec => source0_outs_spec,
      clk => clk,
      rst => rst,
      outs => constant7_outs,
      outs_valid => constant7_outs_valid,
      outs_ready => constant7_outs_ready,
      outs_spec => constant7_outs_spec
    );

  source1 : entity work.handshake_source_0(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source1_outs_valid,
      outs_ready => source1_outs_ready,
      outs_spec => source1_outs_spec
    );

  constant8 : entity work.handshake_constant_2(arch)
    port map(
      ctrl_valid => source1_outs_valid,
      ctrl_ready => source1_outs_ready,
      ctrl_spec => source1_outs_spec,
      clk => clk,
      rst => rst,
      outs => constant8_outs,
      outs_valid => constant8_outs_valid,
      outs_ready => constant8_outs_ready,
      outs_spec => constant8_outs_spec
    );

  buffer1 : entity work.handshake_buffer_5(arch)
    port map(
      ins => trunci0_outs,
      ins_valid => trunci0_outs_valid,
      ins_ready => trunci0_outs_ready,
      ins_spec => trunci0_outs_spec,
      clk => clk,
      rst => rst,
      outs => buffer1_outs,
      outs_valid => buffer1_outs_valid,
      outs_ready => buffer1_outs_ready,
      outs_spec => buffer1_outs_spec
    );

  load0 : entity work.handshake_load_0(arch)
    port map(
      addrIn => buffer1_outs,
      addrIn_valid => buffer1_outs_valid,
      addrIn_ready => buffer1_outs_ready,
      addrIn_spec => buffer1_outs_spec,
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
      dataOut_ready => load0_dataOut_ready,
      dataOut_spec => load0_dataOut_spec
    );

  buffer3 : entity work.handshake_buffer_6(arch)
    port map(
      ins => fork2_outs_2,
      ins_valid => fork2_outs_2_valid,
      ins_ready => fork2_outs_2_ready,
      ins_spec => fork2_outs_2_spec,
      clk => clk,
      rst => rst,
      outs => buffer3_outs,
      outs_valid => buffer3_outs_valid,
      outs_ready => buffer3_outs_ready,
      outs_spec => buffer3_outs_spec
    );

  mulf0 : entity work.handshake_mulf_0(arch)
    port map(
      lhs => load0_dataOut,
      lhs_valid => load0_dataOut_valid,
      lhs_ready => load0_dataOut_ready,
      lhs_spec => load0_dataOut_spec,
      rhs => buffer3_outs,
      rhs_valid => buffer3_outs_valid,
      rhs_ready => buffer3_outs_ready,
      rhs_spec => buffer3_outs_spec,
      clk => clk,
      rst => rst,
      result => mulf0_result,
      result_valid => mulf0_result_valid,
      result_ready => mulf0_result_ready,
      result_spec => mulf0_result_spec
    );

  mulf1 : entity work.handshake_mulf_1(arch)
    port map(
      lhs => fork2_outs_1,
      lhs_valid => fork2_outs_1_valid,
      lhs_ready => fork2_outs_1_ready,
      lhs_spec => fork2_outs_1_spec,
      rhs => constant7_outs,
      rhs_valid => constant7_outs_valid,
      rhs_ready => constant7_outs_ready,
      rhs_spec => constant7_outs_spec,
      clk => clk,
      rst => rst,
      result => mulf1_result,
      result_valid => mulf1_result_valid,
      result_ready => mulf1_result_ready,
      result_spec => mulf1_result_spec
    );

  buffer4 : entity work.handshake_buffer_7(arch)
    port map(
      ins => mulf1_result,
      ins_valid => mulf1_result_valid,
      ins_ready => mulf1_result_ready,
      ins_spec => mulf1_result_spec,
      clk => clk,
      rst => rst,
      outs => buffer4_outs,
      outs_valid => buffer4_outs_valid,
      outs_ready => buffer4_outs_ready,
      outs_spec => buffer4_outs_spec
    );

  addf0 : entity work.handshake_addf_0(arch)
    port map(
      lhs => mulf0_result,
      lhs_valid => mulf0_result_valid,
      lhs_ready => mulf0_result_ready,
      lhs_spec => mulf0_result_spec,
      rhs => buffer4_outs,
      rhs_valid => buffer4_outs_valid,
      rhs_ready => buffer4_outs_ready,
      rhs_spec => buffer4_outs_spec,
      clk => clk,
      rst => rst,
      result => addf0_result,
      result_valid => addf0_result_valid,
      result_ready => addf0_result_ready,
      result_spec => addf0_result_spec
    );

  cmpf0 : entity work.handshake_cmpf_0(arch)
    port map(
      lhs => addf0_result,
      lhs_valid => addf0_result_valid,
      lhs_ready => addf0_result_ready,
      lhs_spec => addf0_result_spec,
      rhs => constant8_outs,
      rhs_valid => constant8_outs_valid,
      rhs_ready => constant8_outs_ready,
      rhs_spec => constant8_outs_spec,
      clk => clk,
      rst => rst,
      result => cmpf0_result,
      result_valid => cmpf0_result_valid,
      result_ready => cmpf0_result_ready,
      result_spec => cmpf0_result_spec
    );

  speculator0 : entity work.handshake_speculator_0(arch)
    port map(
      ins => cmpf0_result,
      ins_valid => cmpf0_result_valid,
      ins_ready => cmpf0_result_ready,
      ins_spec => cmpf0_result_spec,
      trigger_valid => fork4_outs_1_valid,
      trigger_ready => fork4_outs_1_ready,
      trigger_spec => fork4_outs_1_spec,
      clk => clk,
      rst => rst,
      outs => speculator0_outs,
      outs_valid => speculator0_outs_valid,
      outs_ready => speculator0_outs_ready,
      outs_spec => speculator0_outs_spec,
      ctrl_save => speculator0_ctrl_save,
      ctrl_save_valid => speculator0_ctrl_save_valid,
      ctrl_save_ready => speculator0_ctrl_save_ready,
      ctrl_commit => speculator0_ctrl_commit,
      ctrl_commit_valid => speculator0_ctrl_commit_valid,
      ctrl_commit_ready => speculator0_ctrl_commit_ready,
      ctrl_sc_save => speculator0_ctrl_sc_save,
      ctrl_sc_save_valid => speculator0_ctrl_sc_save_valid,
      ctrl_sc_save_ready => speculator0_ctrl_sc_save_ready,
      ctrl_sc_commit => speculator0_ctrl_sc_commit,
      ctrl_sc_commit_valid => speculator0_ctrl_sc_commit_valid,
      ctrl_sc_commit_ready => speculator0_ctrl_sc_commit_ready,
      ctrl_sc_branch => speculator0_ctrl_sc_branch,
      ctrl_sc_branch_valid => speculator0_ctrl_sc_branch_valid,
      ctrl_sc_branch_ready => speculator0_ctrl_sc_branch_ready
    );

  cond_br9 : entity work.handshake_cond_br_0(arch)
    port map(
      condition => cond_br8_trueOut,
      condition_valid => cond_br8_trueOut_valid,
      condition_ready => cond_br8_trueOut_ready,
      data => speculator0_ctrl_sc_commit,
      data_valid => speculator0_ctrl_sc_commit_valid,
      data_ready => speculator0_ctrl_sc_commit_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br9_trueOut,
      trueOut_valid => cond_br9_trueOut_valid,
      trueOut_ready => cond_br9_trueOut_ready,
      falseOut => cond_br9_falseOut,
      falseOut_valid => cond_br9_falseOut_valid,
      falseOut_ready => cond_br9_falseOut_ready
    );

  sink8 : entity work.handshake_sink_0(arch)
    port map(
      ins => cond_br9_falseOut,
      ins_valid => cond_br9_falseOut_valid,
      ins_ready => cond_br9_falseOut_ready,
      clk => clk,
      rst => rst
    );

  merge0 : entity work.handshake_merge_0(arch)
    port map(
      ins(0) => speculator0_ctrl_sc_save,
      ins(1) => cond_br9_trueOut,
      ins_valid(0) => speculator0_ctrl_sc_save_valid,
      ins_valid(1) => cond_br9_trueOut_valid,
      ins_ready(0) => speculator0_ctrl_sc_save_ready,
      ins_ready(1) => cond_br9_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => merge0_outs,
      outs_valid => merge0_outs_valid,
      outs_ready => merge0_outs_ready
    );

  fork8 : entity work.handshake_fork_5(arch)
    port map(
      ins => merge0_outs,
      ins_valid => merge0_outs_valid,
      ins_ready => merge0_outs_ready,
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

  fork12 : entity work.handshake_fork_6(arch)
    port map(
      ins => speculator0_ctrl_commit,
      ins_valid => speculator0_ctrl_commit_valid,
      ins_ready => speculator0_ctrl_commit_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork12_outs_0,
      outs(1) => fork12_outs_1,
      outs_valid(0) => fork12_outs_0_valid,
      outs_valid(1) => fork12_outs_1_valid,
      outs_ready(0) => fork12_outs_0_ready,
      outs_ready(1) => fork12_outs_1_ready
    );

  sink0 : entity work.handshake_sink_1(arch)
    port map(
      ins => speculator0_ctrl_save,
      ins_valid => speculator0_ctrl_save_valid,
      ins_ready => speculator0_ctrl_save_ready,
      clk => clk,
      rst => rst
    );

  buffer23 : entity work.handshake_buffer_0(arch)
    port map(
      ins => fork16_outs_0,
      ins_valid => fork16_outs_0_valid,
      ins_ready => fork16_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => buffer23_outs,
      outs_valid => buffer23_outs_valid,
      outs_ready => buffer23_outs_ready
    );

  cond_br0 : entity work.handshake_cond_br_1(arch)
    port map(
      condition => buffer23_outs,
      condition_valid => buffer23_outs_valid,
      condition_ready => buffer23_outs_ready,
      data => fork12_outs_0,
      data_valid => fork12_outs_0_valid,
      data_ready => fork12_outs_0_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br0_trueOut,
      trueOut_valid => cond_br0_trueOut_valid,
      trueOut_ready => cond_br0_trueOut_ready,
      falseOut => cond_br0_falseOut,
      falseOut_valid => cond_br0_falseOut_valid,
      falseOut_ready => cond_br0_falseOut_ready
    );

  fork14 : entity work.handshake_fork_7(arch)
    port map(
      ins => cond_br0_falseOut,
      ins_valid => cond_br0_falseOut_valid,
      ins_ready => cond_br0_falseOut_ready,
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

  sink1 : entity work.handshake_sink_1(arch)
    port map(
      ins => cond_br0_trueOut,
      ins_valid => cond_br0_trueOut_valid,
      ins_ready => cond_br0_trueOut_ready,
      clk => clk,
      rst => rst
    );

  speculating_branch0 : entity work.handshake_speculating_branch_0(arch)
    port map(
      spec_tag_data => fork17_outs_5,
      spec_tag_data_valid => fork17_outs_5_valid,
      spec_tag_data_ready => fork17_outs_5_ready,
      spec_tag_data_spec => fork17_outs_5_spec,
      data => fork17_outs_6,
      data_valid => fork17_outs_6_valid,
      data_ready => fork17_outs_6_ready,
      data_spec => fork17_outs_6_spec,
      clk => clk,
      rst => rst,
      trueOut => speculating_branch0_trueOut,
      trueOut_valid => speculating_branch0_trueOut_valid,
      trueOut_ready => speculating_branch0_trueOut_ready,
      falseOut => speculating_branch0_falseOut,
      falseOut_valid => speculating_branch0_falseOut_valid,
      falseOut_ready => speculating_branch0_falseOut_ready
    );

  sink3 : entity work.handshake_sink_1(arch)
    port map(
      ins => speculating_branch0_falseOut,
      ins_valid => speculating_branch0_falseOut_valid,
      ins_ready => speculating_branch0_falseOut_ready,
      clk => clk,
      rst => rst
    );

  buffer5 : entity work.handshake_buffer_8(arch)
    port map(
      ins => speculating_branch0_trueOut,
      ins_valid => speculating_branch0_trueOut_valid,
      ins_ready => speculating_branch0_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => buffer5_outs,
      outs_valid => buffer5_outs_valid,
      outs_ready => buffer5_outs_ready
    );

  buffer24 : entity work.handshake_buffer_0(arch)
    port map(
      ins => buffer5_outs,
      ins_valid => buffer5_outs_valid,
      ins_ready => buffer5_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer24_outs,
      outs_valid => buffer24_outs_valid,
      outs_ready => buffer24_outs_ready
    );

  cond_br1 : entity work.handshake_cond_br_1(arch)
    port map(
      condition => buffer24_outs,
      condition_valid => buffer24_outs_valid,
      condition_ready => buffer24_outs_ready,
      data => fork12_outs_1,
      data_valid => fork12_outs_1_valid,
      data_ready => fork12_outs_1_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br1_trueOut,
      trueOut_valid => cond_br1_trueOut_valid,
      trueOut_ready => cond_br1_trueOut_ready,
      falseOut => cond_br1_falseOut,
      falseOut_valid => cond_br1_falseOut_valid,
      falseOut_ready => cond_br1_falseOut_ready
    );

  sink4 : entity work.handshake_sink_1(arch)
    port map(
      ins => cond_br1_falseOut,
      ins_valid => cond_br1_falseOut_valid,
      ins_ready => cond_br1_falseOut_ready,
      clk => clk,
      rst => rst
    );

  fork15 : entity work.handshake_fork_7(arch)
    port map(
      ins => cond_br1_trueOut,
      ins_valid => cond_br1_trueOut_valid,
      ins_ready => cond_br1_trueOut_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork15_outs_0,
      outs(1) => fork15_outs_1,
      outs(2) => fork15_outs_2,
      outs_valid(0) => fork15_outs_0_valid,
      outs_valid(1) => fork15_outs_1_valid,
      outs_valid(2) => fork15_outs_2_valid,
      outs_ready(0) => fork15_outs_0_ready,
      outs_ready(1) => fork15_outs_1_ready,
      outs_ready(2) => fork15_outs_2_ready
    );

  speculating_branch1 : entity work.handshake_speculating_branch_0(arch)
    port map(
      spec_tag_data => fork18_outs_2,
      spec_tag_data_valid => fork18_outs_2_valid,
      spec_tag_data_ready => fork18_outs_2_ready,
      spec_tag_data_spec => fork18_outs_2_spec,
      data => fork18_outs_3,
      data_valid => fork18_outs_3_valid,
      data_ready => fork18_outs_3_ready,
      data_spec => fork18_outs_3_spec,
      clk => clk,
      rst => rst,
      trueOut => speculating_branch1_trueOut,
      trueOut_valid => speculating_branch1_trueOut_valid,
      trueOut_ready => speculating_branch1_trueOut_ready,
      falseOut => speculating_branch1_falseOut,
      falseOut_valid => speculating_branch1_falseOut_valid,
      falseOut_ready => speculating_branch1_falseOut_ready
    );

  sink5 : entity work.handshake_sink_1(arch)
    port map(
      ins => speculating_branch1_falseOut,
      ins_valid => speculating_branch1_falseOut_valid,
      ins_ready => speculating_branch1_falseOut_ready,
      clk => clk,
      rst => rst
    );

  fork16 : entity work.handshake_fork_6(arch)
    port map(
      ins => speculating_branch1_trueOut,
      ins_valid => speculating_branch1_trueOut_valid,
      ins_ready => speculating_branch1_trueOut_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork16_outs_0,
      outs(1) => fork16_outs_1,
      outs_valid(0) => fork16_outs_0_valid,
      outs_valid(1) => fork16_outs_1_valid,
      outs_ready(0) => fork16_outs_0_ready,
      outs_ready(1) => fork16_outs_1_ready
    );

  buffer25 : entity work.handshake_buffer_0(arch)
    port map(
      ins => fork16_outs_1,
      ins_valid => fork16_outs_1_valid,
      ins_ready => fork16_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => buffer25_outs,
      outs_valid => buffer25_outs_valid,
      outs_ready => buffer25_outs_ready
    );

  cond_br8 : entity work.handshake_cond_br_1(arch)
    port map(
      condition => speculator0_ctrl_sc_branch,
      condition_valid => speculator0_ctrl_sc_branch_valid,
      condition_ready => speculator0_ctrl_sc_branch_ready,
      data => buffer25_outs,
      data_valid => buffer25_outs_valid,
      data_ready => buffer25_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br8_trueOut,
      trueOut_valid => cond_br8_trueOut_valid,
      trueOut_ready => cond_br8_trueOut_ready,
      falseOut => cond_br8_falseOut,
      falseOut_valid => cond_br8_falseOut_valid,
      falseOut_ready => cond_br8_falseOut_ready
    );

  sink6 : entity work.handshake_sink_1(arch)
    port map(
      ins => cond_br8_falseOut,
      ins_valid => cond_br8_falseOut_valid,
      ins_ready => cond_br8_falseOut_ready,
      clk => clk,
      rst => rst
    );

  fork17 : entity work.handshake_fork_8(arch)
    port map(
      ins => speculator0_outs,
      ins_valid => speculator0_outs_valid,
      ins_ready => speculator0_outs_ready,
      ins_spec => speculator0_outs_spec,
      clk => clk,
      rst => rst,
      outs(0) => fork17_outs_0,
      outs(1) => fork17_outs_1,
      outs(2) => fork17_outs_2,
      outs(3) => fork17_outs_3,
      outs(4) => fork17_outs_4,
      outs(5) => fork17_outs_5,
      outs(6) => fork17_outs_6,
      outs(7) => fork17_outs_7,
      outs_valid(0) => fork17_outs_0_valid,
      outs_valid(1) => fork17_outs_1_valid,
      outs_valid(2) => fork17_outs_2_valid,
      outs_valid(3) => fork17_outs_3_valid,
      outs_valid(4) => fork17_outs_4_valid,
      outs_valid(5) => fork17_outs_5_valid,
      outs_valid(6) => fork17_outs_6_valid,
      outs_valid(7) => fork17_outs_7_valid,
      outs_ready(0) => fork17_outs_0_ready,
      outs_ready(1) => fork17_outs_1_ready,
      outs_ready(2) => fork17_outs_2_ready,
      outs_ready(3) => fork17_outs_3_ready,
      outs_ready(4) => fork17_outs_4_ready,
      outs_ready(5) => fork17_outs_5_ready,
      outs_ready(6) => fork17_outs_6_ready,
      outs_ready(7) => fork17_outs_7_ready,
      outs_0_spec => fork17_outs_0_spec,
      outs_1_spec => fork17_outs_1_spec,
      outs_2_spec => fork17_outs_2_spec,
      outs_3_spec => fork17_outs_3_spec,
      outs_4_spec => fork17_outs_4_spec,
      outs_5_spec => fork17_outs_5_spec,
      outs_6_spec => fork17_outs_6_spec,
      outs_7_spec => fork17_outs_7_spec
    );

  spec_save_commit0 : entity work.handshake_spec_save_commit_0(arch)
    port map(
      ins => fork1_outs_1,
      ins_valid => fork1_outs_1_valid,
      ins_ready => fork1_outs_1_ready,
      ins_spec => fork1_outs_1_spec,
      ctrl => fork8_outs_1,
      ctrl_valid => fork8_outs_1_valid,
      ctrl_ready => fork8_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => spec_save_commit0_outs,
      outs_valid => spec_save_commit0_outs_valid,
      outs_ready => spec_save_commit0_outs_ready,
      outs_spec => spec_save_commit0_outs_spec
    );

  cond_br2 : entity work.handshake_cond_br_2(arch)
    port map(
      condition => fork17_outs_7,
      condition_valid => fork17_outs_7_valid,
      condition_ready => fork17_outs_7_ready,
      condition_spec => fork17_outs_7_spec,
      data => spec_save_commit0_outs,
      data_valid => spec_save_commit0_outs_valid,
      data_ready => spec_save_commit0_outs_ready,
      data_spec => spec_save_commit0_outs_spec,
      clk => clk,
      rst => rst,
      trueOut => cond_br2_trueOut,
      trueOut_valid => cond_br2_trueOut_valid,
      trueOut_ready => cond_br2_trueOut_ready,
      trueOut_spec => cond_br2_trueOut_spec,
      falseOut => cond_br2_falseOut,
      falseOut_valid => cond_br2_falseOut_valid,
      falseOut_ready => cond_br2_falseOut_ready,
      falseOut_spec => cond_br2_falseOut_spec
    );

  spec_save_commit1 : entity work.handshake_spec_save_commit_1(arch)
    port map(
      ins => fork2_outs_0,
      ins_valid => fork2_outs_0_valid,
      ins_ready => fork2_outs_0_ready,
      ins_spec => fork2_outs_0_spec,
      ctrl => fork8_outs_0,
      ctrl_valid => fork8_outs_0_valid,
      ctrl_ready => fork8_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => spec_save_commit1_outs,
      outs_valid => spec_save_commit1_outs_valid,
      outs_ready => spec_save_commit1_outs_ready,
      outs_spec => spec_save_commit1_outs_spec
    );

  cond_br3 : entity work.handshake_cond_br_3(arch)
    port map(
      condition => fork17_outs_4,
      condition_valid => fork17_outs_4_valid,
      condition_ready => fork17_outs_4_ready,
      condition_spec => fork17_outs_4_spec,
      data => spec_save_commit1_outs,
      data_valid => spec_save_commit1_outs_valid,
      data_ready => spec_save_commit1_outs_ready,
      data_spec => spec_save_commit1_outs_spec,
      clk => clk,
      rst => rst,
      trueOut => cond_br3_trueOut,
      trueOut_valid => cond_br3_trueOut_valid,
      trueOut_ready => cond_br3_trueOut_ready,
      trueOut_spec => cond_br3_trueOut_spec,
      falseOut => cond_br3_falseOut,
      falseOut_valid => cond_br3_falseOut_valid,
      falseOut_ready => cond_br3_falseOut_ready,
      falseOut_spec => cond_br3_falseOut_spec
    );

  spec_save_commit2 : entity work.handshake_spec_save_commit_2(arch)
    port map(
      ins_valid => fork4_outs_0_valid,
      ins_ready => fork4_outs_0_ready,
      ins_spec => fork4_outs_0_spec,
      ctrl => fork8_outs_2,
      ctrl_valid => fork8_outs_2_valid,
      ctrl_ready => fork8_outs_2_ready,
      clk => clk,
      rst => rst,
      outs_valid => spec_save_commit2_outs_valid,
      outs_ready => spec_save_commit2_outs_ready,
      outs_spec => spec_save_commit2_outs_spec
    );

  cond_br4 : entity work.handshake_cond_br_4(arch)
    port map(
      condition => fork17_outs_3,
      condition_valid => fork17_outs_3_valid,
      condition_ready => fork17_outs_3_ready,
      condition_spec => fork17_outs_3_spec,
      data_valid => spec_save_commit2_outs_valid,
      data_ready => spec_save_commit2_outs_ready,
      data_spec => spec_save_commit2_outs_spec,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br4_trueOut_valid,
      trueOut_ready => cond_br4_trueOut_ready,
      trueOut_spec => cond_br4_trueOut_spec,
      falseOut_valid => cond_br4_falseOut_valid,
      falseOut_ready => cond_br4_falseOut_ready,
      falseOut_spec => cond_br4_falseOut_spec
    );

  source2 : entity work.handshake_source_0(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source2_outs_valid,
      outs_ready => source2_outs_ready,
      outs_spec => source2_outs_spec
    );

  constant9 : entity work.handshake_constant_3(arch)
    port map(
      ctrl_valid => source2_outs_valid,
      ctrl_ready => source2_outs_ready,
      ctrl_spec => source2_outs_spec,
      clk => clk,
      rst => rst,
      outs => constant9_outs,
      outs_valid => constant9_outs_valid,
      outs_ready => constant9_outs_ready,
      outs_spec => constant9_outs_spec
    );

  addf1 : entity work.handshake_addf_1(arch)
    port map(
      lhs => cond_br3_falseOut,
      lhs_valid => cond_br3_falseOut_valid,
      lhs_ready => cond_br3_falseOut_ready,
      lhs_spec => cond_br3_falseOut_spec,
      rhs => constant9_outs,
      rhs_valid => constant9_outs_valid,
      rhs_ready => constant9_outs_ready,
      rhs_spec => constant9_outs_spec,
      clk => clk,
      rst => rst,
      result => addf1_result,
      result_valid => addf1_result_valid,
      result_ready => addf1_result_ready,
      result_spec => addf1_result_spec
    );

  fork5 : entity work.handshake_fork_1(arch)
    port map(
      ins => cond_br2_trueOut,
      ins_valid => cond_br2_trueOut_valid,
      ins_ready => cond_br2_trueOut_ready,
      ins_spec => cond_br2_trueOut_spec,
      clk => clk,
      rst => rst,
      outs(0) => fork5_outs_0,
      outs(1) => fork5_outs_1,
      outs_valid(0) => fork5_outs_0_valid,
      outs_valid(1) => fork5_outs_1_valid,
      outs_ready(0) => fork5_outs_0_ready,
      outs_ready(1) => fork5_outs_1_ready,
      outs_0_spec => fork5_outs_0_spec,
      outs_1_spec => fork5_outs_1_spec
    );

  buffer9 : entity work.handshake_buffer_9(arch)
    port map(
      ins => fork5_outs_0,
      ins_valid => fork5_outs_0_valid,
      ins_ready => fork5_outs_0_ready,
      ins_spec => fork5_outs_0_spec,
      clk => clk,
      rst => rst,
      outs => buffer9_outs,
      outs_valid => buffer9_outs_valid,
      outs_ready => buffer9_outs_ready,
      outs_spec => buffer9_outs_spec
    );

  trunci1 : entity work.handshake_trunci_0(arch)
    port map(
      ins => buffer9_outs,
      ins_valid => buffer9_outs_valid,
      ins_ready => buffer9_outs_ready,
      ins_spec => buffer9_outs_spec,
      clk => clk,
      rst => rst,
      outs => trunci1_outs,
      outs_valid => trunci1_outs_valid,
      outs_ready => trunci1_outs_ready,
      outs_spec => trunci1_outs_spec
    );

  fork6 : entity work.handshake_fork_9(arch)
    port map(
      ins => cond_br3_trueOut,
      ins_valid => cond_br3_trueOut_valid,
      ins_ready => cond_br3_trueOut_ready,
      ins_spec => cond_br3_trueOut_spec,
      clk => clk,
      rst => rst,
      outs(0) => fork6_outs_0,
      outs(1) => fork6_outs_1,
      outs_valid(0) => fork6_outs_0_valid,
      outs_valid(1) => fork6_outs_1_valid,
      outs_ready(0) => fork6_outs_0_ready,
      outs_ready(1) => fork6_outs_1_ready,
      outs_0_spec => fork6_outs_0_spec,
      outs_1_spec => fork6_outs_1_spec
    );

  fork7 : entity work.handshake_fork_3(arch)
    port map(
      ins_valid => cond_br4_trueOut_valid,
      ins_ready => cond_br4_trueOut_ready,
      ins_spec => cond_br4_trueOut_spec,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork7_outs_0_valid,
      outs_valid(1) => fork7_outs_1_valid,
      outs_ready(0) => fork7_outs_0_ready,
      outs_ready(1) => fork7_outs_1_ready,
      outs_0_spec => fork7_outs_0_spec,
      outs_1_spec => fork7_outs_1_spec
    );

  buffer11 : entity work.handshake_buffer_1(arch)
    port map(
      ins_valid => fork7_outs_0_valid,
      ins_ready => fork7_outs_0_ready,
      ins_spec => fork7_outs_0_spec,
      clk => clk,
      rst => rst,
      outs_valid => buffer11_outs_valid,
      outs_ready => buffer11_outs_ready,
      outs_spec => buffer11_outs_spec
    );

  constant3 : entity work.handshake_constant_4(arch)
    port map(
      ctrl_valid => buffer11_outs_valid,
      ctrl_ready => buffer11_outs_ready,
      ctrl_spec => buffer11_outs_spec,
      clk => clk,
      rst => rst,
      outs => constant3_outs,
      outs_valid => constant3_outs_valid,
      outs_ready => constant3_outs_ready,
      outs_spec => constant3_outs_spec
    );

  extsi1 : entity work.handshake_extsi_1(arch)
    port map(
      ins => constant3_outs,
      ins_valid => constant3_outs_valid,
      ins_ready => constant3_outs_ready,
      ins_spec => constant3_outs_spec,
      clk => clk,
      rst => rst,
      outs => extsi1_outs,
      outs_valid => extsi1_outs_valid,
      outs_ready => extsi1_outs_ready,
      outs_spec => extsi1_outs_spec
    );

  source3 : entity work.handshake_source_0(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source3_outs_valid,
      outs_ready => source3_outs_ready,
      outs_spec => source3_outs_spec
    );

  constant10 : entity work.handshake_constant_5(arch)
    port map(
      ctrl_valid => source3_outs_valid,
      ctrl_ready => source3_outs_ready,
      ctrl_spec => source3_outs_spec,
      clk => clk,
      rst => rst,
      outs => constant10_outs,
      outs_valid => constant10_outs_valid,
      outs_ready => constant10_outs_ready,
      outs_spec => constant10_outs_spec
    );

  buffer26 : entity work.handshake_buffer_0(arch)
    port map(
      ins => fork15_outs_2,
      ins_valid => fork15_outs_2_valid,
      ins_ready => fork15_outs_2_ready,
      clk => clk,
      rst => rst,
      outs => buffer26_outs,
      outs_valid => buffer26_outs_valid,
      outs_ready => buffer26_outs_ready
    );

  buffer27 : entity work.handshake_buffer_5(arch)
    port map(
      ins => trunci1_outs,
      ins_valid => trunci1_outs_valid,
      ins_ready => trunci1_outs_ready,
      ins_spec => trunci1_outs_spec,
      clk => clk,
      rst => rst,
      outs => buffer27_outs,
      outs_valid => buffer27_outs_valid,
      outs_ready => buffer27_outs_ready,
      outs_spec => buffer27_outs_spec
    );

  spec_commit3 : entity work.handshake_spec_commit_2(arch)
    port map(
      ins => buffer27_outs,
      ins_valid => buffer27_outs_valid,
      ins_ready => buffer27_outs_ready,
      ins_spec => buffer27_outs_spec,
      ctrl => buffer26_outs,
      ctrl_valid => buffer26_outs_valid,
      ctrl_ready => buffer26_outs_ready,
      clk => clk,
      rst => rst,
      outs => spec_commit3_outs,
      outs_valid => spec_commit3_outs_valid,
      outs_ready => spec_commit3_outs_ready
    );

  buffer10 : entity work.handshake_buffer_10(arch)
    port map(
      ins => fork6_outs_0,
      ins_valid => fork6_outs_0_valid,
      ins_ready => fork6_outs_0_ready,
      ins_spec => fork6_outs_0_spec,
      clk => clk,
      rst => rst,
      outs => buffer10_outs,
      outs_valid => buffer10_outs_valid,
      outs_ready => buffer10_outs_ready,
      outs_spec => buffer10_outs_spec
    );

  buffer28 : entity work.handshake_buffer_0(arch)
    port map(
      ins => fork15_outs_0,
      ins_valid => fork15_outs_0_valid,
      ins_ready => fork15_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => buffer28_outs,
      outs_valid => buffer28_outs_valid,
      outs_ready => buffer28_outs_ready
    );

  buffer29 : entity work.handshake_buffer_11(arch)
    port map(
      ins => buffer10_outs,
      ins_valid => buffer10_outs_valid,
      ins_ready => buffer10_outs_ready,
      ins_spec => buffer10_outs_spec,
      clk => clk,
      rst => rst,
      outs => buffer29_outs,
      outs_valid => buffer29_outs_valid,
      outs_ready => buffer29_outs_ready,
      outs_spec => buffer29_outs_spec
    );

  spec_commit4 : entity work.handshake_spec_commit_3(arch)
    port map(
      ins => buffer29_outs,
      ins_valid => buffer29_outs_valid,
      ins_ready => buffer29_outs_ready,
      ins_spec => buffer29_outs_spec,
      ctrl => buffer28_outs,
      ctrl_valid => buffer28_outs_valid,
      ctrl_ready => buffer28_outs_ready,
      clk => clk,
      rst => rst,
      outs => spec_commit4_outs,
      outs_valid => spec_commit4_outs_valid,
      outs_ready => spec_commit4_outs_ready
    );

  store0 : entity work.handshake_store_0(arch)
    port map(
      addrIn => spec_commit3_outs,
      addrIn_valid => spec_commit3_outs_valid,
      addrIn_ready => spec_commit3_outs_ready,
      dataIn => spec_commit4_outs,
      dataIn_valid => spec_commit4_outs_valid,
      dataIn_ready => spec_commit4_outs_ready,
      clk => clk,
      rst => rst,
      addrOut => store0_addrOut,
      addrOut_valid => store0_addrOut_valid,
      addrOut_ready => store0_addrOut_ready,
      dataToMem => store0_dataToMem,
      dataToMem_valid => store0_dataToMem_valid,
      dataToMem_ready => store0_dataToMem_ready
    );

  addf2 : entity work.handshake_addf_2(arch)
    port map(
      lhs => fork6_outs_1,
      lhs_valid => fork6_outs_1_valid,
      lhs_ready => fork6_outs_1_ready,
      lhs_spec => fork6_outs_1_spec,
      rhs => constant10_outs,
      rhs_valid => constant10_outs_valid,
      rhs_ready => constant10_outs_ready,
      rhs_spec => constant10_outs_spec,
      clk => clk,
      rst => rst,
      result => addf2_result,
      result_valid => addf2_result_valid,
      result_ready => addf2_result_ready,
      result_spec => addf2_result_spec
    );

  buffer6 : entity work.handshake_buffer_12(arch)
    port map(
      ins => fork17_outs_1,
      ins_valid => fork17_outs_1_valid,
      ins_ready => fork17_outs_1_ready,
      ins_spec => fork17_outs_1_spec,
      clk => clk,
      rst => rst,
      outs => buffer6_outs,
      outs_valid => buffer6_outs_valid,
      outs_ready => buffer6_outs_ready,
      outs_spec => buffer6_outs_spec
    );

  mux2 : entity work.handshake_mux_2(arch)
    port map(
      index => buffer6_outs,
      index_valid => buffer6_outs_valid,
      index_ready => buffer6_outs_ready,
      index_spec => buffer6_outs_spec,
      ins(0) => addf1_result,
      ins(1) => addf2_result,
      ins_valid(0) => addf1_result_valid,
      ins_valid(1) => addf2_result_valid,
      ins_ready(0) => addf1_result_ready,
      ins_ready(1) => addf2_result_ready,
      ins_0_spec => addf1_result_spec,
      ins_1_spec => addf2_result_spec,
      clk => clk,
      rst => rst,
      outs => mux2_outs,
      outs_valid => mux2_outs_valid,
      outs_ready => mux2_outs_ready,
      outs_spec => mux2_outs_spec
    );

  buffer7 : entity work.handshake_buffer_13(arch)
    port map(
      ins => cond_br2_falseOut,
      ins_valid => cond_br2_falseOut_valid,
      ins_ready => cond_br2_falseOut_ready,
      ins_spec => cond_br2_falseOut_spec,
      clk => clk,
      rst => rst,
      outs => buffer7_outs,
      outs_valid => buffer7_outs_valid,
      outs_ready => buffer7_outs_ready,
      outs_spec => buffer7_outs_spec
    );

  mux3 : entity work.handshake_mux_0(arch)
    port map(
      index => fork17_outs_0,
      index_valid => fork17_outs_0_valid,
      index_ready => fork17_outs_0_ready,
      index_spec => fork17_outs_0_spec,
      ins(0) => buffer7_outs,
      ins(1) => fork5_outs_1,
      ins_valid(0) => buffer7_outs_valid,
      ins_valid(1) => fork5_outs_1_valid,
      ins_ready(0) => buffer7_outs_ready,
      ins_ready(1) => fork5_outs_1_ready,
      ins_0_spec => buffer7_outs_spec,
      ins_1_spec => fork5_outs_1_spec,
      clk => clk,
      rst => rst,
      outs => mux3_outs,
      outs_valid => mux3_outs_valid,
      outs_ready => mux3_outs_ready,
      outs_spec => mux3_outs_spec
    );

  buffer13 : entity work.handshake_buffer_3(arch)
    port map(
      ins => mux3_outs,
      ins_valid => mux3_outs_valid,
      ins_ready => mux3_outs_ready,
      ins_spec => mux3_outs_spec,
      clk => clk,
      rst => rst,
      outs => buffer13_outs,
      outs_valid => buffer13_outs_valid,
      outs_ready => buffer13_outs_ready,
      outs_spec => buffer13_outs_spec
    );

  extsi5 : entity work.handshake_extsi_2(arch)
    port map(
      ins => buffer13_outs,
      ins_valid => buffer13_outs_valid,
      ins_ready => buffer13_outs_ready,
      ins_spec => buffer13_outs_spec,
      clk => clk,
      rst => rst,
      outs => extsi5_outs,
      outs_valid => extsi5_outs_valid,
      outs_ready => extsi5_outs_ready,
      outs_spec => extsi5_outs_spec
    );

  buffer8 : entity work.handshake_buffer_14(arch)
    port map(
      ins_valid => cond_br4_falseOut_valid,
      ins_ready => cond_br4_falseOut_ready,
      ins_spec => cond_br4_falseOut_spec,
      clk => clk,
      rst => rst,
      outs_valid => buffer8_outs_valid,
      outs_ready => buffer8_outs_ready,
      outs_spec => buffer8_outs_spec
    );

  mux4 : entity work.handshake_mux_3(arch)
    port map(
      index => fork17_outs_2,
      index_valid => fork17_outs_2_valid,
      index_ready => fork17_outs_2_ready,
      index_spec => fork17_outs_2_spec,
      ins_valid(0) => buffer8_outs_valid,
      ins_valid(1) => fork7_outs_1_valid,
      ins_ready(0) => buffer8_outs_ready,
      ins_ready(1) => fork7_outs_1_ready,
      ins_0_spec => buffer8_outs_spec,
      ins_1_spec => fork7_outs_1_spec,
      clk => clk,
      rst => rst,
      outs_valid => mux4_outs_valid,
      outs_ready => mux4_outs_ready,
      outs_spec => mux4_outs_spec
    );

  source4 : entity work.handshake_source_0(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source4_outs_valid,
      outs_ready => source4_outs_ready,
      outs_spec => source4_outs_spec
    );

  constant11 : entity work.handshake_constant_6(arch)
    port map(
      ctrl_valid => source4_outs_valid,
      ctrl_ready => source4_outs_ready,
      ctrl_spec => source4_outs_spec,
      clk => clk,
      rst => rst,
      outs => constant11_outs,
      outs_valid => constant11_outs_valid,
      outs_ready => constant11_outs_ready,
      outs_spec => constant11_outs_spec
    );

  source5 : entity work.handshake_source_0(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source5_outs_valid,
      outs_ready => source5_outs_ready,
      outs_spec => source5_outs_spec
    );

  constant4 : entity work.handshake_constant_4(arch)
    port map(
      ctrl_valid => source5_outs_valid,
      ctrl_ready => source5_outs_ready,
      ctrl_spec => source5_outs_spec,
      clk => clk,
      rst => rst,
      outs => constant4_outs,
      outs_valid => constant4_outs_valid,
      outs_ready => constant4_outs_ready,
      outs_spec => constant4_outs_spec
    );

  extsi6 : entity work.handshake_extsi_3(arch)
    port map(
      ins => constant4_outs,
      ins_valid => constant4_outs_valid,
      ins_ready => constant4_outs_ready,
      ins_spec => constant4_outs_spec,
      clk => clk,
      rst => rst,
      outs => extsi6_outs,
      outs_valid => extsi6_outs_valid,
      outs_ready => extsi6_outs_ready,
      outs_spec => extsi6_outs_spec
    );

  source6 : entity work.handshake_source_0(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source6_outs_valid,
      outs_ready => source6_outs_ready,
      outs_spec => source6_outs_spec
    );

  constant5 : entity work.handshake_constant_7(arch)
    port map(
      ctrl_valid => source6_outs_valid,
      ctrl_ready => source6_outs_ready,
      ctrl_spec => source6_outs_spec,
      clk => clk,
      rst => rst,
      outs => constant5_outs,
      outs_valid => constant5_outs_valid,
      outs_ready => constant5_outs_ready,
      outs_spec => constant5_outs_spec
    );

  extsi7 : entity work.handshake_extsi_2(arch)
    port map(
      ins => constant5_outs,
      ins_valid => constant5_outs_valid,
      ins_ready => constant5_outs_ready,
      ins_spec => constant5_outs_spec,
      clk => clk,
      rst => rst,
      outs => extsi7_outs,
      outs_valid => extsi7_outs_valid,
      outs_ready => extsi7_outs_ready,
      outs_spec => extsi7_outs_spec
    );

  buffer12 : entity work.handshake_buffer_15(arch)
    port map(
      ins => mux2_outs,
      ins_valid => mux2_outs_valid,
      ins_ready => mux2_outs_ready,
      ins_spec => mux2_outs_spec,
      clk => clk,
      rst => rst,
      outs => buffer12_outs,
      outs_valid => buffer12_outs_valid,
      outs_ready => buffer12_outs_ready,
      outs_spec => buffer12_outs_spec
    );

  divf0 : entity work.handshake_divf_0(arch)
    port map(
      lhs => constant11_outs,
      lhs_valid => constant11_outs_valid,
      lhs_ready => constant11_outs_ready,
      lhs_spec => constant11_outs_spec,
      rhs => buffer12_outs,
      rhs_valid => buffer12_outs_valid,
      rhs_ready => buffer12_outs_ready,
      rhs_spec => buffer12_outs_spec,
      clk => clk,
      rst => rst,
      result => divf0_result,
      result_valid => divf0_result_valid,
      result_ready => divf0_result_ready,
      result_spec => divf0_result_spec
    );

  buffer14 : entity work.handshake_buffer_16(arch)
    port map(
      ins => extsi5_outs,
      ins_valid => extsi5_outs_valid,
      ins_ready => extsi5_outs_ready,
      ins_spec => extsi5_outs_spec,
      clk => clk,
      rst => rst,
      outs => buffer14_outs,
      outs_valid => buffer14_outs_valid,
      outs_ready => buffer14_outs_ready,
      outs_spec => buffer14_outs_spec
    );

  addi0 : entity work.handshake_addi_0(arch)
    port map(
      lhs => buffer14_outs,
      lhs_valid => buffer14_outs_valid,
      lhs_ready => buffer14_outs_ready,
      lhs_spec => buffer14_outs_spec,
      rhs => extsi6_outs,
      rhs_valid => extsi6_outs_valid,
      rhs_ready => extsi6_outs_ready,
      rhs_spec => extsi6_outs_spec,
      clk => clk,
      rst => rst,
      result => addi0_result,
      result_valid => addi0_result_valid,
      result_ready => addi0_result_ready,
      result_spec => addi0_result_spec
    );

  fork9 : entity work.handshake_fork_10(arch)
    port map(
      ins => addi0_result,
      ins_valid => addi0_result_valid,
      ins_ready => addi0_result_ready,
      ins_spec => addi0_result_spec,
      clk => clk,
      rst => rst,
      outs(0) => fork9_outs_0,
      outs(1) => fork9_outs_1,
      outs_valid(0) => fork9_outs_0_valid,
      outs_valid(1) => fork9_outs_1_valid,
      outs_ready(0) => fork9_outs_0_ready,
      outs_ready(1) => fork9_outs_1_ready,
      outs_0_spec => fork9_outs_0_spec,
      outs_1_spec => fork9_outs_1_spec
    );

  trunci2 : entity work.handshake_trunci_1(arch)
    port map(
      ins => fork9_outs_0,
      ins_valid => fork9_outs_0_valid,
      ins_ready => fork9_outs_0_ready,
      ins_spec => fork9_outs_0_spec,
      clk => clk,
      rst => rst,
      outs => trunci2_outs,
      outs_valid => trunci2_outs_valid,
      outs_ready => trunci2_outs_ready,
      outs_spec => trunci2_outs_spec
    );

  cmpi0 : entity work.handshake_cmpi_0(arch)
    port map(
      lhs => fork9_outs_1,
      lhs_valid => fork9_outs_1_valid,
      lhs_ready => fork9_outs_1_ready,
      lhs_spec => fork9_outs_1_spec,
      rhs => extsi7_outs,
      rhs_valid => extsi7_outs_valid,
      rhs_ready => extsi7_outs_ready,
      rhs_spec => extsi7_outs_spec,
      clk => clk,
      rst => rst,
      result => cmpi0_result,
      result_valid => cmpi0_result_valid,
      result_ready => cmpi0_result_ready,
      result_spec => cmpi0_result_spec
    );

  fork18 : entity work.handshake_fork_11(arch)
    port map(
      ins => cmpi0_result,
      ins_valid => cmpi0_result_valid,
      ins_ready => cmpi0_result_ready,
      ins_spec => cmpi0_result_spec,
      clk => clk,
      rst => rst,
      outs(0) => fork18_outs_0,
      outs(1) => fork18_outs_1,
      outs(2) => fork18_outs_2,
      outs(3) => fork18_outs_3,
      outs(4) => fork18_outs_4,
      outs_valid(0) => fork18_outs_0_valid,
      outs_valid(1) => fork18_outs_1_valid,
      outs_valid(2) => fork18_outs_2_valid,
      outs_valid(3) => fork18_outs_3_valid,
      outs_valid(4) => fork18_outs_4_valid,
      outs_ready(0) => fork18_outs_0_ready,
      outs_ready(1) => fork18_outs_1_ready,
      outs_ready(2) => fork18_outs_2_ready,
      outs_ready(3) => fork18_outs_3_ready,
      outs_ready(4) => fork18_outs_4_ready,
      outs_0_spec => fork18_outs_0_spec,
      outs_1_spec => fork18_outs_1_spec,
      outs_2_spec => fork18_outs_2_spec,
      outs_3_spec => fork18_outs_3_spec,
      outs_4_spec => fork18_outs_4_spec
    );

  cond_br5 : entity work.handshake_cond_br_2(arch)
    port map(
      condition => fork18_outs_4,
      condition_valid => fork18_outs_4_valid,
      condition_ready => fork18_outs_4_ready,
      condition_spec => fork18_outs_4_spec,
      data => trunci2_outs,
      data_valid => trunci2_outs_valid,
      data_ready => trunci2_outs_ready,
      data_spec => trunci2_outs_spec,
      clk => clk,
      rst => rst,
      trueOut => cond_br5_trueOut,
      trueOut_valid => cond_br5_trueOut_valid,
      trueOut_ready => cond_br5_trueOut_ready,
      trueOut_spec => cond_br5_trueOut_spec,
      falseOut => cond_br5_falseOut,
      falseOut_valid => cond_br5_falseOut_valid,
      falseOut_ready => cond_br5_falseOut_ready,
      falseOut_spec => cond_br5_falseOut_spec
    );

  sink2 : entity work.handshake_sink_2(arch)
    port map(
      ins => cond_br5_falseOut,
      ins_valid => cond_br5_falseOut_valid,
      ins_ready => cond_br5_falseOut_ready,
      ins_spec => cond_br5_falseOut_spec,
      clk => clk,
      rst => rst
    );

  cond_br6 : entity work.handshake_cond_br_5(arch)
    port map(
      condition => fork18_outs_0,
      condition_valid => fork18_outs_0_valid,
      condition_ready => fork18_outs_0_ready,
      condition_spec => fork18_outs_0_spec,
      data => divf0_result,
      data_valid => divf0_result_valid,
      data_ready => divf0_result_ready,
      data_spec => divf0_result_spec,
      clk => clk,
      rst => rst,
      trueOut => cond_br6_trueOut,
      trueOut_valid => cond_br6_trueOut_valid,
      trueOut_ready => cond_br6_trueOut_ready,
      trueOut_spec => cond_br6_trueOut_spec,
      falseOut => cond_br6_falseOut,
      falseOut_valid => cond_br6_falseOut_valid,
      falseOut_ready => cond_br6_falseOut_ready,
      falseOut_spec => cond_br6_falseOut_spec
    );

  buffer15 : entity work.handshake_buffer_14(arch)
    port map(
      ins_valid => mux4_outs_valid,
      ins_ready => mux4_outs_ready,
      ins_spec => mux4_outs_spec,
      clk => clk,
      rst => rst,
      outs_valid => buffer15_outs_valid,
      outs_ready => buffer15_outs_ready,
      outs_spec => buffer15_outs_spec
    );

  buffer16 : entity work.handshake_buffer_17(arch)
    port map(
      ins_valid => buffer15_outs_valid,
      ins_ready => buffer15_outs_ready,
      ins_spec => buffer15_outs_spec,
      clk => clk,
      rst => rst,
      outs_valid => buffer16_outs_valid,
      outs_ready => buffer16_outs_ready,
      outs_spec => buffer16_outs_spec
    );

  cond_br7 : entity work.handshake_cond_br_4(arch)
    port map(
      condition => fork18_outs_1,
      condition_valid => fork18_outs_1_valid,
      condition_ready => fork18_outs_1_ready,
      condition_spec => fork18_outs_1_spec,
      data_valid => buffer16_outs_valid,
      data_ready => buffer16_outs_ready,
      data_spec => buffer16_outs_spec,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br7_trueOut_valid,
      trueOut_ready => cond_br7_trueOut_ready,
      trueOut_spec => cond_br7_trueOut_spec,
      falseOut_valid => cond_br7_falseOut_valid,
      falseOut_ready => cond_br7_falseOut_ready,
      falseOut_spec => cond_br7_falseOut_spec
    );

  fork11 : entity work.handshake_fork_3(arch)
    port map(
      ins_valid => cond_br7_falseOut_valid,
      ins_ready => cond_br7_falseOut_ready,
      ins_spec => cond_br7_falseOut_spec,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork11_outs_0_valid,
      outs_valid(1) => fork11_outs_1_valid,
      outs_ready(0) => fork11_outs_0_ready,
      outs_ready(1) => fork11_outs_1_ready,
      outs_0_spec => fork11_outs_0_spec,
      outs_1_spec => fork11_outs_1_spec
    );

  buffer30 : entity work.handshake_buffer_0(arch)
    port map(
      ins => fork14_outs_2,
      ins_valid => fork14_outs_2_valid,
      ins_ready => fork14_outs_2_ready,
      clk => clk,
      rst => rst,
      outs => buffer30_outs,
      outs_valid => buffer30_outs_valid,
      outs_ready => buffer30_outs_ready
    );

  buffer31 : entity work.handshake_buffer_18(arch)
    port map(
      ins => cond_br6_falseOut,
      ins_valid => cond_br6_falseOut_valid,
      ins_ready => cond_br6_falseOut_ready,
      ins_spec => cond_br6_falseOut_spec,
      clk => clk,
      rst => rst,
      outs => buffer31_outs,
      outs_valid => buffer31_outs_valid,
      outs_ready => buffer31_outs_ready,
      outs_spec => buffer31_outs_spec
    );

  spec_commit5 : entity work.handshake_spec_commit_4(arch)
    port map(
      ins => buffer31_outs,
      ins_valid => buffer31_outs_valid,
      ins_ready => buffer31_outs_ready,
      ins_spec => buffer31_outs_spec,
      ctrl => buffer30_outs,
      ctrl_valid => buffer30_outs_valid,
      ctrl_ready => buffer30_outs_ready,
      clk => clk,
      rst => rst,
      outs => spec_commit5_outs,
      outs_valid => spec_commit5_outs_valid,
      outs_ready => spec_commit5_outs_ready
    );

end architecture;
