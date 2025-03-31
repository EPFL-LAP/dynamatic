import cocotb
from cocotb.clock import Clock
from cocotb.triggers import Timer, RisingEdge
import cocotb.runner
import os


def main():
  base_path = os.path.dirname(os.path.abspath(__file__))

  types_file = os.path.join(base_path, "types.vhd")

  runner = cocotb.runner.get_runner("questa")
  runner.build(
      vhdl_sources=[
          "test_suite/types.vhd",
          "test_suite/speculator.vhd",
          "test_suite/spec_save_commit.vhd",
          "test_suite/merge.vhd",
          "test_suite/circuit.vhd"
      ],
      hdl_toplevel="circuit",
      build_args=["-2008"]
  )

  runner.test(
      hdl_toplevel="circuit",
      test_module="test_suite.testbench",
      testcase=["test_nocmp"],
      waves=True
  )


@cocotb.test()
async def test_nocmp(dut):
  await cocotb.start(Clock(dut.clk, 4, "ns").start())
  dut.rst.value = 1
  await Timer(8, "ns")
  dut.rst.value = 0

  cocotb.start
  # Non-spec condition
  dut.speculator_ins.value = 0
  dut.speculator_ins_valid.value = 1
  dut.speculator_ins_spec.value = 0

  for i in range(10):
    print(dut.speculator_ins_ready, dut.speculator_ins_valid, dut.speculator_ins)
    if dut.speculator_ins_ready.value == 1:
      break
    await RisingEdge(dut.clk)

  pass


if __name__ == "__main__":
  main()
