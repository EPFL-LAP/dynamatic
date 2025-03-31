import cocotb
from cocotb.clock import Clock
from cocotb.triggers import Timer, RisingEdge
import cocotb.runner


def main():
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


async def watcher_nocmp(dut):
  print("Watching for Values")
  sc_ctrls = []
  spec_outs = []
  sc_outs = []
  for _ in range(100):
    if dut.sc.ctrl_valid.value == 1 and dut.sc.ctrl_ready.value == 1:
      sc_ctrls.append(dut.sc.ctrl.value)
    if dut.speculator_outs_valid.value == 1 and dut.speculator_outs_ready.value == 1:
      spec_outs.append(dut.speculator_outs.value)
    if dut.sc_outs_valid.value == 1 and dut.sc_outs_ready.value == 1:
      sc_outs.append(dut.sc_outs.value)
    await RisingEdge(dut.clk)
  print(sc_ctrls, spec_outs, sc_outs)
  assert len(spec_outs) == 1
  assert len(sc_outs) == 1


@cocotb.test()
async def test_nocmp(dut):
  await cocotb.start(Clock(dut.clk, 4, "ns").start())
  watcher = cocotb.start_soon(watcher_nocmp(dut))
  dut.rst.value = 1
  await Timer(8, "ns")
  dut.rst.value = 0

  # Initial values
  dut.speculator_ins.value = 0
  dut.speculator_ins_valid.value = 0
  dut.speculator_ins_spec.value = 0
  dut.speculator_trigger_valid.value = 0
  dut.speculator_trigger_spec.value = 0
  dut.speculator_outs_ready.value = 1
  dut.speculator_ctrl_save_ready.value = 1
  dut.speculator_ctrl_commit_ready.value = 1
  dut.speculator_ctrl_sc_branch_ready.value = 1

  # Always send a token
  dut.sc_ins.value = 0
  dut.sc_ins_valid.value = 1
  dut.sc_ins_spec.value = 0

  dut.sc_outs_ready.value = 1
  await RisingEdge(dut.clk)

  # Non-spec condition
  dut.speculator_ins.value = 0
  dut.speculator_ins_valid.value = 1
  dut.speculator_ins_spec.value = 0
  await RisingEdge(dut.clk)

  for _ in range(10):
    if dut.speculator_ins_ready.value == 1:
      dut.speculator_ins_valid.value = 0
    await RisingEdge(dut.clk)

  # Trigger
  dut.speculator_trigger_valid.value = 1
  dut.speculator_trigger_spec.value = 0
  await RisingEdge(dut.clk)

  for _ in range(10):
    if dut.speculator_ins_ready.value == 1:
      dut.speculator_ins_valid.value = 0

    if dut.speculator_trigger_ready.value == 1:
      dut.speculator_trigger_valid.value = 0

    await RisingEdge(dut.clk)

  # Assert watcher lifetime is long enough
  assert not watcher.done()
  await watcher.join()

if __name__ == "__main__":
  main()
