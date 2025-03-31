import cocotb
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

  runner.test(hdl_toplevel="circuit", test_module="test_suite.testbench",
              testcase=["test_nocmp"])


@cocotb.test()
def test_nocmp(dut):
  print(dut)
  pass


if __name__ == "__main__":
  main()
