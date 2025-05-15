build/bin/vivado-ip-adapter \
  "integration-test/iir/out/comp/handshake_export.mlir" \
  > /tmp/wrapper.v


/data/verilator/bin/verilator \
  -Wall --lint-only /tmp/wrapper.v