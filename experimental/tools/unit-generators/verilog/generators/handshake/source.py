
def generate_source(name, params):
    extra_signals = params.get("extra_signals", None)

    return _generate_source(name)


def _generate_source(name):
    return f"""
// Module of source
module {name} (
  input  clk,        
  input  rst,        
  input  outs_ready, 
  output outs_valid 
);
  assign outs_valid = 1;

endmodule

"""