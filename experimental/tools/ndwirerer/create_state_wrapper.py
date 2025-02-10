import argparse
import re
import os


parser = argparse.ArgumentParser(
                    prog='StateWrapper',
                    description='TODO What the program does',
                    epilog='TODO Text at the bottom of help')

parser.add_argument("--mlir")
group = parser.add_mutually_exclusive_group()
group.add_argument("-N", type=int)
group.add_argument("--inf", action="store_true")


args = parser.parse_args()

with open(args.mlir) as f:
  mlir = f.read()

# print(mlir)

arg_names_match = re.search(r'argNames\s*=\s*\[([^\]]+)\]', mlir)
res_names_match = re.search(r'resNames\s*=\s*\[([^\]]+)\]', mlir)

# Extract values and convert to lists
arg_names = arg_names_match.group(1).replace('"', '').split(', ') if arg_names_match else []
res_names = res_names_match.group(1).replace('"', '').split(', ') if res_names_match else []

# print(arg_names)
# print(res_names)

inf = args.inf

# TODO add as argument
buffer_size = args.N


# TODO proper include
print(f'#include "{os.path.basename(args.mlir)}"')
print(
"""
#ifndef BOOL_INPUT
#define BOOL_INPUT
MODULE bool_input(nReady0, max_tokens)
  VAR dataOut0 : boolean;
  VAR counter : 0..31;
  ASSIGN
  init(counter) := 0;
  next(counter) := case
    nReady0 & counter < max_tokens : counter + 1;
    TRUE : counter;
  esac;
  
  -- bool_input persistent
  ASSIGN
  next(dataOut0) := case 
    valid0 & !nReady0 : dataOut0;
    TRUE : {TRUE, FALSE};
  esac;
  DEFINE valid0 := counter < max_tokens;
MODULE bool_input_inf(nReady0)
  VAR dataOut0 : boolean;
  
  -- bool_input persistent
  ASSIGN
  next(dataOut0) := case 
    valid0 & !nReady0 : dataOut0;
    TRUE : {TRUE, FALSE};
  esac;
  DEFINE valid0 := TRUE;
#endif // BOOL_INPUT

MODULE main
"""
)


for i, arg in enumerate(arg_names):
  if inf:
    print(f"VAR seq_generator{i} : bool_input_inf(in_ndw{i}.ready0);")
  else:
    print(f"VAR seq_generator{i} : bool_input(in_ndw{i}.ready0, {buffer_size});")

  print(f"VAR in_ndw{i} : ndw_1_1(seq_generator{i}.dataOut0, seq_generator{i}.valid0, miter.{arg}_ready);")
  # print(f"VAR seq_generator{i} : entry_0_1(miter.{arg}_ready);")

print()


def create_miter_call(arg, res):
  miter = "VAR miter : elastic_miter("
  miter += ", ".join([f"in_ndw{i}.dataOut0, in_ndw{i}.valid0" for i, _ in enumerate(arg)])
  miter += ", "
  miter += ", ".join([f"out_ndw{i}.ready0" for i, _ in enumerate(res)])
  miter += ");\n"
  print(miter)


create_miter_call(arg_names, res_names)

# TODO
print("-- TODO make sure we have sink_1_0")
for i, res in enumerate(res_names):
  print(f"VAR out_ndw{i} : ndw_1_1(miter.{res}_out, miter.{res}_valid, sink{i}.ready0);")
  print(f"VAR sink{i} : sink_1_0(out_ndw{i}.dataOut0, out_ndw{i}.valid0);")

print()
