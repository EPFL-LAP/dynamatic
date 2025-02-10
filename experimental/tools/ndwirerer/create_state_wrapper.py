import re
import os


def create_miter_call(arg, res):
  miter = "VAR miter : elastic_miter("
  miter += ", ".join([f"in_ndw{i}.dataOut0, in_ndw{i}.valid0" for i, _ in enumerate(arg)])
  miter += ", "
  miter += ", ".join([f"out_ndw{i}.ready0" for i, _ in enumerate(res)])
  miter += ");\n"
  return miter


def create_state_wrapper(smv, mlir, N=0, inf=False):

  wrapper = ""
  assert((N == 0 and inf) or (N != 0 and not inf), "Either inf must be set or N>0.")

  with open(mlir) as f:
    mlir = f.read()

  # print(mlir)

  arg_names_match = re.search(r'argNames\s*=\s*\[([^\]]+)\]', mlir)
  res_names_match = re.search(r'resNames\s*=\s*\[([^\]]+)\]', mlir)

  # Extract values and convert to lists
  arg_names = arg_names_match.group(1).replace('"', '').split(', ') if arg_names_match else []
  res_names = res_names_match.group(1).replace('"', '').split(', ') if res_names_match else []

  # print(arg_names)
  # print(res_names)

  # TODO add as argument
  buffer_size = N


  # TODO proper include
  wrapper += (f'#include "{os.path.basename(smv)}"\n')
  wrapper += \
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


  for i, arg in enumerate(arg_names):
    if inf:
      wrapper += (f"VAR seq_generator{i} : bool_input_inf(in_ndw{i}.ready0);\n")
    else:
      wrapper += (f"VAR seq_generator{i} : bool_input(in_ndw{i}.ready0, {buffer_size});\n")

    wrapper += (f"VAR in_ndw{i} : ndw_1_1(seq_generator{i}.dataOut0, seq_generator{i}.valid0, miter.{arg}_ready);\n")
    # print(f"VAR seq_generator{i} : entry_0_1(miter.{arg}_ready);")

  wrapper += "\n"


  wrapper += create_miter_call(arg_names, res_names)

  # TODO
  wrapper += ("-- TODO make sure we have sink_1_0\n")
  for i, res in enumerate(res_names):
    wrapper += (f"VAR out_ndw{i} : ndw_1_1(miter.{res}_out, miter.{res}_valid, sink{i}.ready0);\n")
    wrapper += (f"VAR sink{i} : sink_1_0(out_ndw{i}.dataOut0, out_ndw{i}.valid0);\n")

  return wrapper
