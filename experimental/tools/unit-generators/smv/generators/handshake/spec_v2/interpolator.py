def generate_spec_v2_interpolator(name, _):
    return f"""
MODULE {name}(short, short_valid, long, long_valid, result_ready)
  VAR
  interpolate : boolean;
  ASSIGN
  init(interpolate) := FALSE;
  next(interpolate) := case
    transfer & !interpolate & !short & long : TRUE;
    transfer & interpolate & !long : FALSE;
    TRUE : interpolate;
  esac;

  DEFINE
  transfer := interpolate ? long_valid & result_ready : short_valid & long_valid & result_ready;

  // output
  result := interpolate ? FALSE : short;
  result_valid := interpolate ? long_valid : short_valid & long_valid;
  short_ready := interpolate ? FALSE : long_valid & result_ready;
  long_ready := interpolate ? result_ready : short_valid & result_ready;
"""
