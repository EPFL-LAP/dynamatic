def generate_ndconstant(name, _):
    return f"""
MODULE {name}(ctrl_valid, result_ready)
  VAR result : boolean;
  ASSIGN
  init(result) := {{TRUE, FALSE}};
  next(result) := case
    result_valid & result_ready : {{TRUE, FALSE}};
    TRUE : result;
  esac;
  DEFINE
  result_valid := ctrl_valid;
  ctrl_ready := result_ready;
"""
