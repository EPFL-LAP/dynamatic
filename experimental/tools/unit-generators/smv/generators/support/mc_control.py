def generate_mc_control(name):
    return f"""
MODULE {name}(memStart_valid, memEnd_ready, ctrlEnd_valid, all_requests_done)
  -- the mc_control manages the signals connected to the memory and controls when to
  -- start accessing memory and when no more accesses will be made
  VAR
  memStart_ready_in : boolean;
  memEnd_valid_in : boolean;
  ctrlEnd_ready_in : boolean;
  ASSIGN
  init(memStart_ready_in) := TRUE;
  next(memStart_ready_in) := case
    memEnd_valid_in & memEnd_ready : TRUE;
    memStart_valid & memStart_ready_in : FALSE;
    TRUE : memStart_ready_in;
  esac;
  init(memEnd_valid_in) := FALSE;
  next(memEnd_valid_in) := case
    memEnd_valid_in & memEnd_ready : FALSE;
    ctrlEnd_valid & all_requests_done : TRUE;
    TRUE : memEnd_valid_in;
  esac;
  init(ctrlEnd_ready_in) := FALSE;
  next(ctrlEnd_ready_in) := case
    ctrlEnd_valid & ctrlEnd_ready_in : FALSE;
    ctrlEnd_valid & all_requests_done : TRUE;
    TRUE : ctrlEnd_ready_in;
  esac;
  // outputs
  DEFINE
  memStart_ready := memStart_ready_in;
  memEnd_valid := memEnd_valid_in;
  ctrlEnd_ready := ctrlEnd_ready_in;
"""
