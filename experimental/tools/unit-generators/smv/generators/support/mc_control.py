def generate_mc_control(name):
    return f"""
MODULE {name}(memStart_valid, memEnd_ready, ctrlEnd_valid, all_requests_done)
  -- the mc_control manages the signals connected to the memory and controls when to
  -- start accessing memory and when no more accesses will be made

  VAR
  fsm_state : {{IDLE, RUNNING}};
  no_more_request : boolean;
  
  DEFINE
  fsm_running := (fsm_state = RUNNING);
  
  -- Function is returning if:
  -- 1. There are no more requests (from the circuit)
  -- 2. All pending requests are done (from the controller's counter)
  -- 3. MemEnd pin is ready
  -- 4. The function is running
  function_return := (no_more_request & all_requests_done & memEnd_ready & fsm_running);

  -- We can start the memory only if it is not started yet (IDLE).
  -- During the IDLE state, it can always start.
  memStart_ready := (fsm_state = IDLE);

  -- The memory is okay to return if:
  -- 1. There is no more requests
  -- 2. All pending requests are done
  memEnd_valid := (no_more_request & all_requests_done & fsm_running);

  -- We can accept that a control end once per function execution.
  -- It cannot accept it again before we return.
  ctrlEnd_ready := !no_more_request;

  ASSIGN
  init(fsm_state) := IDLE;
  next(fsm_state) := case
    fsm_state = IDLE : memStart_valid ? RUNNING : fsm_state;
    function_return  : IDLE;
    TRUE : fsm_state;
  esac;

  init(no_more_request) := FALSE;
  next(no_more_request) := case
    function_return : FALSE;
    ctrlEnd_valid : TRUE;
    TRUE : no_more_request;
  esac;
"""
