def generate_buffer_counter(name, slots):
    return f"""
MODULE {name}(ins_valid, ins_ready, outs_valid, outs_ready)
    VAR
    counter : 0..{slots};
    error : boolean;
    DEFINE
    write_en := ins_valid & ins_ready;
    read_en := outs_valid & outs_ready;
    ASSIGN
    init(counter) := 0;
    next(counter) := case
        write_en & read_en : counter;
        write_en & (counter < {slots}) : counter + 1;
        read_en & (counter > 0) : counter - 1;
        TRUE : counter;
    esac;
    init(error) := FALSE;
    next(error) := case
        write_en & read_en : error;
        write_en & (counter = {slots}) : TRUE;
        read_en & (counter = 0) : TRUE;
        TRUE : error;
    esac;
"""
