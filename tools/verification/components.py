import sympy

# =====================================================================
# THE ARCHITECTURAL COMPONENT DATABASE (LIBRARY)
# =====================================================================
class ComponentLibrary:
    """
    A unified component specification library. 
    Keeps track of structural routing laws for dataflow components.
    """
    @staticmethod
    def apply_fork(circuit, source, targets, is_boolean=True):
        """
        Fork equation component tracker.
        Invariants: All forward copies match the source profile 1:1.
        source_c == target_0_c == target_1_c ...
        source_t == target_0_t == target_1_t ...
        """
        v_src = circuit.get_ch(source, is_boolean=is_boolean)
        for target in targets:
            v_tgt = circuit.get_ch(target, is_boolean=is_boolean)
            
            # Count equations
            circuit.equations.append(sympy.Eq(v_tgt[0], v_src[0]))
            
            # True fraction data tracking matching equations
            src_t = circuit._get_val(v_src, 1)
            tgt_t = circuit._get_val(v_tgt, 1)
            circuit.equations.append(sympy.Eq(tgt_t, src_t))

    @staticmethod
    def apply_mux(circuit, cond, inT, inF, out, is_boolean=True):
        v_cond = circuit.get_ch(cond, is_boolean=True)
        v_inT  = circuit.get_ch(inT, is_boolean=is_boolean)
        v_inF  = circuit.get_ch(inF, is_boolean=is_boolean)
        v_out  = circuit.get_ch(out, is_boolean=is_boolean)
        
        circuit.equations.append(sympy.Eq(v_inT[0], v_cond[1]))
        circuit.equations.append(sympy.Eq(v_inF[0], v_cond[0] - v_cond[1]))
        circuit.equations.append(sympy.Eq(v_out[0], v_cond[0]))
        
        inT_t = circuit._get_val(v_inT, 1)
        inF_t = circuit._get_val(v_inF, 1)
        out_t = circuit._get_val(v_out, 1)
        circuit.equations.append(sympy.Eq(out_t, inT_t + inF_t))

    @staticmethod
    def apply_branch(circuit, cond, in_ch, outT, outF, is_boolean=True):
        v_cond = circuit.get_ch(cond, is_boolean=True)
        v_in   = circuit.get_ch(in_ch, is_boolean=is_boolean)
        v_outT = circuit.get_ch(outT, is_boolean=is_boolean)
        v_outF = circuit.get_ch(outF, is_boolean=is_boolean)

        circuit.equations.append(sympy.Eq(v_in[0], v_cond[0]))
        circuit.equations.append(sympy.Eq(v_outT[0], v_cond[1]))
        circuit.equations.append(sympy.Eq(v_outF[0], v_cond[0] - v_cond[1]))
        
        in_t   = circuit._get_val(v_in, 1)
        outT_t = circuit._get_val(v_outT, 1)
        outF_t = circuit._get_val(v_outF, 1)
        circuit.equations.append(sympy.Eq(in_t, outT_t + outF_t))

    # @staticmethod
    # def apply_cmerge(circuit, inT, inF, out, select, is_boolean=True):
    #     v_inT    = circuit.get_ch(inT, is_boolean=is_boolean)
    #     v_inF    = circuit.get_ch(inF, is_boolean=is_boolean)
    #     v_out    = circuit.get_ch(out, is_boolean=is_boolean)
    #     v_select = circuit.get_ch(select, is_boolean=True)

    #     circuit.equations.append(sympy.Eq(v_out[0], v_inT[0] + v_inF[0]))
    #     circuit.equations.append(sympy.Eq(v_select[0], v_inT[0] + v_inF[0]))
    #     circuit.equations.append(sympy.Eq(v_select[1], v_inT[0]))
        
    #     inT_t = circuit._get_val(v_inT, 1)
    #     inF_t = circuit._get_val(v_inF, 1)
    #     out_t = circuit._get_val(v_out, 1)
    #     circuit.equations.append(sympy.Eq(out_t, inT_t + inF_t))


    @staticmethod
    def apply_cmerge(circuit, inputs, out, select, is_boolean=False):
        """
        Adds a Control Merge module handling an arbitrary number of inputs.
        Sums up counts dynamically. For single inputs, it collapses down 
        to a basic 1:1 forwarder pass-through structure.
        """
        v_out    = circuit.get_ch(out, is_boolean=is_boolean)
        v_select = circuit.get_ch(select, is_boolean=True)
        
        # 1. Map output total transaction rates
        input_tuples = [circuit.get_ch(inp, is_boolean=is_boolean) for inp in inputs]
        total_input_count = sum(v_inp[0] for v_inp in input_tuples)
        
        circuit.equations.append(sympy.Eq(v_out[0], total_input_count))
        circuit.equations.append(sympy.Eq(v_select[0], total_input_count))
        
        # 2. Extract specific indexed dataflows based on input count
        v_inT = input_tuples[0]
        v_inF = input_tuples[1] if len(input_tuples) > 1 else None
        
        # True Select index balance tracking
        circuit.equations.append(sympy.Eq(v_select[1], v_inT[0]))
        
        # True fractional properties balance mapping
        inT_t = circuit._get_val(v_inT, 1)
        inF_t = circuit._get_val(v_inF, 1) if v_inF is not None else 0
        out_t = circuit._get_val(v_out, 1)
        
        circuit.equations.append(sympy.Eq(out_t, inT_t + inF_t))

    @staticmethod
    def apply_join(circuit, inLeft, inRight, out, is_boolean=True):
        v_inLeft  = circuit.get_ch(inLeft, is_boolean=is_boolean)
        v_inRight = circuit.get_ch(inRight, is_boolean=is_boolean)
        v_out     = circuit.get_ch(out, is_boolean=is_boolean)

        circuit.equations.append(sympy.Eq(v_inLeft[0], v_inRight[0]))
        circuit.equations.append(sympy.Eq(v_inLeft[0], v_out[0]))

        inLeft_t  = circuit._get_val(v_inLeft, 1)
        inRight_t = circuit._get_val(v_inRight, 1)
        out_t     = circuit._get_val(v_out, 1)
        
        circuit.equations.append(sympy.Eq(inLeft_t, inRight_t))
        circuit.equations.append(sympy.Eq(inLeft_t, out_t))

    @staticmethod
    def apply_computation(circuit, inputs, out, is_boolean=False):
        """
        Applies a multi-input synchronization constraint (like arithmetic/comparison).
        All inputs must be synchronized to compute and emit the output product.
        in_1_c == in_2_c == ... == out_c
        """
        v_out = circuit.get_ch(out, is_boolean=is_boolean)
        out_t = circuit._get_val(v_out, 1)
        
        for inp in inputs:
            v_inp = circuit.get_ch(inp, is_boolean=is_boolean)
            # Enforce execution token counts balance perfectly
            circuit.equations.append(sympy.Eq(v_inp[0], v_out[0]))
            
            # True values balance out 1:1 if boolean parameters are active
            inp_t = circuit._get_val(v_inp, 1)
            circuit.equations.append(sympy.Eq(inp_t, out_t))

    @staticmethod
    def apply_comparator(circuit, inputs, out, p_symbol):
        """
        Applies a comparator condition join constraint.
        Inputs synchronize structurally, but the true-value channel scales 
        based on the evaluation probability parameter: out_t == p_i * out_c
        """
        v_out = circuit.get_ch(out, is_boolean=True)
        out_c, out_t = v_out[0], v_out[1]
        
        # 1. Enforce count synchronization among all inputs and the output
        for inp in inputs:
            v_inp = circuit.get_ch(inp, is_boolean=False)
            circuit.equations.append(sympy.Eq(v_inp[0], out_c))
            
        # 2. Assert the probabilistic true data path tracking equation
        circuit.equations.append(sympy.Eq(out_t, p_symbol * out_c))