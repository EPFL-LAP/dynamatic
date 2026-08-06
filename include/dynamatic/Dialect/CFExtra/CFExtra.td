#ifndef CF_EXTRA_OPS_TD
#define CF_EXTRA_OPS_TD

include "mlir/IR/OpBase.td"

def CFExtra_Dialect : Dialect {
  let name = "cf_extra";
  let cppNamespace = "::dynamatic::cf_extra";
  let summary = "Custom control flow extension dialect";
}

class CFExtra_Op<string mnemonic, list<Trait> traits = []> :
    Op<CFExtra_Dialect, mnemonic, traits>;

// AllTypesMatch enforces that $data and $result must be the same type
def PredicateOp : CFExtra_Op<"predicate", [AllTypesMatch<["data", "result"]>]> {
  let summary = "Gates or suppresses data tokens based on a condition";

  let arguments = (ins
    I1:$cond,
    AnyType:$data
  );

  let results = (outs
    AnyType:$result
  );

  let assemblyFormat = [{
    `cond` `=` $cond `,` `data` `=` $data
    attr-dict `:` type($data)
  }];
}

#endif // CF_EXTRA_OPS_TD