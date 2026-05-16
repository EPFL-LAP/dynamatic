# Preprocessor Pragmas 

## Our Pragmas

Dynamatic currently defines only a single custom pragma, the speculate pragma.

Its syntax is as follows:

```
#pragma DYN speculate variable=a max_predictions=b style=c
```

Which will then place a speculator between the definition of `a` and uses of `a`, which can make a max of `b` in-flight predictions, and predicts according to style `c`. Currently the only style is `standard`, where the speculator predicts initially a value of `0`, but on receiving an input predicts future inputs to be the same as its most recent input. 


## High-Level Overview

Clang's preprocessor allows the use of plugins for registering new custom pragmas, also known as preprocessor directives, into its preprocessor.

Dynamatic uses a single plugin with the "DYN" namespace to register its custom pragma (currently only one). The plugin system means that implementation is very simple, but as it happens at the preprocessor level, the transformations applied must be representable in C source code. 

The pragma plugin therefore has similiar capabilities of the Source Rewriter based on the Transformer system, but the Transformer system does not work on pragmas: the two approaches therefore do very similiar work but on two different input types.

## PragmaHandlerRegistry::Add, PragmaNamespace, PragmaHandler, and HandlePragma()

The four constructs we use in our plugin to interface with clang's preprocessor are `PragmaHandlerRegistery::Add`, `PragmaNamespace`, `PragmaHandler` and `HandlePragma`.

We first construct a `PragmaHandler`, which has two parts. The first is a `HandlePragma` function, which clang's preprocessor will call when they encounter our pragma. The second is our pragma's name, which is set through a base class constructor call. 

For example, with the speculate pragma, the PragmaHandler looks like this:
```c++
// PragmaHandler to handle the speculate pragma
class SpeculatePragmaHandler : public PragmaHandler {
public:
  // matches to #pragma DYN speculate
  SpeculatePragmaHandler() : PragmaHandler("speculate") {}

  // what to do when the pragma is encountered
  void HandlePragma(Preprocessor &PP, PragmaIntroducer Introducer,
                    Token &FirstToken) override {
    // Handle Pragma Body
  }
}
```

The `HandlePragma` takes three args: the `Preprocessor` itself, a `PragmaIntroducer` (which is a small options struct), and `Token` (which contains information about the pragma name that was just lexed and then triggered the function call).

The `HandlePragma` function should have a roughly identical top-level flow regardless of what it does:

```c++
  // what to do when the pragma is encountered
  void HandlePragma(Preprocessor &PP, PragmaIntroducer Introducer,
                    Token &FirstToken) override {
    // initialize the struct we use to store
    // the pragma information
    PragmaInfo PragmaInfo;

    // populate the fields of pragmaInfo
    // by parsing the pragma text
    if (failed(parseOptions(PP, FirstToken, PragmaInfo)))
      return;

    // then do what you want to do
    // based on the pragma options
    takeAction(PP, PragmaInfo);
  }
```

For parsing the pragma options, you must manually lex the pragma contents using the [preprocessor](https://clang.llvm.org/doxygen/classclang_1_1Preprocessor.html).

Once all of our `PragmaHandler` objects are defined, we declare our `PragmaNamespace` object:
```c++
// Declare the DynPragmaNamespace object
// which stores all of our Dyn pragmas
class DynPragmaNamespace : public PragmaNamespace {
public:
  // set the "name" of the namespace to DYN
  DynPragmaNamespace() : PragmaNamespace("DYN") {
    // put the speculate pragma in the DYN namespace
    AddPragma(new SpeculatePragmaHandler());
  }
};
```

This contains the "name" of our namespace and also contains all of our pragmas.

We then use  `PragmaHandlerRegistery::Add` to register our pragma namespace into clang's preprocessor. 

```c++
// Registration object which adds the DynPragmaNamespace
// to clang's PragmaHandlerRegistry
// so that clang nows how to handle pragmas in
// the DYN pragma namespace
static PragmaHandlerRegistry::Add<DynPragmaNamespace>
    X("DYN", "Dynamatic pragma namespace");

```

## Speculate Pragma's HandlePragma body

The speculate pragma does not use the `PragmaIntroducer` struct, which contains mostly irrelevant information for our use case.

### parsePragmaOptions 

The simplest way to explain our parsing is with pseudocode:

```
// struct we want to populate
optionsStruct = OptionsStruct();

// initial token lex for first iteration
token = lexNextToken();
while(token is not end of line){
  assert(token is a word); // loop re-starts for each option
  namedOption = token;

  token = lexNextToken();
  assert(token is a equals sign);

  if(namedOption == variable){
    token = lexNextToken();
    assert(token is a word); // we expect the variable name

    optionsStruct.variable = token;

    // initial token lex for next iteration
    token = lexNextToken();
  } else if (namedOption == max_predictions){
    token = lexNextToken();

    // use existing clang utilities
    // to store the token as an integer literal
    // directly inside the struct
    //
    // It also updates token
    useClangUtilityToConvertTokenToUI64(token, optionsStruct.max_predictions);

    // no initial lex for next iteration
    // since the utility function did it
  } else if (namedOption == style){
    token = lexNextToken();

    assert(token is a word); // we expect the style name

    checkSpeculateStyleIsWhitelisted(token)

    optionsStruct.style = token;

    // initial token lex for next iteration
    token = lexNextToken();
  } else {
    throw_error("unrecognized named option);
  }
}

assert(every named option was found);

return optionsStruct;
```

### emitInjectedTokens

Our "do something" function for the speculate pragma is 
```c++
void emitInjectedTokens(Preprocessor &PP,
                          const SpeculatePragmaInfo &SpecPragmaInfo);
```

which injects a `extern int __dyn_speculate(double, int, const char *)` function declaration: this is the function we use to represent the speculator in the C source kernel.

It then also injects a call of this function, converting 
```
#pragma DYN speculate variable=input_variable max_predictions=5 style=standard
```

to

```
input_variable = __dyn_speculate(input_variable, 5, "standard");
```
Redefining any downstream uses of `input_variable` to be uses of the speculator's output. You may also notice the style gets wrapped in quote marks, as otherwise it will not be valid C syntax. The pragma style input should not contain quote marks.

This is done using a method inspired by how the preprocessor handles included header files. 

The steps are roughly as follows:
1. Build a fake generic file containing the function declaration and call.
2. Convert the generic file to a fake "include", which knows the pragma is the line of code that caused it to be included.
3. Tell the preprocessor to actually include our fake "include".