#ifndef DYNAMATIC_EXPERIMENTAL_ELASTIC_MITER_CONSTRAINTS_H
#define DYNAMATIC_EXPERIMENTAL_ELASTIC_MITER_CONSTRAINTS_H

#include "FabricGeneration.h"
#include <cstddef>
#include <string>

namespace dynamatic::experimental {

class ElasticMiterConstraint {
public:
  virtual ~ElasticMiterConstraint() = default;
  virtual std::string createSmvConstraint(
      const std::string &moduleName,
      const dynamatic::experimental::ElasticMiterConfig &config) const = 0;
};

// A class to describe a Sequence Length Relation constraint.
// It controls the relative lengths of the input sequences.
// The constraint has the form of an arithmetic equation. Once we for the actual
// SMV constraint, the number in the equation will be replaced the respective
// input with the index of the number.
class SequenceLengthRelationConstraint : public ElasticMiterConstraint {
public:
  // We don't need to parse a SequenceLengthRelationConstraint, we can just copy
  // the constraint string.
  SequenceLengthRelationConstraint(const std::string &option)
      : constraint(option) {};
  std::string createSmvConstraint(
      const std::string &moduleName,
      const dynamatic::experimental::ElasticMiterConfig &config) const override;

private:
  std::string constraint;
};

// A class to describe a token limit constraint. The number of tokens at the
// input with index inputSequence can only be up to "limit" higher than the
// number of tokens at the ouput with the index outputSequence.
class TokenLimitConstraint : public ElasticMiterConstraint {
public:
  TokenLimitConstraint(const std::string &option) { parseString(option); };
  std::string createSmvConstraint(
      const std::string &moduleName,
      const dynamatic::experimental::ElasticMiterConfig &config) const override;

private:
  void parseString(const std::string &option);
  size_t inputSequence;
  size_t outputSequence;
  size_t limit;
};

// A class to describe a loop condition constraint. The number of tokens in the
// input with the index dataSequence is equivalent to the number of false tokens
// at the output with the index controlSequence. If lastFalse is set, the last
// token in the controlSequence needs to be false.
class LoopConstraint : public ElasticMiterConstraint {
public:
  LoopConstraint(const std::string &option) { parseString(option); };
  std::string
  createSmvConstraint(const std::string &moduleName,
                      const dynamatic::experimental::ElasticMiterConfig &config)
      const override {
    // Call the more general method with the option for the last token to be
    // false disabled.
    return createConstraintString(moduleName, config, false);
  };
  std::string createConstraintString(
      const std::string &moduleName,
      const dynamatic::experimental::ElasticMiterConfig &config,
      bool lastFalse) const;

private:
  void parseString(const std::string &option);
  size_t dataSequence;
  size_t controlSequence;
};

// The same as a LoopConstraint, but the last token needs to be false.
class StrictLoopConstraint : public LoopConstraint {
public:
  StrictLoopConstraint(const std::string &option) : LoopConstraint(option) {};
  std::string
  createSmvConstraint(const std::string &moduleName,
                      const dynamatic::experimental::ElasticMiterConfig &config)
      const override {
    // Call the LoopConstraint's method with the option for the last token to be
    // false.
    return LoopConstraint::createConstraintString(moduleName, config, true);
  };
};
} // namespace dynamatic::experimental

#endif // DYNAMATIC_EXPERIMENTAL_ELASTIC_MITER_CONSTRAINTS_H
