#ifndef ELASTIC_MITER_CONSTRAINTS_H
#define ELASTIC_MITER_CONSTRAINTS_H

#include "FabricGeneration.h"
#include "llvm/Support/raw_ostream.h"
#include <cstddef>
#include <regex>
#include <string>
#include <utility>

namespace dynamatic::experimental {
class ElasticMiterConstraint {
public:
  virtual ~ElasticMiterConstraint() = default;
  virtual std::string createConstraintString(
      const std::string &moduleName,
      const dynamatic::experimental::ElasticMiterConfig &config) const = 0;
};

// Parse the sequence length relation constraints. They are string in the
// style "0+1+..=4+5+..", where the numbers represent the index of the
// sequence TODO
class SequenceLengthRelationConstraint : public ElasticMiterConstraint {
public:
  SequenceLengthRelationConstraint(const std::string &option)
      : constraint(option) {};
  std::string createConstraintString(
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
  std::string createConstraintString(
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
  createConstraintString(const std::string &moduleName,
                         const dynamatic::experimental::ElasticMiterConfig
                             &config) const override {
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

class StrictLoopConstraint : public LoopConstraint {
public:
  StrictLoopConstraint(const std::string &option) : LoopConstraint(option) {};
  std::string
  createConstraintString(const std::string &moduleName,
                         const dynamatic::experimental::ElasticMiterConfig
                             &config) const override {
    return LoopConstraint::createConstraintString(moduleName, config, true);
  };
};
} // namespace dynamatic::experimental
#endif // ELASTIC_MITER_CONSTRAINTS_H
