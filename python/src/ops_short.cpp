// Copyright © 2023-2024 Apple Inc.

#include <numeric>
#include <ostream>
#include <variant>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include "mlx/einsum.h"
#include "mlx/ops.h"
#include "mlx/utils.h"
#include "python/src/load.h"
#include "python/src/utils.h"

namespace mx = mlx::core;
namespace nb = nanobind;
using namespace nb::literals;

using Scalar = std::variant<bool, int, double>;

mx::Dtype scalar_to_dtype(Scalar s) {
  if (std::holds_alternative<int>(s)) {
    return mx::int32;
  } else if (std::holds_alternative<double>(s)) {
    return mx::float32;
  } else {
    return mx::bool_;
  }
}

double scalar_to_double(Scalar s) {
  if (auto pv = std::get_if<int>(&s); pv) {
    return static_cast<double>(*pv);
  } else if (auto pv = std::get_if<double>(&s); pv) {
    return *pv;
  } else {
    return static_cast<double>(std::get<bool>(s));
  }
}

void init_ops(nb::module_& m) {
  m.def(
      "reshape",
      &mx::reshape,
      nb::arg(),
      "shape"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def reshape(a: array, /, shape: Sequence[int], *, stream: "
              "Union[None, Stream, Device] = None) -> array"),
  m.def(
      "flatten",
      [](const mx::array& a,
         int start_axis,
         int end_axis,
         const mx::StreamOrDevice& s) {
        return mx::flatten(a, start_axis, end_axis);
      },
      nb::arg(),
      "start_axis"_a = 0,
      "end_axis"_a = -1,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def flatten(a: array, /, start_axis: int = 0, end_axis: int = "
              "-1, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "unflatten",
      &mx::unflatten,
      nb::arg(),
      "axis"_a,
      "shape"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def unflatten(a: array, /, axis: int, shape: Sequence[int], *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "squeeze",
      [](const mx::array& a, const IntOrVec& v, const mx::StreamOrDevice& s) {
        if (std::holds_alternative<std::monostate>(v)) {
          return mx::squeeze(a, s);
        } else if (auto pv = std::get_if<int>(&v); pv) {
          return mx::squeeze(a, *pv, s);
        } else {
          return mx::squeeze(a, std::get<std::vector<int>>(v), s);
        }
      },
      nb::arg(),
      "axis"_a = nb::none(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def squeeze(a: array, /, axis: Union[None, int, Sequence[int]] = "
          "None, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "expand_dims",
      [](const mx::array& a,
         const std::variant<int, std::vector<int>>& v,
         mx::StreamOrDevice s) {
        if (auto pv = std::get_if<int>(&v); pv) {
          return mx::expand_dims(a, *pv, s);
        } else {
          return mx::expand_dims(a, std::get<std::vector<int>>(v), s);
        }
      },
      nb::arg(),
      "axis"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def expand_dims(a: array, /, axis: Union[int, Sequence[int]], "
              "*, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "abs",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::abs(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def abs(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "sign",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::sign(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def sign(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "negative",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::negative(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def negative(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "add",
      [](const ScalarOrArray& a_,
         const ScalarOrArray& b_,
         mx::StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return mx::add(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def add(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "subtract",
      [](const ScalarOrArray& a_,
         const ScalarOrArray& b_,
         mx::StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return mx::subtract(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def subtract(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "multiply",
      [](const ScalarOrArray& a_,
         const ScalarOrArray& b_,
         mx::StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return mx::multiply(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def multiply(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "divide",
      [](const ScalarOrArray& a_,
         const ScalarOrArray& b_,
         mx::StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return mx::divide(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def divide(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "divmod",
      [](const ScalarOrArray& a_,
         const ScalarOrArray& b_,
         mx::StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return mx::divmod(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def divmod(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "floor_divide",
      [](const ScalarOrArray& a_,
         const ScalarOrArray& b_,
         mx::StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return mx::floor_divide(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def floor_divide(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "remainder",
      [](const ScalarOrArray& a_,
         const ScalarOrArray& b_,
         mx::StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return mx::remainder(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def remainder(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "equal",
      [](const ScalarOrArray& a_,
         const ScalarOrArray& b_,
         mx::StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return mx::equal(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def equal(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "not_equal",
      [](const ScalarOrArray& a_,
         const ScalarOrArray& b_,
         mx::StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return mx::not_equal(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def not_equal(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "less",
      [](const ScalarOrArray& a_,
         const ScalarOrArray& b_,
         mx::StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return mx::less(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def less(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "less_equal",
      [](const ScalarOrArray& a_,
         const ScalarOrArray& b_,
         mx::StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return mx::less_equal(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def less_equal(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "greater",
      [](const ScalarOrArray& a_,
         const ScalarOrArray& b_,
         mx::StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return mx::greater(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def greater(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "greater_equal",
      [](const ScalarOrArray& a_,
         const ScalarOrArray& b_,
         mx::StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return mx::greater_equal(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def greater_equal(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "array_equal",
      [](const ScalarOrArray& a_,
         const ScalarOrArray& b_,
         bool equal_nan,
         mx::StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return mx::array_equal(a, b, equal_nan, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "equal_nan"_a = false,
      "stream"_a = nb::none(),
      nb::sig(
          "def array_equal(a: Union[scalar, array], b: Union[scalar, array], equal_nan: bool = False, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "matmul",
      &mx::matmul,
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def matmul(a: array, b: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "square",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::square(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def square(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "sqrt",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::sqrt(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def sqrt(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "rsqrt",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::rsqrt(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def rsqrt(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "reciprocal",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::reciprocal(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def reciprocal(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "logical_not",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::logical_not(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def logical_not(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "logical_and",
      [](const ScalarOrArray& a, const ScalarOrArray& b, mx::StreamOrDevice s) {
        return mx::logical_and(to_array(a), to_array(b), s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def logical_and(a: array, b: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),

  m.def(
      "logical_or",
      [](const ScalarOrArray& a, const ScalarOrArray& b, mx::StreamOrDevice s) {
        return mx::logical_or(to_array(a), to_array(b), s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def logical_or(a: array, b: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "logaddexp",
      [](const ScalarOrArray& a_,
         const ScalarOrArray& b_,
         mx::StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return mx::logaddexp(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def logaddexp(a: Union[scalar, array], b: Union[scalar, array], /, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "exp",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::exp(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def exp(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "expm1",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::expm1(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def expm1(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "erf",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::erf(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def erf(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "erfinv",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::erfinv(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def erfinv(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "sin",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::sin(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def sin(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "cos",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::cos(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def cos(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "tan",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::tan(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def tan(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "arcsin",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::arcsin(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def arcsin(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "arccos",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::arccos(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def arccos(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "arctan",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::arctan(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def arctan(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "arctan2",
      &mx::arctan2,
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def arctan2(a: array, b: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "sinh",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::sinh(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def sinh(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "cosh",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::cosh(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def cosh(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "tanh",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::tanh(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def tanh(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "arcsinh",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::arcsinh(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def arcsinh(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "arccosh",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::arccosh(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def arccosh(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "arctanh",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::arctanh(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def arctanh(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "degrees",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::degrees(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def degrees(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "radians",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::radians(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def radians(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "log",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::log(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def log(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "log2",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::log2(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def log2(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "log10",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::log10(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def log10(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "log1p",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::log1p(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def log1p(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "stop_gradient",
      &mx::stop_gradient,
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def stop_gradient(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "sigmoid",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::sigmoid(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def sigmoid(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "power",
      [](const ScalarOrArray& a_,
         const ScalarOrArray& b_,
         mx::StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return mx::power(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def power(a: Union[scalar, array], b: Union[scalar, array], /, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "arange",
      [](Scalar start,
         Scalar stop,
         const std::optional<Scalar>& step,
         const std::optional<mx::Dtype>& dtype_,
         mx::StreamOrDevice s) {
        // Determine the final dtype based on input types
        mx::Dtype dtype = dtype_
            ? *dtype_
            : mx::promote_types(
                  scalar_to_dtype(start),
                  step ? mx::promote_types(
                             scalar_to_dtype(stop), scalar_to_dtype(*step))
                       : scalar_to_dtype(stop));
        return mx::arange(
            scalar_to_double(start),
            scalar_to_double(stop),
            step ? scalar_to_double(*step) : 1.0,
            dtype,
            s);
      },
      "start"_a.noconvert(),
      "stop"_a.noconvert(),
      "step"_a.noconvert() = nb::none(),
      "dtype"_a = nb::none(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def arange(start : Union[int, float], stop : Union[int, float], step : Union[None, int, float], dtype: Optional[Dtype] = None, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "arange",
      [](Scalar stop,
         const std::optional<Scalar>& step,
         const std::optional<mx::Dtype>& dtype_,
         mx::StreamOrDevice s) {
        mx::Dtype dtype = dtype_ ? *dtype_
            : step
            ? mx::promote_types(scalar_to_dtype(stop), scalar_to_dtype(*step))
            : scalar_to_dtype(stop);
        return mx::arange(
            0.0,
            scalar_to_double(stop),
            step ? scalar_to_double(*step) : 1.0,
            dtype,
            s);
      },
      "stop"_a.noconvert(),
      "step"_a.noconvert() = nb::none(),
      "dtype"_a = nb::none(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def arange(stop : Union[int, float], step : Union[None, int, float] = None, dtype: Optional[Dtype] = None, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def(
      "linspace",
      [](Scalar start,
         Scalar stop,
         int num,
         std::optional<mx::Dtype> dtype,
         mx::StreamOrDevice s) {
        return mx::linspace(
            scalar_to_double(start),
            scalar_to_double(stop),
            num,
            dtype.value_or(mx::float32),
            s);
      },
      "start"_a,
      "stop"_a,
      "num"_a = 50,
      "dtype"_a.none() = mx::float32,
      "stream"_a = nb::none(),
      nb::sig(
          "def linspace(start, stop, num: Optional[int] = 50, dtype: Optional[Dtype] = float32, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "kron",
      &mx::kron,
      nb::arg("a"),
      nb::arg("b"),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def kron(a: array, b: array, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "take",
      [](const mx::array& a,
         const std::variant<nb::int_, mx::array>& indices,
         const std::optional<int>& axis,
         mx::StreamOrDevice s) {
        if (auto pv = std::get_if<nb::int_>(&indices); pv) {
          auto idx = nb::cast<int>(*pv);
          return axis ? mx::take(a, idx, axis.value(), s) : mx::take(a, idx, s);
        } else {
          auto indices_ = std::get<mx::array>(indices);
          return axis ? mx::take(a, indices_, axis.value(), s)
                      : mx::take(a, indices_, s);
        }
      },
      nb::arg(),
      "indices"_a,
      "axis"_a = nb::none(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def take(a: array, /, indices: Union[int, array], axis: Optional[int] = None, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "take_along_axis",
      [](const mx::array& a,
         const mx::array& indices,
         const std::optional<int>& axis,
         mx::StreamOrDevice s) {
        if (axis.has_value()) {
          return mx::take_along_axis(a, indices, axis.value(), s);
        } else {
          return mx::take_along_axis(mx::reshape(a, {-1}, s), indices, 0, s);
        }
      },
      nb::arg(),
      "indices"_a,
      "axis"_a.none(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def take_along_axis(a: array, /, indices: array, axis: Optional[int] = None, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "put_along_axis",
      [](const mx::array& a,
         const mx::array& indices,
         const mx::array& values,
         const std::optional<int>& axis,
         mx::StreamOrDevice s) {
        if (axis.has_value()) {
          return mx::put_along_axis(a, indices, values, axis.value(), s);
        } else {
          return mx::reshape(
              mx::put_along_axis(
                  mx::reshape(a, {-1}, s), indices, values, 0, s),
              a.shape(),
              s);
        }
      },
      nb::arg(),
      "indices"_a,
      "values"_a,
      "axis"_a.none(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def put_along_axis(a: array, /, indices: array, values: array, axis: Optional[int] = None, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "full",
      [](const std::variant<int, mx::Shape>& shape,
         const ScalarOrArray& vals,
         std::optional<mx::Dtype> dtype,
         mx::StreamOrDevice s) {
        if (auto pv = std::get_if<int>(&shape); pv) {
          return mx::full({*pv}, to_array(vals, dtype), s);
        } else {
          return mx::full(std::get<mx::Shape>(shape), to_array(vals, dtype), s);
        }
      },
      "shape"_a,
      "vals"_a,
      "dtype"_a = nb::none(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def full(shape: Union[int, Sequence[int]], vals: Union[scalar, array], dtype: Optional[Dtype] = None, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "zeros",
      [](const std::variant<int, mx::Shape>& shape,
         std::optional<mx::Dtype> dtype,
         mx::StreamOrDevice s) {
        auto t = dtype.value_or(mx::float32);
        if (auto pv = std::get_if<int>(&shape); pv) {
          return mx::zeros({*pv}, t, s);
        } else {
          return mx::zeros(std::get<mx::Shape>(shape), t, s);
        }
      },
      "shape"_a,
      "dtype"_a.none() = mx::float32,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def zeros(shape: Union[int, Sequence[int]], dtype: Optional[Dtype] = float32, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "zeros_like",
      &mx::zeros_like,
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def zeros_like(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "ones",
      [](const std::variant<int, mx::Shape>& shape,
         std::optional<mx::Dtype> dtype,
         mx::StreamOrDevice s) {
        auto t = dtype.value_or(mx::float32);
        if (auto pv = std::get_if<int>(&shape); pv) {
          return mx::ones({*pv}, t, s);
        } else {
          return mx::ones(std::get<mx::Shape>(shape), t, s);
        }
      },
      "shape"_a,
      "dtype"_a.none() = mx::float32,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def ones(shape: Union[int, Sequence[int]], dtype: Optional[Dtype] = float32, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "ones_like",
      &mx::ones_like,
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def ones_like(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "eye",
      [](int n,
         std::optional<int> m,
         int k,
         std::optional<mx::Dtype> dtype,
         mx::StreamOrDevice s) {
        return mx::eye(n, m.value_or(n), k, dtype.value_or(mx::float32), s);
      },
      "n"_a,
      "m"_a = nb::none(),
      "k"_a = 0,
      "dtype"_a.none() = mx::float32,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def eye(n: int, m: Optional[int] = None, k: int = 0, dtype: Optional[Dtype] = float32, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "identity",
      [](int n, std::optional<mx::Dtype> dtype, mx::StreamOrDevice s) {
        return mx::identity(n, dtype.value_or(mx::float32), s);
      },
      "n"_a,
      "dtype"_a.none() = mx::float32,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def identity(n: int, dtype: Optional[Dtype] = float32, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "tri",
      [](int n,
         std::optional<int> m,
         int k,
         std::optional<mx::Dtype> type,
         mx::StreamOrDevice s) {
        return mx::tri(n, m.value_or(n), k, type.value_or(mx::float32), s);
      },
      "n"_a,
      "m"_a = nb::none(),
      "k"_a = 0,
      "dtype"_a.none() = mx::float32,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def tri(n: int, m: int, k: int, dtype: Optional[Dtype] = None, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "tril",
      &mx::tril,
      "x"_a,
      "k"_a = 0,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def tril(x: array, k: int, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "triu",
      &mx::triu,
      "x"_a,
      "k"_a = 0,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def triu(x: array, k: int, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "allclose",
      &mx::allclose,
      nb::arg(),
      nb::arg(),
      "rtol"_a = 1e-5,
      "atol"_a = 1e-8,
      nb::kw_only(),
      "equal_nan"_a = false,
      "stream"_a = nb::none(),
      nb::sig(
          "def allclose(a: array, b: array, /, rtol: float = 1e-05, atol: float = 1e-08, *, equal_nan: bool = False, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "isclose",
      &mx::isclose,
      nb::arg(),
      nb::arg(),
      "rtol"_a = 1e-5,
      "atol"_a = 1e-8,
      nb::kw_only(),
      "equal_nan"_a = false,
      "stream"_a = nb::none(),
      nb::sig(
          "def isclose(a: array, b: array, /, rtol: float = 1e-05, atol: float = 1e-08, *, equal_nan: bool = False, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "all",
      [](const mx::array& a,
         const IntOrVec& axis,
         bool keepdims,
         mx::StreamOrDevice s) {
        return mx::all(a, get_reduce_axes(axis, a.ndim()), keepdims, s);
      },
      nb::arg(),
      "axis"_a = nb::none(),
      "keepdims"_a = false,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def all(a: array, /, axis: Union[None, int, Sequence[int]] = None, keepdims: bool = False, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "any",
      [](const mx::array& a,
         const IntOrVec& axis,
         bool keepdims,
         mx::StreamOrDevice s) {
        return mx::any(a, get_reduce_axes(axis, a.ndim()), keepdims, s);
      },
      nb::arg(),
      "axis"_a = nb::none(),
      "keepdims"_a = false,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def any(a: array, /, axis: Union[None, int, Sequence[int]] = None, keepdims: bool = False, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "minimum",
      [](const ScalarOrArray& a_,
         const ScalarOrArray& b_,
         mx::StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return mx::minimum(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def minimum(a: Union[scalar, array], b: Union[scalar, array], /, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "maximum",
      [](const ScalarOrArray& a_,
         const ScalarOrArray& b_,
         mx::StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return mx::maximum(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def maximum(a: Union[scalar, array], b: Union[scalar, array], /, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "floor",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::floor(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def floor(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "ceil",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::ceil(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def ceil(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "isnan",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::isnan(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def isnan(a: array, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "isinf",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::isinf(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def isinf(a: array, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "isfinite",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::isfinite(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def isfinite(a: array, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "isposinf",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::isposinf(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def isposinf(a: array, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "isneginf",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::isneginf(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def isneginf(a: array, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "moveaxis",
      &mx::moveaxis,
      nb::arg(),
      "source"_a,
      "destination"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def moveaxis(a: array, /, source: int, destination: int, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "swapaxes",
      &mx::swapaxes,
      nb::arg(),
      "axis1"_a,
      "axis2"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def swapaxes(a: array, /, axis1 : int, axis2: int, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "transpose",
      [](const mx::array& a,
         const std::optional<std::vector<int>>& axes,
         mx::StreamOrDevice s) {
        if (axes.has_value()) {
          return mx::transpose(a, *axes, s);
        } else {
          return mx::transpose(a, s);
        }
      },
      nb::arg(),
      "axes"_a = nb::none(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def transpose(a: array, /, axes: Optional[Sequence[int]] = None, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "permute_dims",
      [](const mx::array& a,
         const std::optional<std::vector<int>>& axes,
         mx::StreamOrDevice s) {
        if (axes.has_value()) {
          return mx::transpose(a, *axes, s);
        } else {
          return mx::transpose(a, s);
        }
      },
      nb::arg(),
      "axes"_a = nb::none(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def permute_dims(a: array, /, axes: Optional[Sequence[int]] = None, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "sum",
      [](const mx::array& a,
         const IntOrVec& axis,
         bool keepdims,
         mx::StreamOrDevice s) {
        return mx::sum(a, get_reduce_axes(axis, a.ndim()), keepdims, s);
      },
      "array"_a,
      "axis"_a = nb::none(),
      "keepdims"_a = false,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def sum(a: array, /, axis: Union[None, int, Sequence[int]] = None, keepdims: bool = False, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "prod",
      [](const mx::array& a,
         const IntOrVec& axis,
         bool keepdims,
         mx::StreamOrDevice s) {
        return mx::prod(a, get_reduce_axes(axis, a.ndim()), keepdims, s);
      },
      nb::arg(),
      "axis"_a = nb::none(),
      "keepdims"_a = false,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def prod(a: array, /, axis: Union[None, int, Sequence[int]] = None, keepdims: bool = False, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "min",
      [](const mx::array& a,
         const IntOrVec& axis,
         bool keepdims,
         mx::StreamOrDevice s) {
        return mx::min(a, get_reduce_axes(axis, a.ndim()), keepdims, s);
      },
      nb::arg(),
      "axis"_a = nb::none(),
      "keepdims"_a = false,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def min(a: array, /, axis: Union[None, int, Sequence[int]] = None, keepdims: bool = False, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "max",
      [](const mx::array& a,
         const IntOrVec& axis,
         bool keepdims,
         mx::StreamOrDevice s) {
        return mx::max(a, get_reduce_axes(axis, a.ndim()), keepdims, s);
      },
      nb::arg(),
      "axis"_a = nb::none(),
      "keepdims"_a = false,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def max(a: array, /, axis: Union[None, int, Sequence[int]] = None, keepdims: bool = False, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "logsumexp",
      [](const mx::array& a,
         const IntOrVec& axis,
         bool keepdims,
         mx::StreamOrDevice s) {
        return mx::logsumexp(a, get_reduce_axes(axis, a.ndim()), keepdims, s);
      },
      nb::arg(),
      "axis"_a = nb::none(),
      "keepdims"_a = false,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def logsumexp(a: array, /, axis: Union[None, int, Sequence[int]] = None, keepdims: bool = False, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "mean",
      [](const mx::array& a,
         const IntOrVec& axis,
         bool keepdims,
         mx::StreamOrDevice s) {
        return mx::mean(a, get_reduce_axes(axis, a.ndim()), keepdims, s);
      },
      nb::arg(),
      "axis"_a = nb::none(),
      "keepdims"_a = false,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def mean(a: array, /, axis: Union[None, int, Sequence[int]] = None, keepdims: bool = False, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "var",
      [](const mx::array& a,
         const IntOrVec& axis,
         bool keepdims,
         int ddof,
         mx::StreamOrDevice s) {
        return mx::var(a, get_reduce_axes(axis, a.ndim()), keepdims, ddof, s);
      },
      nb::arg(),
      "axis"_a = nb::none(),
      "keepdims"_a = false,
      "ddof"_a = 0,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def var(a: array, /, axis: Union[None, int, Sequence[int]] = None, keepdims: bool = False, ddof: int = 0, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "std",
      [](const mx::array& a,
         const IntOrVec& axis,
         bool keepdims,
         int ddof,
         mx::StreamOrDevice s) {
        return mx::std(a, get_reduce_axes(axis, a.ndim()), keepdims, ddof, s);
      },
      nb::arg(),
      "axis"_a = nb::none(),
      "keepdims"_a = false,
      "ddof"_a = 0,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def std(a: array, /, axis: Union[None, int, Sequence[int]] = None, keepdims: bool = False, ddof: int = 0, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "split",
      [](const mx::array& a,
         const std::variant<int, mx::Shape>& indices_or_sections,
         int axis,
         mx::StreamOrDevice s) {
        if (auto pv = std::get_if<int>(&indices_or_sections); pv) {
          return mx::split(a, *pv, axis, s);
        } else {
          return mx::split(
              a, std::get<mx::Shape>(indices_or_sections), axis, s);
        }
      },
      nb::arg(),
      "indices_or_sections"_a,
      "axis"_a = 0,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def split(a: array, /, indices_or_sections: Union[int, Sequence[int]], axis: int = 0, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "argmin",
      [](const mx::array& a,
         std::optional<int> axis,
         bool keepdims,
         mx::StreamOrDevice s) {
        if (axis) {
          return mx::argmin(a, *axis, keepdims, s);
        } else {
          return mx::argmin(a, keepdims, s);
        }
      },
      nb::arg(),
      "axis"_a = nb::none(),
      "keepdims"_a = false,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def argmin(a: array, /, axis: Union[None, int] = None, keepdims: bool = False, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "argmax",
      [](const mx::array& a,
         std::optional<int> axis,
         bool keepdims,
         mx::StreamOrDevice s) {
        if (axis) {
          return mx::argmax(a, *axis, keepdims, s);
        } else {
          return mx::argmax(a, keepdims, s);
        }
      },
      nb::arg(),
      "axis"_a = nb::none(),
      "keepdims"_a = false,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def argmax(a: array, /, axis: Union[None, int] = None, keepdims: bool = False, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "sort",
      [](const mx::array& a, std::optional<int> axis, mx::StreamOrDevice s) {
        if (axis) {
          return mx::sort(a, *axis, s);
        } else {
          return mx::sort(a, s);
        }
      },
      nb::arg(),
      "axis"_a.none() = -1,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def sort(a: array, /, axis: Union[None, int] = -1, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "argsort",
      [](const mx::array& a, std::optional<int> axis, mx::StreamOrDevice s) {
        if (axis) {
          return mx::argsort(a, *axis, s);
        } else {
          return mx::argsort(a, s);
        }
      },
      nb::arg(),
      "axis"_a.none() = -1,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def argsort(a: array, /, axis: Union[None, int] = -1, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "partition",
      [](const mx::array& a,
         int kth,
         std::optional<int> axis,
         mx::StreamOrDevice s) {
        if (axis) {
          return mx::partition(a, kth, *axis, s);
        } else {
          return mx::partition(a, kth, s);
        }
      },
      nb::arg(),
      "kth"_a,
      "axis"_a.none() = -1,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def partition(a: array, /, kth: int, axis: Union[None, int] = -1, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "argpartition",
      [](const mx::array& a,
         int kth,
         std::optional<int> axis,
         mx::StreamOrDevice s) {
        if (axis) {
          return mx::argpartition(a, kth, *axis, s);
        } else {
          return mx::argpartition(a, kth, s);
        }
      },
      nb::arg(),
      "kth"_a,
      "axis"_a.none() = -1,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def argpartition(a: array, /, kth: int, axis: Union[None, int] = -1, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "topk",
      [](const mx::array& a,
         int k,
         std::optional<int> axis,
         mx::StreamOrDevice s) {
        if (axis) {
          return mx::topk(a, k, *axis, s);
        } else {
          return mx::topk(a, k, s);
        }
      },
      nb::arg(),
      "k"_a,
      "axis"_a.none() = -1,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def topk(a: array, /, k: int, axis: Union[None, int] = -1, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "broadcast_to",
      [](const ScalarOrArray& a, const mx::Shape& shape, mx::StreamOrDevice s) {
        return mx::broadcast_to(to_array(a), shape, s);
      },
      nb::arg(),
      "shape"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def broadcast_to(a: Union[scalar, array], /, shape: Sequence[int], *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "broadcast_arrays",
      [](const nb::args& args, mx::StreamOrDevice s) {
        return broadcast_arrays(nb::cast<std::vector<mx::array>>(args), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def broadcast_arrays(*arrays: array, stream: Union[None, Stream, Device] = None) -> Tuple[array, ...]"),
  m.def(
      "softmax",
      [](const mx::array& a,
         const IntOrVec& axis,
         bool precise,
         mx::StreamOrDevice s) {
        return mx::softmax(a, get_reduce_axes(axis, a.ndim()), precise, s);
      },
      nb::arg(),
      "axis"_a = nb::none(),
      nb::kw_only(),
      "precise"_a = false,
      "stream"_a = nb::none(),
      nb::sig(
          "def softmax(a: array, /, axis: Union[None, int, Sequence[int]] = None, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "concatenate",
      [](const std::vector<mx::array>& arrays,
         std::optional<int> axis,
         mx::StreamOrDevice s) {
        if (axis) {
          return mx::concatenate(arrays, *axis, s);
        } else {
          return mx::concatenate(arrays, s);
        }
      },
      nb::arg(),
      "axis"_a.none() = 0,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def concatenate(arrays: list[array], axis: Optional[int] = 0, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "concat",
      [](const std::vector<mx::array>& arrays,
         std::optional<int> axis,
         mx::StreamOrDevice s) {
        if (axis) {
          return mx::concatenate(arrays, *axis, s);
        } else {
          return mx::concatenate(arrays, s);
        }
      },
      nb::arg(),
      "axis"_a.none() = 0,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def concat(arrays: list[array], axis: Optional[int] = 0, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "stack",
      [](const std::vector<mx::array>& arrays,
         std::optional<int> axis,
         mx::StreamOrDevice s) {
        if (axis.has_value()) {
          return mx::stack(arrays, axis.value(), s);
        } else {
          return mx::stack(arrays, s);
        }
      },
      nb::arg(),
      "axis"_a = 0,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def stack(arrays: list[array], axis: Optional[int] = 0, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "meshgrid",
      [](nb::args arrays_,
         bool sparse,
         std::string indexing,
         mx::StreamOrDevice s) {
        std::vector<mx::array> arrays =
            nb::cast<std::vector<mx::array>>(arrays_);
        return mx::meshgrid(arrays, sparse, indexing, s);
      },
      "arrays"_a,
      "sparse"_a = false,
      "indexing"_a = "xy",
      "stream"_a = nb::none(),
      nb::sig(
          "def meshgrid(*arrays: array, sparse: Optional[bool] = False, indexing: Optional[str] = 'xy', stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "repeat",
      [](const mx::array& array,
         int repeats,
         std::optional<int> axis,
         mx::StreamOrDevice s) {
        if (axis.has_value()) {
          return mx::repeat(array, repeats, axis.value(), s);
        } else {
          return mx::repeat(array, repeats, s);
        }
      },
      nb::arg(),
      "repeats"_a,
      "axis"_a = nb::none(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def repeat(array: array, repeats: int, axis: Optional[int] = None, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "clip",
      [](const mx::array& a,
         const std::optional<ScalarOrArray>& min,
         const std::optional<ScalarOrArray>& max,
         mx::StreamOrDevice s) {
        std::optional<mx::array> min_ = std::nullopt;
        std::optional<mx::array> max_ = std::nullopt;
        if (min) {
          min_ = to_arrays(a, min.value()).second;
        }
        if (max) {
          max_ = to_arrays(a, max.value()).second;
        }
        return mx::clip(a, min_, max_, s);
      },
      nb::arg(),
      "a_min"_a.none(),
      "a_max"_a.none(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def clip(a: array, /, a_min: Union[scalar, array, None], a_max: Union[scalar, array, None], *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "pad",
      [](const mx::array& a,
         const std::variant<
             int,
             std::tuple<int>,
             std::pair<int, int>,
             std::vector<std::pair<int, int>>>& pad_width,
         const std::string mode,
         const ScalarOrArray& constant_value,
         mx::StreamOrDevice s) {
        if (auto pv = std::get_if<int>(&pad_width); pv) {
          return mx::pad(a, *pv, to_array(constant_value), mode, s);
        } else if (auto pv = std::get_if<std::tuple<int>>(&pad_width); pv) {
          return mx::pad(
              a, std::get<0>(*pv), to_array(constant_value), mode, s);
        } else if (auto pv = std::get_if<std::pair<int, int>>(&pad_width); pv) {
          return mx::pad(a, *pv, to_array(constant_value), mode, s);
        } else {
          auto v = std::get<std::vector<std::pair<int, int>>>(pad_width);
          if (v.size() == 1) {
            return mx::pad(a, v[0], to_array(constant_value), mode, s);
          } else {
            return mx::pad(a, v, to_array(constant_value), mode, s);
          }
        }
      },
      nb::arg(),
      "pad_width"_a,
      "mode"_a = "constant",
      "constant_values"_a = 0,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def pad(a: array, pad_width: Union[int, tuple[int], tuple[int, int], list[tuple[int, int]]], mode: Literal['constant', 'edge'] = 'constant', constant_values: Union[scalar, array] = 0, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "as_strided",
      [](const mx::array& a,
         std::optional<mx::Shape> shape,
         std::optional<mx::Strides> strides,
         size_t offset,
         mx::StreamOrDevice s) {
        auto a_shape = (shape) ? *shape : a.shape();
        mx::Strides a_strides;
        if (strides) {
          a_strides = *strides;
        } else {
          a_strides = mx::Strides(a_shape.size(), 1);
          for (int i = a_shape.size() - 1; i > 0; i--) {
            a_strides[i - 1] = a_shape[i] * a_strides[i];
          }
        }
        return mx::as_strided(a, a_shape, a_strides, offset, s);
      },
      nb::arg(),
      "shape"_a = nb::none(),
      "strides"_a = nb::none(),
      "offset"_a = 0,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def as_strided(a: array, /, shape: Optional[Sequence[int]] = None, strides: Optional[Sequence[int]] = None, offset: int = 0, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "cumsum",
      [](const mx::array& a,
         std::optional<int> axis,
         bool reverse,
         bool inclusive,
         mx::StreamOrDevice s) {
        if (axis) {
          return mx::cumsum(a, *axis, reverse, inclusive, s);
        } else {
          return mx::cumsum(mx::reshape(a, {-1}, s), 0, reverse, inclusive, s);
        }
      },
      nb::arg(),
      "axis"_a = nb::none(),
      nb::kw_only(),
      "reverse"_a = false,
      "inclusive"_a = true,
      "stream"_a = nb::none(),
      nb::sig(
          "def cumsum(a: array, /, axis: Optional[int] = None, *, reverse: bool = False, inclusive: bool = True, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "cumprod",
      [](const mx::array& a,
         std::optional<int> axis,
         bool reverse,
         bool inclusive,
         mx::StreamOrDevice s) {
        if (axis) {
          return mx::cumprod(a, *axis, reverse, inclusive, s);
        } else {
          return mx::cumprod(mx::reshape(a, {-1}, s), 0, reverse, inclusive, s);
        }
      },
      nb::arg(),
      "axis"_a = nb::none(),
      nb::kw_only(),
      "reverse"_a = false,
      "inclusive"_a = true,
      "stream"_a = nb::none(),
      nb::sig(
          "def cumprod(a: array, /, axis: Optional[int] = None, *, reverse: bool = False, inclusive: bool = True, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "cummax",
      [](const mx::array& a,
         std::optional<int> axis,
         bool reverse,
         bool inclusive,
         mx::StreamOrDevice s) {
        if (axis) {
          return mx::cummax(a, *axis, reverse, inclusive, s);
        } else {
          return mx::cummax(mx::reshape(a, {-1}, s), 0, reverse, inclusive, s);
        }
      },
      nb::arg(),
      "axis"_a = nb::none(),
      nb::kw_only(),
      "reverse"_a = false,
      "inclusive"_a = true,
      "stream"_a = nb::none(),
      nb::sig(
          "def cummax(a: array, /, axis: Optional[int] = None, *, reverse: bool = False, inclusive: bool = True, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "cummin",
      [](const mx::array& a,
         std::optional<int> axis,
         bool reverse,
         bool inclusive,
         mx::StreamOrDevice s) {
        if (axis) {
          return mx::cummin(a, *axis, reverse, inclusive, s);
        } else {
          return mx::cummin(mx::reshape(a, {-1}, s), 0, reverse, inclusive, s);
        }
      },
      nb::arg(),
      "axis"_a = nb::none(),
      nb::kw_only(),
      "reverse"_a = false,
      "inclusive"_a = true,
      "stream"_a = nb::none(),
      nb::sig(
          "def cummin(a: array, /, axis: Optional[int] = None, *, reverse: bool = False, inclusive: bool = True, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "conj",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::conjugate(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def conj(a: array, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "conjugate",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::conjugate(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def conjugate(a: array, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "convolve",
      [](const mx::array& a,
         const mx::array& v,
         const std::string& mode,
         mx::StreamOrDevice s) {
        if (a.ndim() != 1 || v.ndim() != 1) {
          throw std::invalid_argument("[convolve] Inputs must be 1D.");
        }

        if (a.size() == 0 || v.size() == 0) {
          throw std::invalid_argument("[convolve] Inputs cannot be empty.");
        }

        mx::array in = a.size() < v.size() ? v : a;
        mx::array wt = a.size() < v.size() ? a : v;
        wt = mx::slice(wt, {wt.shape(0) - 1}, {-wt.shape(0) - 1}, {-1}, s);

        in = mx::reshape(in, {1, -1, 1}, s);
        wt = mx::reshape(wt, {1, -1, 1}, s);

        int padding = 0;

        if (mode == "full") {
          padding = wt.size() - 1;
        } else if (mode == "valid") {
          padding = 0;
        } else if (mode == "same") {
          // Odd sizes use symmetric padding
          if (wt.size() % 2) {
            padding = wt.size() / 2;
          } else { // Even sizes use asymmetric padding
            int pad_l = wt.size() / 2;
            int pad_r = std::max(0, pad_l - 1);
            in = mx::pad(
                in,
                {{0, 0}, {pad_l, pad_r}, {0, 0}},
                mx::array(0),
                "constant",
                s);
          }

        } else {
          throw std::invalid_argument("[convolve] Invalid mode.");
        }

        mx::array out = mx::conv1d(
            in,
            wt,
            /*stride = */ 1,
            /*padding = */ padding,
            /*dilation = */ 1,
            /*groups = */ 1,
            s);

        return mx::reshape(out, {-1}, s);
      },
      nb::arg(),
      nb::arg(),
      "mode"_a = "full",
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          R"(def convolve(a: array, v: array, /, mode: str = "full", *, stream: Union[None, Stream, Device] = None) -> array)"),
  m.def(
      "conv1d",
      &mx::conv1d,
      nb::arg(),
      nb::arg(),
      "stride"_a = 1,
      "padding"_a = 0,
      "dilation"_a = 1,
      "groups"_a = 1,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def conv1d(input: array, weight: array, /, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "conv2d",
      [](const mx::array& input,
         const mx::array& weight,
         const std::variant<int, std::pair<int, int>>& stride,
         const std::variant<int, std::pair<int, int>>& padding,
         const std::variant<int, std::pair<int, int>>& dilation,
         int groups,
         mx::StreamOrDevice s) {
        std::pair<int, int> stride_pair{1, 1};
        std::pair<int, int> padding_pair{0, 0};
        std::pair<int, int> dilation_pair{1, 1};

        if (auto pv = std::get_if<int>(&stride); pv) {
          stride_pair = std::pair<int, int>{*pv, *pv};
        } else {
          stride_pair = std::get<std::pair<int, int>>(stride);
        }

        if (auto pv = std::get_if<int>(&padding); pv) {
          padding_pair = std::pair<int, int>{*pv, *pv};
        } else {
          padding_pair = std::get<std::pair<int, int>>(padding);
        }

        if (auto pv = std::get_if<int>(&dilation); pv) {
          dilation_pair = std::pair<int, int>{*pv, *pv};
        } else {
          dilation_pair = std::get<std::pair<int, int>>(dilation);
        }

        return mx::conv2d(
            input, weight, stride_pair, padding_pair, dilation_pair, groups, s);
      },
      nb::arg(),
      nb::arg(),
      "stride"_a = 1,
      "padding"_a = 0,
      "dilation"_a = 1,
      "groups"_a = 1,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def conv2d(input: array, weight: array, /, stride: Union[int, tuple[int, int]] = 1, padding: Union[int, tuple[int, int]] = 0, dilation: Union[int, tuple[int, int]] = 1, groups: int = 1, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "conv3d",
      [](const mx::array& input,
         const mx::array& weight,
         const std::variant<int, std::tuple<int, int, int>>& stride,
         const std::variant<int, std::tuple<int, int, int>>& padding,
         const std::variant<int, std::tuple<int, int, int>>& dilation,
         int groups,
         mx::StreamOrDevice s) {
        std::tuple<int, int, int> stride_tuple{1, 1, 1};
        std::tuple<int, int, int> padding_tuple{0, 0, 0};
        std::tuple<int, int, int> dilation_tuple{1, 1, 1};

        if (auto pv = std::get_if<int>(&stride); pv) {
          stride_tuple = std::tuple<int, int, int>{*pv, *pv, *pv};
        } else {
          stride_tuple = std::get<std::tuple<int, int, int>>(stride);
        }

        if (auto pv = std::get_if<int>(&padding); pv) {
          padding_tuple = std::tuple<int, int, int>{*pv, *pv, *pv};
        } else {
          padding_tuple = std::get<std::tuple<int, int, int>>(padding);
        }

        if (auto pv = std::get_if<int>(&dilation); pv) {
          dilation_tuple = std::tuple<int, int, int>{*pv, *pv, *pv};
        } else {
          dilation_tuple = std::get<std::tuple<int, int, int>>(dilation);
        }

        return mx::conv3d(
            input,
            weight,
            stride_tuple,
            padding_tuple,
            dilation_tuple,
            groups,
            s);
      },
      nb::arg(),
      nb::arg(),
      "stride"_a = 1,
      "padding"_a = 0,
      "dilation"_a = 1,
      "groups"_a = 1,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def conv3d(input: array, weight: array, /, stride: Union[int, tuple[int, int, int]] = 1, padding: Union[int, tuple[int, int, int]] = 0, dilation: Union[int, tuple[int, int, int]] = 1, groups: int = 1, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "conv_transpose1d",
      &mx::conv_transpose1d,
      nb::arg(),
      nb::arg(),
      "stride"_a = 1,
      "padding"_a = 0,
      "dilation"_a = 1,
      "groups"_a = 1,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def conv_transpose1d(input: array, weight: array, /, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "conv_transpose2d",
      [](const mx::array& input,
         const mx::array& weight,
         const std::variant<int, std::pair<int, int>>& stride,
         const std::variant<int, std::pair<int, int>>& padding,
         const std::variant<int, std::pair<int, int>>& dilation,
         int groups,
         mx::StreamOrDevice s) {
        std::pair<int, int> stride_pair{1, 1};
        std::pair<int, int> padding_pair{0, 0};
        std::pair<int, int> dilation_pair{1, 1};

        if (auto pv = std::get_if<int>(&stride); pv) {
          stride_pair = std::pair<int, int>{*pv, *pv};
        } else {
          stride_pair = std::get<std::pair<int, int>>(stride);
        }

        if (auto pv = std::get_if<int>(&padding); pv) {
          padding_pair = std::pair<int, int>{*pv, *pv};
        } else {
          padding_pair = std::get<std::pair<int, int>>(padding);
        }

        if (auto pv = std::get_if<int>(&dilation); pv) {
          dilation_pair = std::pair<int, int>{*pv, *pv};
        } else {
          dilation_pair = std::get<std::pair<int, int>>(dilation);
        }

        return mx::conv_transpose2d(
            input, weight, stride_pair, padding_pair, dilation_pair, groups, s);
      },
      nb::arg(),
      nb::arg(),
      "stride"_a = 1,
      "padding"_a = 0,
      "dilation"_a = 1,
      "groups"_a = 1,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def conv_transpose2d(input: array, weight: array, /, stride: Union[int, Tuple[int, int]] = 1, padding: Union[int, Tuple[int, int]] = 0, dilation: Union[int, Tuple[int, int]] = 1, groups: int = 1, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "conv_transpose3d",
      [](const mx::array& input,
         const mx::array& weight,
         const std::variant<int, std::tuple<int, int, int>>& stride,
         const std::variant<int, std::tuple<int, int, int>>& padding,
         const std::variant<int, std::tuple<int, int, int>>& dilation,
         int groups,
         mx::StreamOrDevice s) {
        std::tuple<int, int, int> stride_tuple{1, 1, 1};
        std::tuple<int, int, int> padding_tuple{0, 0, 0};
        std::tuple<int, int, int> dilation_tuple{1, 1, 1};

        if (auto pv = std::get_if<int>(&stride); pv) {
          stride_tuple = std::tuple<int, int, int>{*pv, *pv, *pv};
        } else {
          stride_tuple = std::get<std::tuple<int, int, int>>(stride);
        }

        if (auto pv = std::get_if<int>(&padding); pv) {
          padding_tuple = std::tuple<int, int, int>{*pv, *pv, *pv};
        } else {
          padding_tuple = std::get<std::tuple<int, int, int>>(padding);
        }

        if (auto pv = std::get_if<int>(&dilation); pv) {
          dilation_tuple = std::tuple<int, int, int>{*pv, *pv, *pv};
        } else {
          dilation_tuple = std::get<std::tuple<int, int, int>>(dilation);
        }

        return mx::conv_transpose3d(
            input,
            weight,
            stride_tuple,
            padding_tuple,
            dilation_tuple,
            groups,
            s);
      },
      nb::arg(),
      nb::arg(),
      "stride"_a = 1,
      "padding"_a = 0,
      "dilation"_a = 1,
      "groups"_a = 1,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def conv_transpose3d(input: array, weight: array, /, stride: Union[int, Tuple[int, int, int]] = 1, padding: Union[int, Tuple[int, int, int]] = 0, dilation: Union[int, Tuple[int, int, int]] = 1, groups: int = 1, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "conv_general",
      [](const mx::array& input,
         const mx::array& weight,
         const std::variant<int, std::vector<int>>& stride,
         const std::variant<
             int,
             std::vector<int>,
             std::pair<std::vector<int>, std::vector<int>>>& padding,
         const std::variant<int, std::vector<int>>& kernel_dilation,
         const std::variant<int, std::vector<int>>& input_dilation,
         int groups,
         bool flip,
         mx::StreamOrDevice s) {
        std::vector<int> stride_vec;
        std::vector<int> padding_lo_vec;
        std::vector<int> padding_hi_vec;
        std::vector<int> kernel_dilation_vec;
        std::vector<int> input_dilation_vec;

        if (auto pv = std::get_if<int>(&stride); pv) {
          stride_vec.push_back(*pv);
        } else {
          stride_vec = std::get<std::vector<int>>(stride);
        }

        if (auto pv = std::get_if<int>(&padding); pv) {
          padding_lo_vec.push_back(*pv);
          padding_hi_vec.push_back(*pv);
        } else if (auto pv = std::get_if<std::vector<int>>(&padding); pv) {
          padding_lo_vec = *pv;
          padding_hi_vec = *pv;
        } else {
          auto [pl, ph] =
              std::get<std::pair<std::vector<int>, std::vector<int>>>(padding);
          padding_lo_vec = pl;
          padding_hi_vec = ph;
        }

        if (auto pv = std::get_if<int>(&kernel_dilation); pv) {
          kernel_dilation_vec.push_back(*pv);
        } else {
          kernel_dilation_vec = std::get<std::vector<int>>(kernel_dilation);
        }

        if (auto pv = std::get_if<int>(&input_dilation); pv) {
          input_dilation_vec.push_back(*pv);
        } else {
          input_dilation_vec = std::get<std::vector<int>>(input_dilation);
        }

        return mx::conv_general(
            /* array input = */ std::move(input),
            /* array weight = */ std::move(weight),
            /* std::vector<int> stride = */ std::move(stride_vec),
            /* std::vector<int> padding_lo = */ std::move(padding_lo_vec),
            /* std::vector<int> padding_hi = */ std::move(padding_hi_vec),
            /* std::vector<int> kernel_dilation = */
            std::move(kernel_dilation_vec),
            /* std::vector<int> input_dilation = */
            std::move(input_dilation_vec),
            /* int groups = */ groups,
            /* bool flip = */ flip,
            s);
      },
      nb::arg(),
      nb::arg(),
      "stride"_a = 1,
      "padding"_a = 0,
      "kernel_dilation"_a = 1,
      "input_dilation"_a = 1,
      "groups"_a = 1,
      "flip"_a = false,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def conv_general(input: array, weight: array, /, stride: Union[int, Sequence[int]] = 1, padding: Union[int, Sequence[int], tuple[Sequence[int], Sequence[int]]] = 0, kernel_dilation: Union[int, Sequence[int]] = 1, input_dilation: Union[int, Sequence[int]] = 1, groups: int = 1, flip: bool = False, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "save",
      &mlx_save_helper,
      "file"_a,
      "arr"_a,
      nb::sig("def save(file: str, arr: array) -> None"),
  m.def(
      "savez",
      [](nb::object file, nb::args args, const nb::kwargs& kwargs) {
        mlx_savez_helper(file, args, kwargs, /* compressed= */ false);
      },
      "file"_a,
      "args"_a,
      "kwargs"_a,
  m.def(
      "savez_compressed",
      [](nb::object file, nb::args args, const nb::kwargs& kwargs) {
        mlx_savez_helper(file, args, kwargs, /*compressed=*/true);
      },
      nb::arg(),
      "args"_a,
      "kwargs"_a,
      nb::sig("def savez_compressed(file: str, *args, **kwargs)"),
  m.def(
      "load",
      &mlx_load_helper,
      nb::arg(),
      "format"_a = nb::none(),
      "return_metadata"_a = false,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def load(file: str, /, format: Optional[str] = None, return_metadata: bool = False, *, stream: Union[None, Stream, Device] = None) -> Union[array, dict[str, array]]"),
  m.def(
      "save_safetensors",
      &mlx_save_safetensor_helper,
      "file"_a,
      "arrays"_a,
      "metadata"_a = nb::none(),
      nb::sig(
          "def save_safetensors(file: str, arrays: dict[str, array], metadata: Optional[dict[str, str]] = None)"),
  m.def(
      "save_gguf",
      &mlx_save_gguf_helper,
      "file"_a,
      "arrays"_a,
      "metadata"_a = nb::none(),
      nb::sig(
          "def save_gguf(file: str, arrays: dict[str, array], metadata: dict[str, Union[array, str, list[str]]])"),
  m.def(
      "where",
      [](const ScalarOrArray& condition,
         const ScalarOrArray& x_,
         const ScalarOrArray& y_,
         mx::StreamOrDevice s) {
        auto [x, y] = to_arrays(x_, y_);
        return mx::where(to_array(condition), x, y, s);
      },
      "condition"_a,
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def where(condition: Union[scalar, array], x: Union[scalar, array], y: Union[scalar, array], /, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "nan_to_num",
      [](const ScalarOrArray& a,
         float nan,
         std::optional<float>& posinf,
         std::optional<float>& neginf,
         mx::StreamOrDevice s) {
        return mx::nan_to_num(to_array(a), nan, posinf, neginf, s);
      },
      nb::arg(),
      "nan"_a = 0.0f,
      "posinf"_a = nb::none(),
      "neginf"_a = nb::none(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def nan_to_num(a: Union[scalar, array], nan: float = 0, posinf: Optional[float] = None, neginf: Optional[float] = None, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "round",
      [](const ScalarOrArray& a, int decimals, mx::StreamOrDevice s) {
        return mx::round(to_array(a), decimals, s);
      },
      nb::arg(),
      "decimals"_a = 0,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def round(a: array, /, decimals: int = 0, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "quantized_matmul",
      &mx::quantized_matmul,
      nb::arg(),
      nb::arg(),
      "scales"_a,
      "biases"_a,
      "transpose"_a = true,
      "group_size"_a = 64,
      "bits"_a = 4,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def quantized_matmul(x: array, w: array, /, scales: array, biases: array, transpose: bool = True, group_size: int = 64, bits: int = 4, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "quantize",
      &mx::quantize,
      nb::arg(),
      "group_size"_a = 64,
      "bits"_a = 4,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def quantize(w: array, /, group_size: int = 64, bits : int = 4, *, stream: Union[None, Stream, Device] = None) -> tuple[array, array, array]"),
  m.def(
      "dequantize",
      &mx::dequantize,
      nb::arg(),
      "scales"_a,
      "biases"_a,
      "group_size"_a = 64,
      "bits"_a = 4,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def dequantize(w: array, /, scales: array, biases: array, group_size: int = 64, bits: int = 4, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "gather_qmm",
      &mx::gather_qmm,
      nb::arg(),
      nb::arg(),
      "scales"_a,
      "biases"_a,
      "lhs_indices"_a = nb::none(),
      "rhs_indices"_a = nb::none(),
      "transpose"_a = true,
      "group_size"_a = 64,
      "bits"_a = 4,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def gather_qmm(x: array, w: array, /, scales: array, biases: array, lhs_indices: Optional[array] = None, rhs_indices: Optional[array] = None, transpose: bool = True, group_size: int = 64, bits: int = 4, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "tensordot",
      [](const mx::array& a,
         const mx::array& b,
         const std::variant<int, std::vector<std::vector<int>>>& axes,
         mx::StreamOrDevice s) {
        if (auto pv = std::get_if<int>(&axes); pv) {
          return mx::tensordot(a, b, *pv, s);
        } else {
          auto& x = std::get<std::vector<std::vector<int>>>(axes);
          if (x.size() != 2) {
            throw std::invalid_argument(
                "[tensordot] axes must be a list of two lists.");
          }
          return mx::tensordot(a, b, x[0], x[1], s);
        }
      },
      nb::arg(),
      nb::arg(),
      "axes"_a = 2,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def tensordot(a: array, b: array, /, axes: Union[int, list[Sequence[int]]] = 2, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "inner",
      &mx::inner,
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def inner(a: array, b: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "outer",
      &mx::outer,
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def outer(a: array, b: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "tile",
      [](const mx::array& a,
         const std::variant<int, std::vector<int>>& reps,
         mx::StreamOrDevice s) {
        if (auto pv = std::get_if<int>(&reps); pv) {
          return mx::tile(a, {*pv}, s);
        } else {
          return mx::tile(a, std::get<std::vector<int>>(reps), s);
        }
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def tile(a: array, reps: Union[int, Sequence[int]], /, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "addmm",
      &mx::addmm,
      nb::arg(),
      nb::arg(),
      nb::arg(),
      "alpha"_a = 1.0f,
      "beta"_a = 1.0f,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def addmm(c: array, a: array, b: array, /, alpha: float = 1.0, beta: float = 1.0,  *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "block_masked_mm",
      &mx::block_masked_mm,
      nb::arg(),
      nb::arg(),
      "block_size"_a = 64,
      "mask_out"_a = nb::none(),
      "mask_lhs"_a = nb::none(),
      "mask_rhs"_a = nb::none(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def block_masked_mm(a: array, b: array, /, block_size: int = 64, mask_out: Optional[array] = None, mask_lhs: Optional[array] = None, mask_rhs: Optional[array] = None, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "gather_mm",
      &mx::gather_mm,
      nb::arg(),
      nb::arg(),
      "lhs_indices"_a = nb::none(),
      "rhs_indices"_a = nb::none(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def gather_mm(a: array, b: array, /, lhs_indices: array, rhs_indices: array, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "diagonal",
      &mx::diagonal,
      "a"_a,
      "offset"_a = 0,
      "axis1"_a = 0,
      "axis2"_a = 1,
      "stream"_a = nb::none(),
      nb::sig(
          "def diagonal(a: array, offset: int = 0, axis1: int = 0, axis2: int = 1, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "diag",
      &mx::diag,
      nb::arg(),
      "k"_a = 0,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def diag(a: array, /, k: int = 0, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "trace",
      [](const mx::array& a,
         int offset,
         int axis1,
         int axis2,
         std::optional<mx::Dtype> dtype,
         mx::StreamOrDevice s) {
        if (!dtype.has_value()) {
          return mx::trace(a, offset, axis1, axis2, s);
        }
        return mx::trace(a, offset, axis1, axis2, dtype.value(), s);
      },
      nb::arg(),
      "offset"_a = 0,
      "axis1"_a = 0,
      "axis2"_a = 1,
      "dtype"_a = nb::none(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def trace(a: array, /, offset: int = 0, axis1: int = 0, axis2: int = 1, dtype: Optional[Dtype] = None, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "atleast_1d",
      [](const nb::args& arys, mx::StreamOrDevice s) -> nb::object {
        if (arys.size() == 1) {
          return nb::cast(mx::atleast_1d(nb::cast<mx::array>(arys[0]), s));
        }
        return nb::cast(
            mx::atleast_1d(nb::cast<std::vector<mx::array>>(arys), s));
      },
      "arys"_a,
      "stream"_a = nb::none(),
      nb::sig(
          "def atleast_1d(*arys: array, stream: Union[None, Stream, Device] = None) -> Union[array, list[array]]"),
  m.def(
      "atleast_2d",
      [](const nb::args& arys, mx::StreamOrDevice s) -> nb::object {
        if (arys.size() == 1) {
          return nb::cast(mx::atleast_2d(nb::cast<mx::array>(arys[0]), s));
        }
        return nb::cast(
            mx::atleast_2d(nb::cast<std::vector<mx::array>>(arys), s));
      },
      "arys"_a,
      "stream"_a = nb::none(),
      nb::sig(
          "def atleast_2d(*arys: array, stream: Union[None, Stream, Device] = None) -> Union[array, list[array]]"),
  m.def(
      "atleast_3d",
      [](const nb::args& arys, mx::StreamOrDevice s) -> nb::object {
        if (arys.size() == 1) {
          return nb::cast(mx::atleast_3d(nb::cast<mx::array>(arys[0]), s));
        }
        return nb::cast(
            mx::atleast_3d(nb::cast<std::vector<mx::array>>(arys), s));
      },
      "arys"_a,
      "stream"_a = nb::none(),
      nb::sig(
          "def atleast_3d(*arys: array, stream: Union[None, Stream, Device] = None) -> Union[array, list[array]]"),
  m.def(
      "issubdtype",
      [](const nb::object& d1, const nb::object& d2) {
        auto dispatch_second = [](const auto& t1, const auto& d2) {
          if (nb::isinstance<mx::Dtype>(d2)) {
            return mx::issubdtype(t1, nb::cast<mx::Dtype>(d2));
          } else if (nb::isinstance<mx::Dtype::Category>(d2)) {
            return mx::issubdtype(t1, nb::cast<mx::Dtype::Category>(d2));
          } else {
            throw std::invalid_argument(
                "[issubdtype] Received invalid type for second input.");
          }
        };
        if (nb::isinstance<mx::Dtype>(d1)) {
          return dispatch_second(nb::cast<mx::Dtype>(d1), d2);
        } else if (nb::isinstance<mx::Dtype::Category>(d1)) {
          return dispatch_second(nb::cast<mx::Dtype::Category>(d1), d2);
        } else {
          throw std::invalid_argument(
              "[issubdtype] Received invalid type for first input.");
        }
      },
      ""_a,
      ""_a,
      nb::sig(
          "def issubdtype(arg1: Union[Dtype, DtypeCategory], arg2: Union[Dtype, DtypeCategory]) -> bool"),
  m.def(
      "bitwise_and",
      [](const ScalarOrArray& a_,
         const ScalarOrArray& b_,
         mx::StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return mx::bitwise_and(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def bitwise_and(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "bitwise_or",
      [](const ScalarOrArray& a_,
         const ScalarOrArray& b_,
         mx::StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return mx::bitwise_or(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def bitwise_or(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "bitwise_xor",
      [](const ScalarOrArray& a_,
         const ScalarOrArray& b_,
         mx::StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return mx::bitwise_xor(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def bitwise_xor(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "left_shift",
      [](const ScalarOrArray& a_,
         const ScalarOrArray& b_,
         mx::StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return mx::left_shift(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def left_shift(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "right_shift",
      [](const ScalarOrArray& a_,
         const ScalarOrArray& b_,
         mx::StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return mx::right_shift(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def right_shift(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "bitwise_invert",
      [](const ScalarOrArray& a_, mx::StreamOrDevice s) {
        auto a = to_array(a_);
        return mx::bitwise_invert(a, s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def bitwise_invert(a: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "view",
      [](const ScalarOrArray& a, const mx::Dtype& dtype, mx::StreamOrDevice s) {
        return mx::view(to_array(a), dtype, s);
      },
      nb::arg(),
      "dtype"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def view(a: Union[scalar, array], dtype: Dtype, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "hadamard_transform",
      &mx::hadamard_transform,
      nb::arg(),
      "scale"_a = nb::none(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def hadamard_transform(a: array, scale: Optional[float] = None, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "einsum_path",
      [](const std::string& equation, const nb::args& operands) {
        auto arrays_list = nb::cast<std::vector<mx::array>>(operands);
        auto [path, str] = mx::einsum_path(equation, arrays_list);
        // Convert to list of tuples
        std::vector<nb::tuple> tuple_path;
        for (auto& p : path) {
          tuple_path.push_back(nb::tuple(nb::cast(p)));
        }
        return std::make_pair(tuple_path, str);
      },
      "subscripts"_a,
      "operands"_a,
      nb::sig("def einsum_path(subscripts: str, *operands)"),
  m.def(
      "einsum",
      [](const std::string& subscripts,
         const nb::args& operands,
         mx::StreamOrDevice s) {
        auto arrays_list = nb::cast<std::vector<mx::array>>(operands);
        return mx::einsum(subscripts, arrays_list, s);
      },
      "subscripts"_a,
      "operands"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def einsum(subscripts: str, *operands, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "roll",
      [](const mx::array& a,
         const std::variant<int, mx::Shape>& shift,
         const IntOrVec& axis,
         mx::StreamOrDevice s) {
        return std::visit(
            [&](auto sh, auto ax) -> mx::array {
              if constexpr (std::is_same_v<decltype(ax), std::monostate>) {
                return mx::roll(a, sh, s);
              } else {
                return mx::roll(a, sh, ax, s);
              }
            },
            shift,
            axis);
      },
      nb::arg(),
      "shift"_a,
      "axis"_a = nb::none(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def roll(a: array, shift: Union[int, Tuple[int]], axis: Union[None, int, Tuple[int]] = None, /, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "real",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::real(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def real(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "imag",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::imag(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def imag(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "slice",
      [](const mx::array& a,
         const mx::array& start_indices,
         std::vector<int> axes,
         mx::Shape slice_size,
         mx::StreamOrDevice s) {
        return mx::slice(
            a, start_indices, std::move(axes), std::move(slice_size), s);
      },
      nb::arg(),
      "start_indices"_a,
      "axes"_a,
      "slice_size"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def slice(a: array, start_indices: array, axes: Sequence[int], slice_size: Sequence[int], *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "slice_update",
      [](const mx::array& src,
         const mx::array& update,
         const mx::array& start_indices,
         std::vector<int> axes,
         mx::StreamOrDevice s) {
        return mx::slice_update(src, update, start_indices, axes, s);
      },
      nb::arg(),
      "update"_a,
      "start_indices"_a,
      "axes"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def slice_update(a: array, update: array, start_indices: array, axes: Sequence[int], *, stream: Union[None, Stream, Device] = None) -> array"),
  m.def(
      "contiguous",
      &mx::contiguous,
      nb::arg(),
      "allow_col_major"_a = false,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def contiguous(a: array, /, allow_col_major: bool = False, *, stream: Union[None, Stream, Device] = None) -> array"),
}
