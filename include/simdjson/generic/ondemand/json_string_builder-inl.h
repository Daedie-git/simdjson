/**
 * This file is part of the builder API. It is temporarily in the ondemand
 * directory but we will move it to a builder directory later.
 */
#include <array>
#include <cstring>
#include <type_traits>
#ifndef SIMDJSON_GENERIC_STRING_BUILDER_INL_H

#ifndef SIMDJSON_CONDITIONAL_INCLUDE
#define SIMDJSON_GENERIC_STRING_BUILDER_INL_H
#include "simdjson/generic/builder/json_string_builder.h"
#endif // SIMDJSON_CONDITIONAL_INCLUDE

/*
 * Empirically, we have found that an inlined optimization is important for
 * performance. The following macros are not ideal. We should find a better
 * way to inline the code.
 */

#if defined(__SSE2__) || defined(__x86_64__) || defined(__x86_64) ||           \
    (defined(_M_AMD64) || defined(_M_X64) ||                                   \
     (defined(_M_IX86_FP) && _M_IX86_FP == 2))
#ifndef SIMDJSON_EXPERIMENTAL_HAS_SSE2
#define SIMDJSON_EXPERIMENTAL_HAS_SSE2 1
#endif
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
#ifndef SIMDJSON_EXPERIMENTAL_HAS_NEON
#define SIMDJSON_EXPERIMENTAL_HAS_NEON 1
#endif
#endif
#if SIMDJSON_EXPERIMENTAL_HAS_NEON
#include <arm_neon.h>
#endif
#if SIMDJSON_EXPERIMENTAL_HAS_SSE2
#include <emmintrin.h>
#endif

namespace simdjson {
namespace SIMDJSON_IMPLEMENTATION {
namespace builder {

// Storage policy implementations (see json_string_builder.h)
simdjson_inline owning_storage::owning_storage(size_t initial_capacity) noexcept {
  if (initial_capacity == 0) { return; }
  std::unique_ptr<char[]> ptr(new (std::nothrow) char[initial_capacity]);
  if (!ptr) { cap = 0; buffer.reset(); return; }
  cap = initial_capacity;
  buffer.swap(ptr);
}

simdjson_inline bool owning_storage::grow(size_t desired_capacity, size_t bytes_to_copy) noexcept {
  std::unique_ptr<char[]> new_buffer(new (std::nothrow) char[desired_capacity]);
  if (!new_buffer) { return false; }
  if (buffer && bytes_to_copy) {
    std::memcpy(new_buffer.get(), buffer.get(), bytes_to_copy);
  }
  buffer.swap(new_buffer);
  cap = desired_capacity;
  return true;
}

static SIMDJSON_CONSTEXPR_LAMBDA std::array<uint8_t, 256>
    json_quotable_character = {
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

/**

A possible SWAR implementation of has_json_escapable_byte. It is not used
because it is slower than the current implementation. It is kept here for
reference (to show that we tried it).

inline bool has_json_escapable_byte(uint64_t x) {
  uint64_t is_ascii = 0x8080808080808080ULL & ~x;
  uint64_t xor2 = x ^ 0x0202020202020202ULL;
  uint64_t lt32_or_eq34 = xor2 - 0x2121212121212121ULL;
  uint64_t sub92 = x ^ 0x5C5C5C5C5C5C5C5CULL;
  uint64_t eq92 = (sub92 - 0x0101010101010101ULL);
  return ((lt32_or_eq34 | eq92) & is_ascii) != 0;
}

**/

SIMDJSON_CONSTEXPR_LAMBDA simdjson_inline bool
simple_needs_escaping(std::string_view v) {
  for (char c : v) {
    // a table lookup is faster than a series of comparisons
    if (json_quotable_character[static_cast<uint8_t>(c)]) {
      return true;
    }
  }
  return false;
}

#if SIMDJSON_EXPERIMENTAL_HAS_NEON
simdjson_inline bool fast_needs_escaping(std::string_view view) {
  if (view.size() < 16) {
    return simple_needs_escaping(view);
  }
  size_t i = 0;
  uint8x16_t running = vdupq_n_u8(0);
  uint8x16_t v34 = vdupq_n_u8(34);
  uint8x16_t v92 = vdupq_n_u8(92);

  for (; i + 15 < view.size(); i += 16) {
    uint8x16_t word = vld1q_u8((const uint8_t *)view.data() + i);
    running = vorrq_u8(running, vceqq_u8(word, v34));
    running = vorrq_u8(running, vceqq_u8(word, v92));
    running = vorrq_u8(running, vcltq_u8(word, vdupq_n_u8(32)));
  }
  if (i < view.size()) {
    uint8x16_t word =
        vld1q_u8((const uint8_t *)view.data() + view.length() - 16);
    running = vorrq_u8(running, vceqq_u8(word, v34));
    running = vorrq_u8(running, vceqq_u8(word, v92));
    running = vorrq_u8(running, vcltq_u8(word, vdupq_n_u8(32)));
  }
  return vmaxvq_u32(vreinterpretq_u32_u8(running)) != 0;
}
#elif SIMDJSON_EXPERIMENTAL_HAS_SSE2
simdjson_inline bool fast_needs_escaping(std::string_view view) {
  if (view.size() < 16) {
    return simple_needs_escaping(view);
  }
  size_t i = 0;
  __m128i running = _mm_setzero_si128();
  for (; i + 15 < view.size(); i += 16) {

    __m128i word =
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(view.data() + i));
    running = _mm_or_si128(running, _mm_cmpeq_epi8(word, _mm_set1_epi8(34)));
    running = _mm_or_si128(running, _mm_cmpeq_epi8(word, _mm_set1_epi8(92)));
    running = _mm_or_si128(
        running, _mm_cmpeq_epi8(_mm_subs_epu8(word, _mm_set1_epi8(31)),
                                _mm_setzero_si128()));
  }
  if (i < view.size()) {
    __m128i word = _mm_loadu_si128(
        reinterpret_cast<const __m128i *>(view.data() + view.length() - 16));
    running = _mm_or_si128(running, _mm_cmpeq_epi8(word, _mm_set1_epi8(34)));
    running = _mm_or_si128(running, _mm_cmpeq_epi8(word, _mm_set1_epi8(92)));
    running = _mm_or_si128(
        running, _mm_cmpeq_epi8(_mm_subs_epu8(word, _mm_set1_epi8(31)),
                                _mm_setzero_si128()));
  }
  return _mm_movemask_epi8(running) != 0;
}
#else
simdjson_inline bool fast_needs_escaping(std::string_view view) {
  return simple_needs_escaping(view);
}
#endif

SIMDJSON_CONSTEXPR_LAMBDA inline size_t
find_next_json_quotable_character(const std::string_view view,
                                  size_t location) noexcept {

  for (auto pos = view.begin() + location; pos != view.end(); ++pos) {
    if (json_quotable_character[static_cast<uint8_t>(*pos)]) {
      return pos - view.begin();
    }
  }
  return size_t(view.size());
}

SIMDJSON_CONSTEXPR_LAMBDA static std::string_view control_chars[] = {
    "\\u0000", "\\u0001", "\\u0002", "\\u0003", "\\u0004", "\\u0005", "\\u0006",
    "\\u0007", "\\b",     "\\t",     "\\n",     "\\u000b", "\\f",     "\\r",
    "\\u000e", "\\u000f", "\\u0010", "\\u0011", "\\u0012", "\\u0013", "\\u0014",
    "\\u0015", "\\u0016", "\\u0017", "\\u0018", "\\u0019", "\\u001a", "\\u001b",
    "\\u001c", "\\u001d", "\\u001e", "\\u001f"};

// All Unicode characters may be placed within the quotation marks, except for
// the characters that MUST be escaped: quotation mark, reverse solidus, and the
// control characters (U+0000 through U+001F). There are two-character sequence
// escape representations of some popular characters:
// \", \\, \b, \f, \n, \r, \t.
SIMDJSON_CONSTEXPR_LAMBDA void escape_json_char(char c, char *&out) {
  if (c == '"') {
    memcpy(out, "\\\"", 2);
    out += 2;
  } else if (c == '\\') {
    memcpy(out, "\\\\", 2);
    out += 2;
  } else {
    std::string_view v = control_chars[uint8_t(c)];
    memcpy(out, v.data(), v.size());
    out += v.size();
  }
}

inline size_t write_string_escaped(const std::string_view input, char *out) {
  size_t mysize = input.size();
  if (!fast_needs_escaping(input)) { // fast path!
    memcpy(out, input.data(), input.size());
    return input.size();
  }
  const char *const initout = out;
  size_t location = find_next_json_quotable_character(input, 0);
  memcpy(out, input.data(), location);
  out += location;
  escape_json_char(input[location], out);
  location += 1;
  while (location < mysize) {
    size_t newlocation = find_next_json_quotable_character(input, location);
    memcpy(out, input.data() + location, newlocation - location);
    out += newlocation - location;
    location = newlocation;
    if (location == mysize) {
      break;
    }
    escape_json_char(input[location], out);
    location += 1;
  }
  return out - initout;
}


template <typename Storage>
simdjson_inline bool string_builder_base<Storage>::capacity_check(size_t upcoming_bytes) {
  // We use the convention that when is_valid is false, then the capacity and
  // the position are 0.
  // Most of the time, this function will return true.
  const size_t cap = Storage::capacity();
  if (simdjson_likely(upcoming_bytes <= cap - position)) {
    return true;
  }
  // check for overflow, most of the time there is no overflow
  if (simdjson_likely(position + upcoming_bytes < position)) {
    return false;
  }
  // We will rarely get here.
  grow_buffer((std::max)(cap * 2, position + upcoming_bytes));
  // If the buffer allocation failed, we set is_valid to false.
  return is_valid;
}

template <typename Storage>
simdjson_inline void string_builder_base<Storage>::grow_buffer(size_t desired_capacity) {
  if (!is_valid) {
    return;
  }
  if (!Storage::grow(desired_capacity, position)) {
    set_valid(false);
  }
}

template <typename Storage>
simdjson_inline void string_builder_base<Storage>::set_valid(bool valid) noexcept {
  if (!valid) {
    is_valid = false;
    position = 0;
    Storage::invalidate();
  } else {
    is_valid = true;
  }
}

template <typename Storage>
simdjson_inline size_t string_builder_base<Storage>::size() const noexcept {
  return position;
}

template <typename Storage>
simdjson_inline void string_builder_base<Storage>::append(char c) noexcept {
  if (capacity_check(1)) {
    Storage::data()[position++] = c;
  }
}

template <typename Storage>
simdjson_inline void string_builder_base<Storage>::append_null() noexcept {
  constexpr char null_literal[] = "null";
  constexpr size_t null_len = sizeof(null_literal) - 1;
  if (capacity_check(null_len)) {
    std::memcpy(Storage::data() + position, null_literal, null_len);
    position += null_len;
  }
}

template <typename Storage>
simdjson_inline void string_builder_base<Storage>::clear() noexcept {
  position = 0;
  // if it was invalid, we should try to repair it
  if (!is_valid) {
    Storage::invalidate();
    is_valid = true;
  }
}

namespace internal {

template <typename number_type, typename = typename std::enable_if<
                                    std::is_unsigned<number_type>::value>::type>
simdjson_really_inline int int_log2(number_type x) {
  return 63 - leading_zeroes(uint64_t(x) | 1);
}

simdjson_really_inline int fast_digit_count_32(uint32_t x) {
  static uint64_t table[] = {
      4294967296,  8589934582,  8589934582,  8589934582,  12884901788,
      12884901788, 12884901788, 17179868184, 17179868184, 17179868184,
      21474826480, 21474826480, 21474826480, 21474826480, 25769703776,
      25769703776, 25769703776, 30063771072, 30063771072, 30063771072,
      34349738368, 34349738368, 34349738368, 34349738368, 38554705664,
      38554705664, 38554705664, 41949672960, 41949672960, 41949672960,
      42949672960, 42949672960};
  return uint32_t((x + table[int_log2(x)]) >> 32);
}

simdjson_really_inline int fast_digit_count_64(uint64_t x) {
  static uint64_t table[] = {9,
                             99,
                             999,
                             9999,
                             99999,
                             999999,
                             9999999,
                             99999999,
                             999999999,
                             9999999999,
                             99999999999,
                             999999999999,
                             9999999999999,
                             99999999999999,
                             999999999999999ULL,
                             9999999999999999ULL,
                             99999999999999999ULL,
                             999999999999999999ULL,
                             9999999999999999999ULL};
  int y = (19 * int_log2(x) >> 6);
  y += x > table[y];
  return y + 1;
}

template <typename number_type, typename = typename std::enable_if<
                                    std::is_unsigned<number_type>::value>::type>
simdjson_really_inline size_t digit_count(number_type v) noexcept {
  static_assert(sizeof(number_type) == 8 || sizeof(number_type) == 4 ||
                    sizeof(number_type) == 2 || sizeof(number_type) == 1,
                "We only support 8-bit, 16-bit, 32-bit and 64-bit numbers");
  SIMDJSON_IF_CONSTEXPR(sizeof(number_type) <= 4) {
    return fast_digit_count_32(static_cast<uint32_t>(v));
  }
  else {
    return fast_digit_count_64(static_cast<uint64_t>(v));
  }
}
static const char decimal_table[200] = {
    0x30, 0x30, 0x30, 0x31, 0x30, 0x32, 0x30, 0x33, 0x30, 0x34, 0x30, 0x35,
    0x30, 0x36, 0x30, 0x37, 0x30, 0x38, 0x30, 0x39, 0x31, 0x30, 0x31, 0x31,
    0x31, 0x32, 0x31, 0x33, 0x31, 0x34, 0x31, 0x35, 0x31, 0x36, 0x31, 0x37,
    0x31, 0x38, 0x31, 0x39, 0x32, 0x30, 0x32, 0x31, 0x32, 0x32, 0x32, 0x33,
    0x32, 0x34, 0x32, 0x35, 0x32, 0x36, 0x32, 0x37, 0x32, 0x38, 0x32, 0x39,
    0x33, 0x30, 0x33, 0x31, 0x33, 0x32, 0x33, 0x33, 0x33, 0x34, 0x33, 0x35,
    0x33, 0x36, 0x33, 0x37, 0x33, 0x38, 0x33, 0x39, 0x34, 0x30, 0x34, 0x31,
    0x34, 0x32, 0x34, 0x33, 0x34, 0x34, 0x34, 0x35, 0x34, 0x36, 0x34, 0x37,
    0x34, 0x38, 0x34, 0x39, 0x35, 0x30, 0x35, 0x31, 0x35, 0x32, 0x35, 0x33,
    0x35, 0x34, 0x35, 0x35, 0x35, 0x36, 0x35, 0x37, 0x35, 0x38, 0x35, 0x39,
    0x36, 0x30, 0x36, 0x31, 0x36, 0x32, 0x36, 0x33, 0x36, 0x34, 0x36, 0x35,
    0x36, 0x36, 0x36, 0x37, 0x36, 0x38, 0x36, 0x39, 0x37, 0x30, 0x37, 0x31,
    0x37, 0x32, 0x37, 0x33, 0x37, 0x34, 0x37, 0x35, 0x37, 0x36, 0x37, 0x37,
    0x37, 0x38, 0x37, 0x39, 0x38, 0x30, 0x38, 0x31, 0x38, 0x32, 0x38, 0x33,
    0x38, 0x34, 0x38, 0x35, 0x38, 0x36, 0x38, 0x37, 0x38, 0x38, 0x38, 0x39,
    0x39, 0x30, 0x39, 0x31, 0x39, 0x32, 0x39, 0x33, 0x39, 0x34, 0x39, 0x35,
    0x39, 0x36, 0x39, 0x37, 0x39, 0x38, 0x39, 0x39,
};
} // namespace internal

template <typename Storage>
template <typename number_type, typename>
simdjson_inline void string_builder_base<Storage>::append(number_type v) noexcept {
  static_assert(std::is_same<number_type, bool>::value ||
                    std::is_integral<number_type>::value ||
                    std::is_floating_point<number_type>::value,
                "Unsupported number type");
  // If C++17 is available, we can 'if constexpr' here.
  SIMDJSON_IF_CONSTEXPR(std::is_same<number_type, bool>::value) {
    if (v) {
      constexpr char true_literal[] = "true";
      constexpr size_t true_len = sizeof(true_literal) - 1;
      if (capacity_check(true_len)) {
        std::memcpy(Storage::data() + position, true_literal, true_len);
        position += true_len;
      }
    } else {
      constexpr char false_literal[] = "false";
      constexpr size_t false_len = sizeof(false_literal) - 1;
      if (capacity_check(false_len)) {
        std::memcpy(Storage::data() + position, false_literal, false_len);
        position += false_len;
      }
    }
  }
  else SIMDJSON_IF_CONSTEXPR(std::is_unsigned<number_type>::value) {
    constexpr size_t max_number_size = 20;
    if (capacity_check(max_number_size)) {
      using unsigned_type = typename std::make_unsigned<number_type>::type;
      unsigned_type pv = static_cast<unsigned_type>(v);
      size_t dc = internal::digit_count(pv);
      char *write_pointer = Storage::data() + position + dc - 1;
      while (pv >= 100) {
        memcpy(write_pointer - 1, &internal::decimal_table[(pv % 100) * 2], 2);
        write_pointer -= 2;
        pv /= 100;
      }
      if (pv >= 10) {
        *write_pointer-- = char('0' + (pv % 10));
        pv /= 10;
      }
      *write_pointer = char('0' + pv);
      position += dc;
    }
  }
  else SIMDJSON_IF_CONSTEXPR(std::is_integral<number_type>::value) {
    constexpr size_t max_number_size = 20;
    if (capacity_check(max_number_size)) {
      using unsigned_type = typename std::make_unsigned<number_type>::type;
      bool negative = v < 0;
      unsigned_type pv = static_cast<unsigned_type>(v);
      if (negative) {
        pv = 0 - pv; // the 0 is for Microsoft
      }
      size_t dc = internal::digit_count(pv);
      // by always writing the minus sign, we avoid the branch.
      Storage::data()[position] = '-';
      position += negative ? 1 : 0;
      char *write_pointer = Storage::data() + position + dc - 1;
      while (pv >= 100) {
        memcpy(write_pointer - 1, &internal::decimal_table[(pv % 100) * 2], 2);
        write_pointer -= 2;
        pv /= 100;
      }
      if (pv >= 10) {
        *write_pointer-- = char('0' + (pv % 10));
        pv /= 10;
      }
      *write_pointer = char('0' + pv);
      position += dc;
    }
  }
  else SIMDJSON_IF_CONSTEXPR(std::is_floating_point<number_type>::value) {
    constexpr size_t max_number_size = 24;
    if (capacity_check(max_number_size)) {
      // We could specialize for float.
      char *end = simdjson::internal::to_chars(Storage::data() + position, nullptr,
                                               double(v));
      position = end - Storage::data();
    }
  }
}

template <typename Storage>
simdjson_inline void string_builder_base<Storage>::escape_and_append(std::string_view input) noexcept {
  // escaping might turn a control character into \x00xx so 6 characters.
  if (capacity_check(6 * input.size())) {
    position += write_string_escaped(input, Storage::data() + position);
  }
}

template <typename Storage>
simdjson_inline void string_builder_base<Storage>::escape_and_append_with_quotes(std::string_view input) noexcept {
  // escaping might turn a control character into \x00xx so 6 characters.
  if (capacity_check(2 + 6 * input.size())) {
    Storage::data()[position++] = '"';
    position += write_string_escaped(input, Storage::data() + position);
    Storage::data()[position++] = '"';
  }
}

template <typename Storage>
simdjson_inline void string_builder_base<Storage>::escape_and_append_with_quotes(char input) noexcept {
  // escaping might turn a control character into \x00xx so 6 characters.
  if (capacity_check(2 + 6 * 1)) {
    Storage::data()[position++] = '"';
    std::string_view cinput(&input, 1);
    position += write_string_escaped(cinput, Storage::data() + position);
    Storage::data()[position++] = '"';
  }
}

template <typename Storage>
simdjson_inline void string_builder_base<Storage>::escape_and_append_with_quotes(const char *input) noexcept {
  std::string_view cinput(input);
  escape_and_append_with_quotes(cinput);
}
#if SIMDJSON_SUPPORTS_CONCEPTS
template <typename Storage>
template <constevalutil::fixed_string key>
simdjson_inline void string_builder_base<Storage>::escape_and_append_with_quotes() noexcept {
  escape_and_append_with_quotes(constevalutil::string_constant<key>::value);
}
#endif

template <typename Storage>
simdjson_inline void string_builder_base<Storage>::append_raw(const char *c) noexcept {
  size_t len = std::strlen(c);
  append_raw(c, len);
}

template <typename Storage>
simdjson_inline void string_builder_base<Storage>::append_raw(std::string_view input) noexcept {
  if (capacity_check(input.size())) {
    std::memcpy(Storage::data() + position, input.data(), input.size());
    position += input.size();
  }
}

template <typename Storage>
simdjson_inline void string_builder_base<Storage>::append_raw(const char *str, size_t len) noexcept {
  if (capacity_check(len)) {
    std::memcpy(Storage::data() + position, str, len);
    position += len;
  }
}
#if SIMDJSON_SUPPORTS_CONCEPTS
// Support for optional types (std::optional, etc.)
template <typename Storage>
template <concepts::optional_type T>
  requires(!require_custom_serialization<T>)
simdjson_inline void string_builder_base<Storage>::append(const T &opt) {
  if (opt) {
    append(*opt);
  } else {
    append_null();
  }
}

// custom-serialization append for string_builder is defined inline in header

template <typename Storage>
template <typename T>
  requires(std::is_convertible<T, std::string_view>::value ||
           std::is_same<T, const char *>::value)
simdjson_inline void string_builder_base<Storage>::append(const T &value) {
  escape_and_append_with_quotes(value);
}
#endif

#if SIMDJSON_SUPPORTS_RANGES && SIMDJSON_SUPPORTS_CONCEPTS
// Support for range-based appending (std::ranges::view, etc.)
template <typename Storage>
template <std::ranges::range R>
  requires(!std::is_convertible<R, std::string_view>::value)
simdjson_inline void string_builder_base<Storage>::append(const R &range) noexcept {
  auto it = std::ranges::begin(range);
  auto end = std::ranges::end(range);
  if constexpr (concepts::is_pair<typename R::value_type>) {
    start_object();

    if (it == end) {
      end_object();
      return; // Handle empty range
    }
    // Append first item without leading comma
    append_key_value(it->first, it->second);
    ++it;

    // Append remaining items with preceding commas
    for (; it != end; ++it) {
      append_comma();
      append_key_value(it->first, it->second);
    }
    end_object();
  } else {
    start_array();
    if (it == end) {
      end_array();
      return; // Handle empty range
    }

    // Append first item without leading comma
    append(*it);
    ++it;

    // Append remaining items with preceding commas
    for (; it != end; ++it) {
      append_comma();
      append(*it);
    }
    end_array();
  }
}

#endif

#if SIMDJSON_EXCEPTIONS
template <typename Storage>
simdjson_inline string_builder_base<Storage>::operator std::string() const noexcept(false) {
  return std::string(operator std::string_view());
}

template <typename Storage>
simdjson_inline string_builder_base<Storage>::operator std::string_view() const
    noexcept(false) simdjson_lifetime_bound {
  return view();
}
#endif

template <typename Storage>
simdjson_inline simdjson_result<std::string_view>
string_builder_base<Storage>::view() const noexcept {
  if (!is_valid) {
    return simdjson::OUT_OF_CAPACITY;
  }
  return std::string_view(Storage::data(), position);
}

template <typename Storage>
simdjson_inline simdjson_result<const char *> string_builder_base<Storage>::c_str() noexcept {
  if (capacity_check(1)) {
    Storage::data()[position] = '\0';
    return Storage::data();
  }
  return simdjson::OUT_OF_CAPACITY;
}

template <typename Storage>
simdjson_inline bool string_builder_base<Storage>::validate_unicode() const noexcept {
  return simdjson::validate_utf8(Storage::data(), position);
}

template <typename Storage>
simdjson_inline void string_builder_base<Storage>::start_object() noexcept {
  if (capacity_check(1)) {
    Storage::data()[position++] = '{';
  }
}

template <typename Storage>
simdjson_inline void string_builder_base<Storage>::end_object() noexcept {
  if (capacity_check(1)) {
    Storage::data()[position++] = '}';
  }
}

template <typename Storage>
simdjson_inline void string_builder_base<Storage>::start_array() noexcept {
  if (capacity_check(1)) {
    Storage::data()[position++] = '[';
  }
}

template <typename Storage>
simdjson_inline void string_builder_base<Storage>::end_array() noexcept {
  if (capacity_check(1)) {
    Storage::data()[position++] = ']';
  }
}

template <typename Storage>
simdjson_inline void string_builder_base<Storage>::append_comma() noexcept {
  if (capacity_check(1)) {
    Storage::data()[position++] = ',';
  }
}

template <typename Storage>
simdjson_inline void string_builder_base<Storage>::append_colon() noexcept {
  if (capacity_check(1)) {
    Storage::data()[position++] = ':';
  }
}

template <typename Storage>
template <typename key_type, typename value_type>
simdjson_inline void string_builder_base<Storage>::append_key_value(key_type key, value_type value) noexcept {
  static_assert(std::is_same<key_type, const char *>::value ||
                    std::is_convertible<key_type, std::string_view>::value,
                "Unsupported key type");
  escape_and_append_with_quotes(key);
  append_colon();
  SIMDJSON_IF_CONSTEXPR(std::is_same<value_type, std::nullptr_t>::value) {
    append_null();
  }
  else SIMDJSON_IF_CONSTEXPR(std::is_same<value_type, char>::value) {
    escape_and_append_with_quotes(value);
  }
  else SIMDJSON_IF_CONSTEXPR(
      std::is_convertible<value_type, std::string_view>::value) {
    escape_and_append_with_quotes(value);
  }
  else SIMDJSON_IF_CONSTEXPR(std::is_same<value_type, const char *>::value) {
    escape_and_append_with_quotes(value);
  }
  else {
    append(value);
  }
}

#if SIMDJSON_SUPPORTS_CONCEPTS
 template <typename Storage>
 template <constevalutil::fixed_string key, typename value_type>
 simdjson_inline void string_builder_base<Storage>::append_key_value(value_type value) noexcept {
  escape_and_append_with_quotes<key>();
  append_colon();
  SIMDJSON_IF_CONSTEXPR(std::is_same<value_type, std::nullptr_t>::value) {
    append_null();
  }
  else SIMDJSON_IF_CONSTEXPR(std::is_same<value_type, char>::value) {
    escape_and_append_with_quotes(value);
  }
  else SIMDJSON_IF_CONSTEXPR(
      std::is_convertible<value_type, std::string_view>::value) {
    escape_and_append_with_quotes(value);
  }
  else SIMDJSON_IF_CONSTEXPR(std::is_same<value_type, const char *>::value) {
    escape_and_append_with_quotes(value);
  }
  else {
    append(value);
  }
}
#endif

} // namespace builder
} // namespace SIMDJSON_IMPLEMENTATION
} // namespace simdjson

#endif // SIMDJSON_GENERIC_STRING_BUILDER_INL_H
