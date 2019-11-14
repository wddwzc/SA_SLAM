#ifndef SEGMATCH_ID_PAIR_HASH_HPP_
#define SEGMATCH_ID_PAIR_HASH_HPP_

#include <functional>

#include "common.hpp"

namespace segmatch {

/// \brief Struct providing an hashing function for pairs of segment IDs.
struct IdPairHash {
  /// \brief Hashing function for pairs of segment IDs.
  /// \param pair Pair of IDs to be hashed.
  /// \returns The hash of the ID pair.
  // 分割块ID对的哈希函数
  // 参数：ID对
  // 返回：ID对的散列
  size_t operator() (const IdPair& pair) const {
    static_assert(std::is_same<IdPair, std::pair<int64_t, int64_t>>::value,
                  "The hashing function is valid only if IdPair is defined as "
                  "std::pair<int64_t, int64_t>");
    // We expect IDs to be always positive, which enables this trick for combining the two IDs. If
    // that would not be the case the hashing function could be less efficient, but still
    // functional.
	// ID应该是正的，才可以这一操作。如果不是正的，功能性可以满足，但效率会降低
    return std::hash<uint64_t>{}(static_cast<uint64_t>(pair.first) << 1 +
                                 static_cast<uint64_t>(pair.second));
  }
};

} // namespace segmatch

#endif // SEGMATCH_ID_PAIR_HASH_HPP_
