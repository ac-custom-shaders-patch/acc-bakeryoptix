#include "string_operations.h"

#define XXH_STATIC_LINKING_ONLY
#include <array>
#include "xxhash/xxhash.h"

namespace utils
{
	std::array<char, 2048>& utils_inner::format_storage()
	{
		static thread_local std::array<char, 2048> s{};
		return s;
	}

	uint64_t hash_code(const void* data, size_t size)
	{
		return XXH3_64bits(data, size);
	}
}
