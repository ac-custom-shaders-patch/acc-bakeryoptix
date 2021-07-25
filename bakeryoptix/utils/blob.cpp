#include "blob.h"

#include "string_operations.h"

namespace utils
{
	utils_inner::blob_cursor utils_inner::blob_read_trait::cursor(size_t offset) const
	{
		return blob_cursor{data_(), size_(), offset};
	}

	blob_view utils_inner::blob_read_trait::slice(size_t start, size_t length) const
	{
		if (start + length > size_())
		{
			throw std::runtime_error(format("Invalid length: start=%zu, length=%zu, size=%zu", start, length, size_()));
		}
		return blob_view{data_() + start, length};
	}

	uint64_t utils_inner::blob_read_trait::hash_code() const { return utils::hash_code(data_(), size_()); }
	blob utils_inner::blob_cursor::at_impl<blob>::get() const { return blob(&b[o + sizeof(uint32_t)], *(uint32_t*)&b[o]); }
	blob_view utils_inner::blob_cursor::at_impl<blob_view>::get() const { return blob_view{&b[o + sizeof(uint32_t)], *(uint32_t*)&b[o]}; }
	std::string utils_inner::blob_read_trait::string() const { return std::string(data_(), size_()); }

	blob_view::blob_view(const std::string& str) noexcept : begin_(&*str.begin()), end_(&*str.end()) {}
	blob::blob() noexcept : vector<char>() {}
	blob::blob(const blob_view& view) noexcept : vector<char>(view.data(), view.data() + view.size()) {}
	blob::blob(size_type count) noexcept : vector<char>(count) {}
	blob::blob(const char* data, size_t size) noexcept : vector<char>(data, data + size) {}
	blob::blob(const char* begin, const char* end) noexcept : vector<char>(begin, end) {}
	blob::blob(const iterator& begin, const iterator& end) noexcept : vector<char>(begin, end) {}
	blob::blob(iterator&& begin, iterator&& end) noexcept : vector<char>(begin, end) {}
	blob::blob(const std::string& s) noexcept : blob(s.data(), s.size()) {}

	blob& blob::append(const void* data, size_t size)
	{
		const auto s = std::vector<char>::size();
		resize(s + size);
		memcpy(std::vector<char>::data() + s, data, size);
		return *this;
	}

	blob blob::operator+(const blob_view& b) const
	{
		blob ret(size() + b.size());
		memcpy(ret.data(), data(), size());
		memcpy(ret.data() + size(), b.data(), b.size());
		return ret;
	}

	blob& blob::operator+=(const blob_view& b)
	{
		return append(b.data(), b.size());
	}

	blob& blob::operator^=(const blob_view& key)
	{
		uint64_t i, k;
		const auto data = this->data();
		const auto td = size() / 8;
		const auto tk = key.size() / 8;
		const auto ld = (uint64_t*)data, lk = (uint64_t*)key.data();
		for (i = 0, k = 0; i < td; i++, k++)
		{
			ld[i] ^= lk[k == tk ? (k = 0) : k];
		}
		for (i *= 8, k *= 8; i < size(); i++, k++)
		{
			if (k == key.size()) k = 0;
			data[i] ^= key[k];
		}
		return *this;
	}

	blob blob::operator^(const blob_view& b) const
	{
		auto r = *this;
		r ^= b;
		return r;
	}
}
