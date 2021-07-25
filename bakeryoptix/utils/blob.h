/* For chunks of binary data to simplify handling, writing and reading. Two new types:
 * 1. blob: A vector owning binary data, can be used for writing and reading.
 * 2. blob_view: Does not own anything, stores pointer to existing data, reading only.
 *
 * Type blob_read_trait is more like an internal trait thing.
 */

#pragma once

#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace utils
{
	struct blob;
	struct blob_view;

	namespace utils_inner
	{
		struct blob_cursor;

		struct blob_read_trait
		{
		protected:
			blob_read_trait() = default;
			~blob_read_trait() = default;

		public:
			template <typename T>
			T read(size_t offset) const
			{
				return cursor(offset).next<T>();
			}

			template <typename T>
			const T& item(size_t offset) const
			{
				if ((offset + 1) * sizeof(T) > size_()) throw std::runtime_error("Unexpected end");
				return *(T*)&data_()[offset * sizeof(T)];
			}
			
			operator bool() const { return size_() != 0; }
			blob_cursor cursor(size_t offset = 0) const;
			blob_view slice(size_t start, size_t length) const;
			std::string string() const;
			uint64_t hash_code() const;

		private:
			struct ptrs_
			{
				const char* begin;
				const char* end;
			};
			size_t size_() const { return end_() - begin_(); }
			const char* data_() const { return begin_(); }
			const char* begin_() const { return ((ptrs_*)this)->begin; }
			const char* end_() const { return ((ptrs_*)this)->end; }
			friend blob;
		};

		struct blob_cursor
		{
			template <typename T>
			struct at_impl
			{
				static_assert(std::is_trivially_copyable<T>::value, "Not trivially copyable");
				const char* b;
				size_t s, o;
				T get() const { return *(T*)&b[o]; }
				size_t len() const { return sizeof(T); }
			};

			template <>
			struct at_impl<blob>
			{
				const char* b;
				size_t s, o;
				blob get() const;
				size_t len() const { return s >= o + sizeof(uint32_t) ? sizeof(uint32_t) + *(uint32_t*)&b[o] : sizeof(uint32_t); }
			};

			template <>
			struct at_impl<blob_view>
			{
				const char* b;
				size_t s, o;
				blob_view get() const;
				size_t len() const { return s >= o + sizeof(uint32_t) ? sizeof(uint32_t) + *(uint32_t*)&b[o] : sizeof(uint32_t); }
			};

			template <>
			struct at_impl<std::string>
			{
				const char* b;
				size_t s, o;
				std::string get() const { return std::string(&b[o + sizeof(uint32_t)], *(uint32_t*)&b[o]); }
				size_t len() const { return s >= o + sizeof(uint32_t) ? sizeof(uint32_t) + *(uint32_t*)&b[o] : sizeof(uint32_t); }
			};

			template <>
			struct at_impl<std::wstring>
			{
				const char* b;
				size_t s, o;
				std::wstring get() const { return std::wstring((const wchar_t*)&b[o + sizeof(uint32_t)], *(uint32_t*)&b[o]); }
				size_t len() const { return s >= o + sizeof(uint32_t) ? sizeof(uint32_t) + *(uint32_t*)&b[o] * 2 : sizeof(uint32_t); }
			};

			template <typename T>
			struct at_impl<std::vector<T>>
			{
				const char* b;
				size_t s, o;
				std::vector<T> get() const { return std::vector<T>((T*)&b[o + sizeof(uint32_t)], (T*)&b[o + sizeof(uint32_t)] + *(uint32_t*)&b[o]); }
				size_t len() const { return s >= o + sizeof(uint32_t) ? sizeof(uint32_t) + *(uint32_t*)&b[o] * sizeof(T) : sizeof(uint32_t); }
			};

			const char* data;
			size_t size;
			size_t cursor;
			blob_cursor(const char* data, size_t size, size_t offset) : data(data), size(size), cursor(offset) {}

			size_t require_and_advance(size_t len)
			{
				if (!has(len)) throw std::runtime_error("Unexpected end");
				const auto ret = cursor;
				cursor += len;
				return ret;
			}

			template <typename T>
			auto next()
			{
				const auto at = at_impl<T>{data, size, cursor};
				require_and_advance(at.len());
				return at.get();
			}

			template <typename T>
			T* next_raw(size_t len)
			{
				return (T*)&data[require_and_advance(sizeof(T) * len)];
			}

			template <typename T>
			bool has() const
			{
				return at_impl<T>{data, size, cursor}.has();
			}

			bool has(size_t len) const { return cursor + len <= size; }
			operator bool() const { return cursor < size; }
		};
	}

	struct blob_view : utils_inner::blob_read_trait
	{
		blob_view() noexcept = default;
		blob_view(const std::string& str) noexcept;
		blob_view(const char* begin, const char* end) noexcept : begin_(begin), end_(end) {}
		blob_view(const char* data, size_t size) noexcept : begin_(data), end_(data + size) {}

		template <typename T>
		blob_view(const T* data, size_t count) noexcept : begin_((const char*)data), end_((const char*)data + sizeof(T) * count) {}

		template <typename T>
		blob_view(const std::vector<T>& vec) noexcept : begin_((const char*)vec.data()), end_((const char*)vec.data() + sizeof(T) * vec.size()) {}

		size_t size() const { return end_ - begin_; }
		const char* data() const { return begin_; }
		const char& operator[](size_t index) const { return begin_[index]; }
		bool empty() const { return size() == 0; }
		const char* begin() const { return begin_; }
		const char* end() const { return end_; }

	private:
		const char* begin_{};
		const char* end_{};
	};

	struct blob : utils_inner::blob_read_trait, std::vector<char>
	{
		blob() noexcept;
		explicit blob(const blob_view& view) noexcept;
		explicit blob(size_type count) noexcept;
		explicit blob(const char* data, size_t size) noexcept;
		explicit blob(const char* begin, const char* end) noexcept;
		explicit blob(const iterator& begin, const iterator& end) noexcept;
		explicit blob(iterator&& begin, iterator&& end) noexcept;
		blob(const blob& rhs) noexcept : blob(rhs.data(), rhs.size()) { }
		blob(const std::vector<char>& rhs) noexcept : blob(rhs.data(), rhs.size()) { }
		blob(blob&& rhs) noexcept { rhs.swap(*this); }
		blob(std::vector<char>&& rhs) noexcept { rhs.swap(*this); }

		// Does not include zero terminating part!
		explicit blob(const std::string& s) noexcept;

		blob& operator=(const blob& rhs) noexcept
		{
			*this = blob(rhs.data(), rhs.size());
			return *this;
		}

		blob& operator=(blob&& rhs) noexcept
		{
			rhs.swap(*this);
			return *this;
		}

		void swap(std::vector<char>& other) noexcept { std::vector<char>::swap(other); }
		
		operator const blob_view&() const { return *(blob_view*)this; }

		blob& append(const void* data, size_t size);
		blob operator +(const blob_view& b) const;
		blob& operator+=(const blob_view& b);
		blob operator +(const blob& b) const { return *this + blob_view(b); }
		blob& operator+=(const blob& b) { return *this += blob_view(b); }

		template <typename T>
		static blob from(const T& v)
		{
			return blob((char*)&v, sizeof(T));
		}

		template <typename T, typename std::enable_if<std::is_trivially_copyable<T>::value, int>::type = 0>
		blob& operator<<(const T& v)
		{
			static_assert(std::is_trivially_copyable<T>::value, "Not trivially copyable");
			if (sizeof(T) <= 8)
			{
				const auto s = size();
				resize(s + sizeof(T));
				*(T*)&data()[s] = v;
				return *this;
			}
			return append(&v, sizeof(T));
		}

		template <typename T>
		blob& operator<<(const std::vector<T>& v)
		{
			reserve(size() + sizeof(uint32_t) + sizeof(T) * v.size());
			*this << uint32_t(v.size());
			append((char*)v.data(), sizeof(T) * v.size());
			return *this;
		}

		blob& operator<<(const std::string& v)
		{
			reserve(size() + sizeof(uint32_t) + sizeof(char) * v.size());
			*this << uint32_t(v.size());
			append((char*)v.data(), sizeof(char) * v.size());
			return *this;
		}

		blob& operator<<(const std::wstring& v)
		{
			reserve(size() + sizeof(uint32_t) + sizeof(wchar_t) * v.size());
			*this << uint32_t(v.size());
			append((wchar_t*)v.data(), sizeof(wchar_t) * v.size());
			return *this;
		}

		blob& operator<<(const blob_read_trait& v)
		{
			reserve(size() + sizeof(uint32_t) + sizeof(char) * v.size_());
			*this << uint32_t(v.size_());
			append((char*)v.data_(), sizeof(char) * v.size_());
			return *this;
		}

		blob& operator^=(const blob_view& key);
		blob operator^(const blob_view& b) const;		
		operator blob_view() noexcept { return {data(), size()}; }
	};

	typedef std::unique_ptr<blob> pblob;
}
