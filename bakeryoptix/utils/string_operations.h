#pragma once
#include <array>
#include <string>
#include <vector>
#include <utils/string_codecvt.h>

namespace utils
{
	template <size_t Size>
	bool tpl_equals(const void* a, const void* b)
	{
		__pragma(warning(push))
		__pragma(warning(disable:4127))
		if (Size == 0) return true;
		if (Size == 1) return *(char*)a == *(char*)b;
		if (Size == 2) return *(uint16_t*)a == *(uint16_t*)b;
		if (Size == 3) return *(uint16_t*)a == *(uint16_t*)b && ((char*)a)[2] == ((char*)b)[2];
		if (Size == 4) return *(uint32_t*)a == *(uint32_t*)b;
		if (Size == 5) return *(uint32_t*)a == *(uint32_t*)b && ((char*)a)[4] == ((char*)b)[4];
		if (Size == 6) return *(uint32_t*)a == *(uint32_t*)b && ((uint16_t*)a)[2] == ((uint16_t*)b)[2];
		if (Size == 7) return *(uint32_t*)a == *(uint32_t*)b && tpl_equals<3>(&((char*)a)[4], &((char*)b)[4]);
		if (Size == 8) return *(uint64_t*)a == *(uint64_t*)b;
		if (Size == 9) return *(uint64_t*)a == *(uint64_t*)b && ((char*)a)[8] == ((char*)b)[8];
		if (Size == 10) return *(uint64_t*)a == *(uint64_t*)b && ((uint16_t*)a)[4] == ((uint16_t*)b)[4];
		if (Size == 11) return *(uint64_t*)a == *(uint64_t*)b && tpl_equals<3>(&((char*)a)[8], &((char*)b)[8]);
		if (Size == 12) return *(uint64_t*)a == *(uint64_t*)b && ((uint32_t*)a)[2] == ((uint32_t*)b)[2];
		if (Size == 13) return *(uint64_t*)a == *(uint64_t*)b && ((uint32_t*)a)[2] == ((uint32_t*)b)[2] && ((char*)a)[12] == ((char*)b)[12];
		if (Size == 14) return *(uint64_t*)a == *(uint64_t*)b && ((uint32_t*)a)[2] == ((uint32_t*)b)[2] && ((uint16_t*)a)[6] == ((uint16_t*)b)[6];
		if (Size == 15) return *(uint64_t*)a == *(uint64_t*)b && ((uint32_t*)a)[2] == ((uint32_t*)b)[2] && tpl_equals<3>(&((char*)a)[12], &((char*)b)[12]);
		if (Size == 16) return *(uint64_t*)a == *(uint64_t*)b && ((uint64_t*)a)[1] == ((uint64_t*)b)[1];
		__pragma(warning(pop))
		return std::memcmp(a, b, Size) == 0;
	}

	template <size_t Size>
	bool tpl_equals_ci(const char* a, const char* b)
	{
		return _strnicmp(a, b, Size) == 0;
	}

	template <size_t Size>
	bool tpl_equals_ci(const wchar_t* a, const wchar_t* b)
	{
		return _wcsnicmp(a, b, Size) == 0;
	}
	
	template <typename CharT, size_t Size>
	size_t tpl_find_ci(const CharT* str1, const CharT* str2, size_t runs)
	{
		for (auto i = 0U; i < runs; ++i)
		{
			if (tpl_equals_ci<Size>(&str1[i], str2)) return i;
		}
		return std::basic_string<CharT>::npos;
	}

	template <typename CharT, size_t N>
	constexpr int case_str(CharT const (&)[N]) { return N - 1; }

	template <typename CharT, std::size_t N>
	bool equals(const std::basic_string<CharT>& s, CharT const (&cs)[N]) { return s.size() == N - 1 && tpl_equals<(N - 1) * sizeof(CharT)>(&s[0], cs); }

	template <typename CharT, std::size_t N>
	bool starts_with(const std::basic_string<CharT>& s, CharT const (&cs)[N]) { return N == 1 || s.size() >= N - 1 && tpl_equals<(N - 1) * sizeof(CharT)>(&s[0], cs); }

	template <typename CharT, std::size_t N>
	bool ends_with(const std::basic_string<CharT>& s, CharT const (&cs)[N]) { return N == 1 || s.size() >= N - 1 && tpl_equals<(N - 1) * sizeof(CharT)>(&s[s.size() - (N - 1)], cs); }

	template <typename CharT, size_t N>
	bool contains(const std::basic_string<CharT>& s, CharT const (&cs)[N]) { return N == 1 || s.size() >= N - 1 && s.find(cs) != std::basic_string<CharT>::npos; }

	template <typename CharT, size_t N>
	size_t find(const std::basic_string<CharT>& s, CharT const (&cs)[N]) { return N == 1 ? 0 : s.size() < N - 1 ? std::basic_string<CharT>::npos : s.find(cs); }

	template <typename CharT, std::size_t N>
	bool equals_ci(const std::basic_string<CharT>& s, CharT const (&cs)[N]) { return s.size() == N - 1 && tpl_equals_ci<N - 1>(&s[0], cs); }

	template <typename CharT, std::size_t N>
	bool starts_with_ci(const std::basic_string<CharT>& s, CharT const (&cs)[N]) { return N == 1 || s.size() >= N - 1 && tpl_equals_ci<N - 1>(&s[0], cs); }

	template <typename CharT, std::size_t N>
	bool ends_with_ci(const std::basic_string<CharT>& s, CharT const (&cs)[N]) { return N == 1 || s.size() >= N - 1 && tpl_equals_ci<N - 1>(&s[s.size() - (N - 1)], cs); }

	template <typename CharT, size_t N>
	bool contains_ci(const std::basic_string<CharT>& s, CharT const (&cs)[N]) { return N == 1 || s.size() >= N - 1 && tpl_find_ci<CharT, N - 1>(&s[0], cs, s.size() - N + 2) != std::basic_string<CharT>::npos; }

	template <typename CharT, size_t N>
	bool find_ci(const std::basic_string<CharT>& s, CharT const (&cs)[N]) { return N == 1 ? 0 : s.size() < N - 1 ? std::basic_string<CharT>::npos : tpl_find_ci<CharT, N - 1>(&s[0], cs, s.size() - N + 2); }
	
	template <typename CharT>
	bool equals_ci(const std::basic_string<CharT>& s, const std::basic_string<CharT>& o) { return s.size() == o.size() && _wcsnicmp(&s[0], &o[0], s.size()) == 0; }
	
	template <typename CharT, size_t N>
	bool contains(const std::vector<std::basic_string<CharT>>& s, CharT const (&cs)[N])
	{
		for (const auto& i : s)
		{
			if (equals(i, cs)) return true;
		}
		return false;
	}

	namespace utils_inner
	{
		std::array<char, 2048>& format_storage();

		template <typename T>
		auto format_arg_inner(const T& a) { return a; }

		template <>
		inline auto format_arg_inner<>(const std::string& a) { return a.c_str(); }

		template <>
		inline auto format_arg_inner<>(const std::wstring& a) { return a.c_str(); }

		template <typename... Args>
		size_t format_inner(const char* format, char*& ret, Args ...args)
		{
			auto& storage = format_storage();
			ret = storage.data();
			const auto size = _snprintf_s(ret, storage.size(), _TRUNCATE, format, format_arg_inner(args)...);
			return size < 0 ? storage.size() - 1 : size_t(size);
		}

		template <typename... Args>
		size_t format_inner(const wchar_t* format, wchar_t*& ret, Args ...args)
		{
			auto& storage = format_storage();
			ret = (wchar_t*)storage.data();
			const auto size = _snwprintf_s(ret, storage.size() / sizeof(wchar_t), _TRUNCATE, format, format_arg_inner(args)...);
			return size < 0 ? storage.size() / sizeof(wchar_t) - 1 : size_t(size);
		}

		template <typename CharT, typename T>
		auto format_arg_prepare(CharT, const T& a) { return a; }

		template <>
		inline auto format_arg_prepare<>(char, const std::wstring& a) { return utf16_to_utf8(a); }

		template <>
		inline auto format_arg_prepare<>(wchar_t, const std::string& a) { return utf8_to_utf16(a); }

		template <>
		inline auto format_arg_prepare<>(char, const wchar_t* const& a) { return utf16_to_utf8(a); }

		template <>
		inline auto format_arg_prepare<>(wchar_t, const char* const& a) { return utf8_to_utf16(a); }

		template <typename CharT, typename... Args>
		size_t format_prepare(const CharT* format, CharT*& ret, Args ...args)
		{
			return format_inner(format, ret, format_arg_prepare(format[0], args)...);
		}

		template <typename CharT>
		struct basic_switch_str
		{
			basic_switch_str(std::basic_string<CharT> s)
				: s_(std::move(s))
			{
				size_ = s_.size();
			}

			template <std::size_t N>
			bool operator ==(CharT const (&cs)[N]) const
			{
				return size_ == N - 1 && std::memcmp(&s_[0], cs, (N - 1) * sizeof(CharT)) == 0;
			}

			template <std::size_t N, typename Callback>
			basic_switch_str<CharT>& when(CharT const (&cs)[N], Callback callback)
			{
				if (!done_ && *this == cs)
				{
					done_ = true;
					callback();
				}
				return *this;
			}

			template <typename Callback>
			basic_switch_str<CharT>& fallback(Callback callback)
			{
				if (!done_)
				{
					done_ = true;
					callback();
				}
				return *this;
			}

		private:
			std::basic_string<CharT> s_;
			size_t size_{};
			bool done_{};
		};

		template <typename TEnum>
		struct parse_enum_helper
		{
			parse_enum_helper(std::string s, TEnum fallback)
				: s_(std::move(s))
			{
				size_ = s_.size();
				value_ = fallback;
			}

			template <std::size_t N>
			parse_enum_helper& when(char const (&cs)[N], TEnum value)
			{
				if (!got_value_ && equals(s_, cs))
				{
					got_value_ = true;
					value_ = value;
				}
				return *this;
			}

			operator TEnum() const
			{
				return value_;
			}

		private:
			std::string s_;
			size_t size_{};
			TEnum value_{};
			bool got_value_{};
		};
	}

	template <typename CharT, typename... Args>
	std::basic_string<CharT> format(const CharT* format, Args ...args)
	{
		CharT* storage;
		return std::basic_string<CharT>(storage, utils_inner::format_prepare(format, storage, std::forward<Args>(args)...));
	}

	template <typename CharT, typename... Args>
	std::basic_string<CharT> format(const std::basic_string<CharT>& format, Args ...args)
	{
		CharT* storage;
		return std::basic_string<CharT>(storage, utils_inner::format_prepare(format.c_str(), storage, std::forward<Args>(args)...));
	}

	template <typename CharT, typename... Args>
	void format_to(std::basic_string<CharT>& dst, const std::basic_string<CharT>& format, Args ...args)
	{
		CharT* storage;
		const int size = utils_inner::format_prepare(format, storage, std::forward<Args>(args)...);
		dst.resize(size);
		memcpy(&dst[0], &storage[0], size);
	}

	inline void ac_string_clear(std::wstring& dst)
	{
		if (*(void**)&dst)
		{
			free(*(wchar_t**)&dst);
		}
		memset(&dst, 0, sizeof(std::wstring));
	}

	struct switch_str : utils_inner::basic_switch_str<char>
	{
		explicit switch_str(const std::basic_string<char>& s) : basic_switch_str<char>(s) {}
	};

	template <typename TEnum>
	utils_inner::parse_enum_helper<TEnum> parse_enum(std::string s, TEnum fallback)
	{
		return utils_inner::parse_enum_helper<TEnum>(std::move(s), fallback);
	}

	uint64_t hash_code(const void* data, size_t size);

	template <typename CharT>
	uint64_t hash_code(const std::basic_string<CharT>& str)
	{
		return hash_code((void*)str.data(), str.size() * sizeof(CharT));
	}
}
