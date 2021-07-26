#pragma once

// #include <math/math.h>
#include "string_codecvt.h"
#include <vector>
#include <memory>
#include <sstream>

#undef min
#undef max

#define DONT_COPY_ME(x) \
	private:\
		x(const x&); x& operator=(const x&) = delete;

#define DONT_COPY_ME_OR_MY_SON(x) \
	public:\
		x(){} \
	private:\
		x(const x&); x& operator=(const x&) = delete;

#define DELETE_PTR(x) if (x){ delete x; x = nullptr; }

namespace std_ext // NOLINT
{
	/*namespace adl_helper
	{
		using std::to_string;
		using namespace math;

		template <class T>
		std::string as_string(T&& t)
		{
			return to_string(std::forward<T>(t));
		}

		template <class T>
		std::string as_string(math::float2&& t)
		{
			return std::to_string(t.x) + ", " + std::to_string(t.y);
		}

		template <class T>
		std::string as_string(math::float3&& t)
		{
			return std::to_string(t.x) + ", " + std::to_string(t.y) + ", " + std::to_string(t.z);
		}

		template <class T>
		std::string as_string(math::float4&& t)
		{
			return std::to_string(t.x) + ", " + std::to_string(t.y) + ", " + std::to_string(t.z) + ", " + std::to_string(t.w);
		}
	}*/

	size_t decode_html_entities(char* dest, const char* src);
	std::string decode_html_entities(const std::string& src);
	void decode_html_entities_self(std::string& src);

	/*template <class T>
	std::string to_string(T&& t)
	{
		return adl_helper::as_string(std::forward<T>(t));
	}*/

	std::string join_to_string(const std::vector<std::string>& l, const std::string& separator = ",");

	template <class T>
	std::string join_to_string(const std::vector<T>& l, const std::string& separator = ",")
	{
		std::stringstream ss;
		auto second = false;
		for (auto& v : l)
		{
			if (second) ss << separator;
			else second = true;
			ss << v;
		}
		return ss.str();
	}

	template <class T>
	std::string join_to_brakets_string(const std::vector<T>& l)
	{
		std::stringstream ss;
		ss << "[";
		auto second = false;
		for (auto& v : l)
		{
			if (second) ss << ", ";
			else second = true;
			ss << v;
		}
		ss << "]";
		return ss.str();
	}

	bool match(const std::string& pattern, const std::string& candidate);
	bool match(const std::wstring& pattern, const std::wstring& candidate);

	template <typename T>
	bool match(const std::vector<std::basic_string<T>>& pattern, const std::basic_string<T>& candidate)
	{
		for (auto& p : pattern)
		{
			if (match(p, candidate)) return true;
		}
		return false;
	}

	template <typename T>
	bool mask_fits(const std::basic_string<T>& str, const T* find)
	{
		const auto found = str.find(find);
		return found != std::basic_string<T>::npos;
	}

	template <typename T>
	bool contains(const std::basic_string<T>& str, const T* find)
	{
		const auto found = str.find(find);
		return found != std::basic_string<T>::npos;
	}

	template <typename T>
	bool starts_with(const std::basic_string<T>& name, const T* check_against)
	{
		T c;
		for (auto i = 0; (c = check_against[i]) != '\0'; i++)
		{
			if (name[i] != c) return false;
		}
		return true;
	}

	template <typename T>
	bool replace(std::basic_string<T>& str, const T* from, const T* to)
	{
		size_t start_from = 0;
		const auto from_length = std::basic_string<T>(from).length();
		const auto to_length = std::basic_string<T>(to).length();
		for (auto i = start_from;; i++)
		{
			const auto start_pos = str.find(from, start_from);
			if (start_pos == std::basic_string<T>::npos) return i > 0;
			str.replace(start_pos, from_length, to);
			start_from = start_pos + to_length;
		}
	}

	template <typename T>
	std::basic_string<T> replace_to(const std::basic_string<T>& str, const T* from, const T* to)
	{
		auto result = str;
		replace(result, from, to);
		return result;
	}

	void trim_self(std::string& str, const char* chars = " \t");
	std::string trim(const std::string& str, const char* chars = " \t");
	void trim_self(std::wstring& str, const wchar_t* chars = L" \t");
	std::wstring trim(const std::wstring& str, const wchar_t* chars = L" \t");

	template <typename ... Args>
	std::string format(const std::string& format, Args ... args)
	{
		const int size = std::snprintf(nullptr, 0, format.c_str(), args ...) + 1; // Extra space for '\0'
		std::unique_ptr<char[]> buf(new char[ size ]);
		std::snprintf(buf.get(), size, format.c_str(), args ...);
		return std::string(buf.get(), buf.get() + size - 1); // We don't want the '\0' inside
	}

	template <typename ... Args>
	std::wstring format(const std::wstring& wformat, Args ... args)
	{
		const auto format = utf16_to_utf8(wformat);
		const int size = std::snprintf(nullptr, 0, format.c_str(), args ...) + 1; // Extra space for '\0'
		std::unique_ptr<char[]> buf(new char[ size ]);
		std::snprintf(buf.get(), size, format.c_str(), args ...);
		return utf8_to_utf16(std::string(buf.get(), buf.get() + size - 1)); // We don't want the '\0' inside
	}

	std::vector<std::string> split_string(const std::string& input, const std::string& separator, bool skip_empty = false,
		bool trim_result = true);
	std::vector<std::string> split_string_spaces(const std::string& input);
	std::vector<std::wstring> split_string(const std::wstring& input, const std::wstring& separator, bool skip_empty = false,
		bool trim_result = true);
}
