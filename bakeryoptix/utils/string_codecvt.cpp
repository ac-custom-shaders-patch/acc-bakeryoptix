#include "string_codecvt.h"

#include <string>
#include <Windows.h>

const std::string& utf16_to_utf8(const std::string& s)
{
	return s;
}

std::string utf16_to_utf8(const wchar_t* s, size_t len)
{
	std::string result;
	if (len > 0)
	{
		result.resize(len * 2);
		auto r = WideCharToMultiByte(CP_UTF8, 0, s, int(len), &result[0], int(result.size()), nullptr, nullptr);
		if (r == 0)
		{
			result.resize(WideCharToMultiByte(CP_UTF8, 0, s, int(len), nullptr, 0, nullptr, nullptr));
			r = WideCharToMultiByte(CP_UTF8, 0, s, int(len), &result[0], int(result.size()), nullptr, nullptr);
		}
		result.resize(r);
	}
	return result;
}

std::string utf16_to_utf8(const std::wstring& s)
{
	return utf16_to_utf8(s.c_str(), s.size());
}

const std::wstring& utf8_to_utf16(const std::wstring& s)
{
	return s;
}

std::wstring utf8_to_utf16(const char* s, size_t len)
{
	// return std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>>().from_bytes(s, s + len);

	std::wstring result;
	result.resize(len);
	if (len > 0)
	{
		auto r = MultiByteToWideChar(CP_UTF8, 0, s, int(len), &result[0], int(result.size()));
		if (r == 0)
		{
			result.resize(MultiByteToWideChar(CP_UTF8, 0, s, int(len), nullptr, 0));
			r = MultiByteToWideChar(CP_UTF8, 0, s, int(len), &result[0], int(result.size()));
		}
		result.resize(r);
	}
	return result;
}

std::wstring utf8_to_utf16(const std::string& s)
{
	return utf8_to_utf16(s.c_str(), s.size());
}
