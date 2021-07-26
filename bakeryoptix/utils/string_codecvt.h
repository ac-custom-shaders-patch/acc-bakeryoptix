#pragma once
#include <algorithm>
#include <codecvt>
#include <string>

const std::string& utf16_to_utf8(const std::string& s);
std::string utf16_to_utf8(const wchar_t* s, size_t len);
std::string utf16_to_utf8(const std::wstring& s);

const std::wstring& utf8_to_utf16(const std::wstring& s);
std::wstring utf8_to_utf16(const char* s, size_t len);
std::wstring utf8_to_utf16(const std::string& s);
