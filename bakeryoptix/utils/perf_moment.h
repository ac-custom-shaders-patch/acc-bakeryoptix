#pragma once
#include <string>
#include <chrono>
#include <iostream>
#include <utils/string_codecvt.h>
#include <iomanip>

struct perf_moment
{
	std::string v;
	std::chrono::steady_clock::time_point t;
	bool a;

	perf_moment(const std::string& v, bool active = true)
		: v(v), t(std::chrono::high_resolution_clock::now()), a(active)
	{
		if (a) std::cout << v << utf16_to_utf8(std::wstring(L"…"));
	}

	~perf_moment()
	{
		try
		{
			const auto passed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - t).count();
			if (a) std::cout << "\b: " << passed << " ms" << std::endl;
		}
		catch (std::exception&) {}
	}
};


#ifndef STR
	#define STR(x) #x
#endif
#ifndef STRINGIFY
	#define STRINGIFY(x) STR(x)
#endif

#define PERF_TOKENPASTE(x, y) x ## y
#define PERF_TOKENPASTE2(x, y) PERF_TOKENPASTE(x, y)
#define PERF_UNIQUE PERF_TOKENPASTE2(__perf_, __LINE__)

#define PERF(NAME)\
	perf_moment PERF_UNIQUE(NAME);
