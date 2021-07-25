#pragma once
#include <string>

namespace utils
{
	struct base64
	{
		static std::string decode(const std::string& encoded_string);
	};
}
