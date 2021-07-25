#include "base64.h"

namespace utils
{
	static const std::string base64_chars =
		"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
		"abcdefghijklmnopqrstuvwxyz"
		"0123456789+/";

	static bool is_base64(char c)
	{
		return isalnum(c) || c == '+' || c == '/';
	}

	std::string base64::decode(const std::string& encoded_string)
	{
		auto in_len = int(encoded_string.size());
		auto i = 0;
		auto in = 0;
		uint8_t char_array_4[4], char_array_3[3];
		std::string ret;

		while (in_len-- && encoded_string[in] != '=' && is_base64(encoded_string[in]))
		{
			char_array_4[i++] = encoded_string[in];
			in++;
			if (i == 4)
			{
				for (i = 0; i < 4; i++) char_array_4[i] = uint8_t(base64_chars.find(char_array_4[i]));
				char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
				char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
				char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];
				for (i = 0; i < 3; i++) ret.push_back(char_array_3[i]);
				i = 0;
			}
		}

		if (i)
		{
			for (auto j = i; j < 4; j++) char_array_4[j] = 0;
			for (auto j = 0; j < 4; j++) char_array_4[j] = uint8_t(base64_chars.find(char_array_4[j]));
			char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
			char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
			char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];
			for (auto j = 0; j < i - 1; j++) ret.push_back(char_array_3[j]);
		}

		return ret;
	}
}
