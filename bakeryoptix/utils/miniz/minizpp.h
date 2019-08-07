#pragma once
#include <string>

namespace utils
{
	enum class mz_result : int
	{
		ok = 0,
		stream_end = 1,
		need_dict = 2,
		err_no = -1,
		stream_error = -2,
		data_error = -3,
		mem_error = -4,
		buf_error = -5,
		version_error = -6,
		limit_exceed_1 = 13,
		limit_exceed_2 = 14,
		param_error = -10000
	};

	std::ostream& operator<<(std::ostream& os, mz_result self);
	mz_result mz_compress_dynamic(std::string& output, const char* data, std::size_t size);
	mz_result mz_uncompress_dynamic(std::string& output, const char* data, std::size_t size, std::size_t size_limit = 1000000000);
}
