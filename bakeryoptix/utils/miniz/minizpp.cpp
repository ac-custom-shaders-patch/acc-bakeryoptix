#include "minizpp.h"

#include <utils/miniz/miniz.h>
#include <ostream>

namespace utils
{
	std::ostream& operator<<(std::ostream& os, mz_result self)
	{
		switch (self)
		{
		case mz_result::ok: return os << "mz::OK";
		case mz_result::stream_end: return os << "mz::stream_end";
		case mz_result::need_dict: return os << "mz::need_dict";
		case mz_result::err_no: return os << "mz::errno";
		case mz_result::stream_error: return os << "mz::stream_error";
		case mz_result::data_error: return os << "mz::data_error";
		case mz_result::mem_error: return os << "mz::mem_error";
		case mz_result::buf_error: return os << "mz::buf_error";
		case mz_result::version_error: return os << "mz::version_error";
		case mz_result::limit_exceed_1: return os << "mz::limit_exceed_1";
		case mz_result::limit_exceed_2: return os << "mz::limit_exceed_2";
		case mz_result::param_error: return os << "mz::param_error";
		default: return os << "mz::?";
		}
	}

	mz_result mz_compress_dynamic(std::string& output, const char* data, std::size_t size)
	{
		auto compressed_size = mz_compressBound(mz_ulong(size));
		output.resize(compressed_size);
		const auto result = mz_compress((uint8_t*)&output[0], &compressed_size, reinterpret_cast<const uint8_t*>(data), mz_ulong(size));
		if (result != MZ_OK)
		{
			output.clear();
			return mz_result(result);
		}
		output.resize(compressed_size);
		return mz_result::ok;
	}

	mz_result mz_uncompress_dynamic(std::string& output, const char* data, std::size_t size, std::size_t size_limit)
	{
		mz_stream stream{};
		stream.next_in = reinterpret_cast<const uint8_t*>(data);
		stream.avail_in = mz_uint32(size);

		const auto status = mz_inflateInit(&stream);
		if (status != MZ_OK)
		{
			return mz_result(status);
		}

		if (size > size_limit)
		{
			mz_inflateEnd(&stream);
			return mz_result::limit_exceed_1;
		}

		std::size_t size_uncompressed = 0;
		const auto chunk = uint32_t(2 * size);

		do
		{
			const auto resize_to = size_uncompressed + chunk;
			if (resize_to > size_limit)
			{
				mz_inflateEnd(&stream);
				return mz_result::limit_exceed_2;
			}

			output.resize(resize_to);
			stream.avail_out = chunk;
			stream.next_out = reinterpret_cast<Bytef*>(&output[0] + size_uncompressed);

			const auto ret = mz_inflate(&stream, MZ_SYNC_FLUSH);
			if (ret != MZ_STREAM_END && ret != MZ_OK && ret != MZ_BUF_ERROR)
			{
				mz_inflateEnd(&stream);
				return mz_result(ret);
			}

			const auto got = chunk - stream.avail_out;
			size_uncompressed += got;
		}
		while (stream.avail_out == 0 || stream.avail_in > 0);

		mz_inflateEnd(&stream);
		output.resize(size_uncompressed);
		return mz_result::ok;
	}
}
