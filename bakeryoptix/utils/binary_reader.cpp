#include "binary_reader.h"
#include <istream>
#include <utils/filesystem.h>
#include <fstream>

namespace utils
{
	binary_reader::binary_reader(std::basic_istream<char>& stream, const uint32_t buffer_size)
		: stream_(&stream), buffer_(new char[buffer_size]), own_(false),
		  buffer_size_(buffer_size) { }

	binary_reader::binary_reader(const path& path, const uint32_t buffer_size)
		: stream_(new std::ifstream(path.wstring(), std::ios::binary)), buffer_(new char[buffer_size]), own_(true),
		  buffer_size_(buffer_size) { }

	binary_reader::~binary_reader()
	{
		if (own_) delete stream_;
		delete[] buffer_;
	}

	void binary_reader::skip(const uint32_t count)
	{
		if (left_ >= count)
		{
			get_pos_and_move(count);
		}
		else
		{
			stream_->seekg(size_t(long(stream_->tellg()) - long(left_) + count), SEEK_SET);
			left_ = 0;
		}
	}

	void binary_reader::skip_string()
	{
		skip(read_uint());
	}

	std::string binary_reader::read_string()
	{
		const auto length = read_uint();
		return std::string(&buffer_[get_pos_and_move(length)], length);
	}

	bool binary_reader::match(const char* str)
	{
		while (*str)
		{
			if (read_char() != *str) return false;
			str++;
		}
		return true;
	}

	char binary_reader::read_char()
	{
		return buffer_[get_pos_and_move(1)];
	}

	bool binary_reader::read_bool()
	{
		return read_char() != 0;
	}

	char binary_reader::next_char()
	{
		return buffer_[total_ - left_--];
	}

	void binary_reader::require(const uint32_t count)
	{
		if (left_ < count)
		{
			if (left_ > 0)
			{
				memcpy(buffer_, &buffer_[total_ - left_], left_);
			}

			const auto left_to_fill = buffer_size_ - left_;
			stream_->read(&buffer_[left_], left_to_fill);
			left_ += uint32_t(stream_->gcount());
			total_ = left_;

			if (left_ < count) throw std::runtime_error("Unexpected end");
		}
	}

	int binary_reader::get_pos_and_move(const uint32_t count)
	{
		require(count);
		const auto p = total_ - left_;
		left_ -= count;
		return p;
	}
}
