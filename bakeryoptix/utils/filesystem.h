/**
 * Copyright (C) 2014 Patrick Mours. All rights reserved.
 * License: https://github.com/crosire/reshade#license
 */

#pragma once

#include <string>
#include <vector>
#include <ostream>
#include <memory>
#include "string_codecvt.h"

typedef unsigned char byte;

namespace utils
{
	class path
	{
	public:
		path() {}

		path(const char* data) : data_(data) {}

		path(const wchar_t* data) : data_(utf16_to_utf8(data)) {}

		path(std::string data) : data_(std::move(data)) {}

		path(const std::wstring& data) : data_(utf16_to_utf8(data)) {}

		bool operator==(const path& other) const;
		bool operator!=(const path& other) const;

		std::string& string()
		{
			return data_;
		}

		const std::string& string() const
		{
			return data_;
		}

		std::wstring wstring() const;

		friend std::ostream& operator<<(std::ostream& stream, const path& path);

		bool empty() const
		{
			return data_.empty();
		}

		size_t length() const
		{
			return data_.length();
		}

		bool is_absolute() const;

		path parent_path() const;
		std::string relative_ac() const;
		path filename() const;
		path filename_without_extension() const;
		std::string extension() const;

		path& remove_filename()
		{
			return operator=(parent_path());
		}

		path& replace_extension(const std::string& extension);

		path operator/(const path& more) const;

		path operator+(char c) const
		{
			return data_ + c;
		}

		path operator+(const path& more) const
		{
			return data_ + more.data_;
		}

	private:
		std::string data_;
	};

	bool exists(const path& path);
	bool create_dir(const path& path);
	long long get_file_size(const path& path);
	path resolve(const path& filename, const std::vector<path>& paths);
	path absolute(const path& filename, const path& parent_path);

	path get_module_path(void* handle);

	std::vector<path> list_files(const path& path_val, const std::string& mask = "*", bool recursive = false);
	bool try_find_file(const path& path_val, const std::string& file_name, path& result);

	struct data_chunk
	{
		size_t size;

		union
		{
			const byte* data;
			const char* c_str;
		};

		data_chunk(const byte* data, size_t size);
		~data_chunk();

		static std::shared_ptr<data_chunk> read_file(const path& filename);
	};

	std::vector<byte> read_file(const path& filename);
}
