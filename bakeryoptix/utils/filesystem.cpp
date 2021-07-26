/**
 * Copyright (C) 2014 Patrick Mours. All rights reserved.
 * License: https://github.com/crosire/reshade#license
 */

#include <ShlObj.h>
#include <Shlwapi.h>

#include "filesystem.h"
#include "string_codecvt.h"
#include <fstream>
#include <iterator>

namespace utils
{
	bool path::operator==(const path& other) const
	{
		return _stricmp(data_.c_str(), other.data_.c_str()) == 0;
	}

	bool path::operator!=(const path& other) const
	{
		return !operator==(other);
	}

	std::wstring path::wstring() const
	{
		return utf8_to_utf16(data_);
	}

	std::ostream& operator<<(std::ostream& stream, const path& path)
	{
		return stream << '\'' << path.string() << '\'';
	}

	bool path::is_absolute() const
	{
		return data_.size() > 2 && data_[1] == ':' && (data_[2] == '/' || data_[2] == '\\');
		// return PathIsRelativeW(utf8_to_utf16(data_).c_str()) == FALSE;
	}

	path path::parent_path() const
	{
		const auto e = data_.find_last_of("/\\");
		return e == std::string::npos ? path() : data_.substr(0, e);
	}

	std::string path::relative_ac() const
	{
		auto e0 = data_.find("/content/");
		if (e0 == std::string::npos)
		{
			e0 = data_.find("\\content\\");
		}
		if (e0 != std::string::npos)
		{
			return data_.substr(e0 + 1);
		}
		return data_;
	}

	path path::filename() const
	{
		const auto e = data_.find_last_of("/\\");
		return e == std::string::npos ? data_ : data_.substr(e + 1);

		// WCHAR buffer[MAX_PATH] = {};
		// utf8_to_utf16(data_, buffer);
		// return utf16_to_utf8(PathFindFileNameW(buffer));
	}

	path path::filename_without_extension() const
	{
		const auto e = data_.find_last_of("/\\");
		const auto o = e == std::string::npos ? 0 : e + 1;
		auto s = data_.find_last_of('.');
		if (s <= o) s = std::string::npos;
		return s == std::string::npos ? data_.substr(o) : data_.substr(o, s - o);
	}

	std::string path::extension() const
	{
		const auto e = data_.find_last_of("/\\");
		const auto o = e == std::string::npos ? 0 : e + 1;
		auto s = data_.find_last_of('.');
		if (s <= o) s = std::string::npos;
		return s == std::string::npos ? std::string() : data_.substr(s);
	}

	path path::operator/(const path& more) const
	{
		if (data_.empty()) return more;
		if (data_[data_.size() - 1] == '\\') return data_ + more.string();
		if (data_[data_.size() - 1] == '/') return data_.substr(0, data_.size() - 1) + "\\" + more.string();
		return data_ + "\\" + more.string();
	}

	bool exists(const path& path)
	{
		return GetFileAttributesW(path.wstring().c_str()) != INVALID_FILE_ATTRIBUTES;
	}

	bool create_dir(const path& path)
	{
		const auto parent = path.parent_path();
		if (parent.string().size() > 4 && parent != path && !exists(parent))
		{
			create_dir(parent);
		}

		return CreateDirectoryW(path.wstring().c_str(), nullptr) != 0;
	}

	long long get_file_size(const path& path)
	{
		WIN32_FIND_DATAW data;
		const auto h = FindFirstFileW(path.wstring().c_str(), &data);
		if (h == INVALID_HANDLE_VALUE) return -1;
		FindClose(h);
		return data.nFileSizeLow | (long long)data.nFileSizeHigh << 32;
	}

	path resolve(const path& filename, const std::vector<path>& paths)
	{
		for (const auto& path : paths)
		{
			auto result = absolute(filename, path);

			if (exists(result))
			{
				return result;
			}
		}

		return filename;
	}

	path absolute(const path& filename, const path& parent_path)
	{
		if (filename.is_absolute())
		{
			return filename;
		}

		WCHAR result[MAX_PATH] = {};
		PathCombineW(result, utf8_to_utf16(parent_path.string()).c_str(), utf8_to_utf16(filename.string()).c_str());

		return utf16_to_utf8(result);
	}

	path get_module_path(void* handle)
	{
		WCHAR result[MAX_PATH] = {};
		GetModuleFileNameW(static_cast<HMODULE>(handle), result, MAX_PATH);

		return utf16_to_utf8(result);
	}

	std::vector<path> list_files(const path& path_val, const std::string& mask, const bool recursive)
	{
		if (!PathIsDirectoryW(path_val.wstring().c_str()))
		{
			return {};
		}

		WIN32_FIND_DATAW ffd;
		const auto handle = FindFirstFileW((path_val / mask).wstring().c_str(), &ffd);
		if (handle == INVALID_HANDLE_VALUE)
		{
			return {};
		}

		std::vector<path> result;

		do
		{
			const auto filename = utf16_to_utf8(ffd.cFileName);
			if (ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
			{
				if (recursive)
				{
					const auto recursive_result = list_files(filename, mask, true);
					result.insert(result.end(), recursive_result.begin(), recursive_result.end());
				}
			}
			else
			{
				result.push_back(path_val / filename);
			}
		}
		while (FindNextFileW(handle, &ffd));
		FindClose(handle);
		return result;
	}

	bool try_find_file(const path& path_val, const std::string& file_name, path& result)
	{
		if (!PathIsDirectoryW(path_val.wstring().c_str())) return false;

		WIN32_FIND_DATAW ffd;
		const auto handle = FindFirstFileW((path_val / "*").wstring().c_str(), &ffd);
		if (handle == INVALID_HANDLE_VALUE) return false;

		do
		{
			if (ffd.cFileName[0] == '.') continue;

			const auto filename = utf16_to_utf8(ffd.cFileName);
			if (ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
			{
				if (try_find_file(path_val / filename, file_name, result)) return true;
			}
			else if (_stricmp(filename.c_str(), file_name.c_str()) == 0)
			{
				result = path_val / filename;
				return true;
			}
		}
		while (FindNextFileW(handle, &ffd));

		FindClose(handle);
		return false;
	}

	data_chunk::data_chunk(const byte* data, const size_t size): size(size), data(data) { }

	data_chunk::~data_chunk()
	{
		delete[] data;
	}

	std::shared_ptr<data_chunk> data_chunk::read_file(const path& filename)
	{
		if (!exists(filename)) return nullptr;

		std::ifstream file(filename.wstring(), std::ios::binary);
		file.unsetf(std::ios::skipws);

		file.seekg(0, std::ios::end);
		const auto file_size = file.tellg();
		file.seekg(0, std::ios::beg);

		const auto data = new byte[file_size];
		file.read((char*)data, file_size);
		return std::make_shared<data_chunk>(data, file_size);
	}

	std::vector<byte> read_file(const path& filename)
	{
		if (!exists(filename))
		{
			return std::vector<byte>();
		}

		std::ifstream file(filename.wstring(), std::ios::binary);
		file.unsetf(std::ios::skipws);

		file.seekg(0, std::ios::end);
		const auto file_size = file.tellg();
		file.seekg(0, std::ios::beg);

		std::vector<byte> vec;
		vec.reserve(file_size);
		vec.insert(vec.begin(), std::istream_iterator<byte>(file), std::istream_iterator<byte>());
		return vec;
	}
}
