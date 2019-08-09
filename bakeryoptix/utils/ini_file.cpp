/**
 * Copyright (C) 2014 Patrick Mours. All rights reserved.
 * License: https://github.com/crosire/reshade#license
 */

#include "ini_file.h"

#include <utils/std_ext.h>
#include <fstream>

namespace utils
{
	using section = std::unordered_map<std::string, variant>;

	template <typename T>
	static bool try_get(std::unordered_map<std::string, section>& sections, const std::string& section,
		const std::string& key, std::vector<T>& values)
	{
		const auto it1 = sections.find(section);
		if (it1 == sections.end()) return false;

		const auto it2 = it1->second.find(key);
		if (it2 == it1->second.end()) return false;

		values.clear();
		for (size_t i = 0; i < it2->second.data().size(); i++)
		{
			values.emplace_back(it2->second.as<T>(i));
		}
		return true;
	}

	static bool matches_from(const std::string& s, int index, const char* c)
	{
		for (auto h = c; *h; h++)
		{
			if (index >= s.size() || s[index++] != *h) return false;
		}
		return true;
	}

	static bool is_solid(const std::string& s, int index)
	{
		return index >= 0 && matches_from(s, index, "data:image/png;base64,");
	}

	static void parse_ini_finish(section* cs, const std::string& cs_key, const std::string& data, const int non_space, std::string& key,
		int& started, int& end_at)
	{
		if (!key.empty() && cs)
		{
			std::string value;
			if (started != -1)
			{
				const auto length = 1 + non_space - started;
				value = length < 0 ? "" : data.substr(started, length);
			}
			else
			{
				value = "";
			}

			const auto new_key = (*cs).find(key) == (*cs).end();
			if (!is_solid(data, started) && end_at == -1)
			{
				std::vector<std::string> value_splitted;
				for (size_t i = 0, len = value.size(), found; i < len; i = found + 1)
				{
					found = value.find_first_of(',', i);
					if (found == std::string::npos) found = len;

					auto value_piece = value.substr(i, found - i);
					std_ext::trim_self(value_piece);
					value_splitted.push_back(value_piece);
				}

				if (new_key)
				{
					(*cs)[key] = value_splitted;
				}
				else if (cs_key == "INCLUDE" && key == "INCLUDE")
				{
					auto& existing = (*cs)[key];
					for (const auto& piece : value_splitted)
					{
						existing.data().push_back(piece);
					}
				}
			}
			else if (new_key)
			{
				(*cs)[key] = std::vector<std::string>{value};
			}


			key.clear();
		}

		started = -1;
		end_at = -1;
	}

	static int seq_counter = 0;

	static void parse_ini_values(std::unordered_map<std::string, section>& sections, const char* data, const int data_size)
	{
		std::string cs_key;
		section* cs = nullptr;

		int started = -1;
		int non_space = -1;
		int end_at = -1;
		std::string key;

		for (int i = 0; i < data_size; i++)
		{
			const auto c = data[i];
			switch (c)
			{
			case '[':
				{
					if (end_at != -1 || is_solid(data, started)) goto LAB_DEF;
					parse_ini_finish(cs, cs_key, data, non_space, key, started, end_at);

					const auto s = ++i;
					if (s == data_size) break;
					for (; i < data_size && data[i] != ']'; i++) {}
					cs_key = std::string(&data[s], i - s);
					if (cs_key.size() > 4 && cs_key.substr(cs_key.size() - 4) == "_...")
					{
						cs_key = cs_key.substr(0, cs_key.size() - 3) + ":$SEQ:" + std::to_string(++seq_counter);
					}
					cs = &sections[cs_key];
					break;
				}

			case '\n':
				{
					if (end_at != -1) goto LAB_DEF;
					parse_ini_finish(cs, cs_key, data, non_space, key, started, end_at);
					break;
				}

			case '=':
				{
					if (end_at != -1 || is_solid(data, started)) goto LAB_DEF;
					if (started != -1 && key.empty() && cs != nullptr)
					{
						key = std::string(&data[started], 1 + non_space - started);
						started = -1;
						end_at = -1;
					}
					break;
				}

			case '/':
				{
					if (end_at != -1 || is_solid(data, started)) goto LAB_DEF;
					if (i + 1 < data_size && data[i + 1] == '/')
					{
						goto LAB_SEMIC;
					}
					goto LAB_DEF;
				}

			case ';':
				{
					if (end_at != -1 || is_solid(data, started)) goto LAB_DEF;
				LAB_SEMIC:
					parse_ini_finish(cs, cs_key, data, non_space, key, started, end_at);
					for (i++; i < data_size && data[i] != '\n'; i++) {}
					break;
				}

			case '"':
			case '\'':
			case '`':
				{
					if (!key.empty())
					{
						if (started == -1)
						{
							end_at = c;
							started = i + 1;
							non_space = i + 1;
						}
						else if (c == end_at)
						{
							non_space = i - 1;
							parse_ini_finish(cs, cs_key, data, non_space, key, started, end_at);
						}
					}
				}

			default:
				{
				LAB_DEF:
					if (c != ' ' && c != '\t' && c != '\r')
					{
						non_space = i;
						if (started == -1)
						{
							started = i;
						}
					}
					break;
				}
			}
		}

		parse_ini_finish(cs, cs_key, data, non_space, key, started, end_at);
	}

	static void parse_ini_values(std::unordered_map<std::string, section>& sections, const std::string& data)
	{
		return parse_ini_values(sections, &data[0], int(data.size()));
	}

	static void resolve_sequential(std::unordered_map<std::string, section>& c)
	{
		// std::unordered_map<std::string, section> renamed;
		for (auto it = c.begin(); it != c.end();)
		{
			auto& k = it->first;
			const auto index = k.find(":$SEQ:");
			if (index != std::string::npos)
			{
				auto prefix = k.substr(0, index);
				for (auto i = 0; i < 10000; i++)
				{
					auto candidate = prefix + std::to_string(i);
					if (c.find(candidate) == c.end())
					{
						c[candidate] = it->second;
						c.erase(it);
						goto next;
					}
				}
			}

			++it;
		next: {}
		}

		/*for (auto i : renamed)
		{
			c[i.first] = i.second;
		}*/
	}

	static bool is_dataacd_file(const path& path, utils::path& acd_path, std::string& car_id)
	{
		const auto parent = path.parent_path();
		if (parent.filename() != "data") return false;

		const auto car_dir = parent.parent_path();
		acd_path = car_dir / "data.acd";
		if (!exists(acd_path)) return false;

		car_id = car_dir.filename().string();
		return true;
	}

	static std::string create_key_byte(const int v)
	{
		return std::to_string((v % 256 + 256) % 256);
	}

	static std::string create_key(const std::string& car_id)
	{
		auto input = std::string(car_id);
		const auto input_size = (int)input.size();
		std::transform(input.begin(), input.end(), input.begin(), tolower);

		int values[8]{
			0, 0, 0, 0x1683, 0x42, 0x65, 0xab, 0xab
		};

		for (auto i = 0; i < input_size; i++)
		{
			values[0] += input[i];
		}

		for (auto i = 0; i < input_size - 1; i += 2)
		{
			values[1] = values[1] * input[i] - input[i + 1];
		}

		for (auto i = 1; i < input_size - 3; i += 3)
		{
			values[2] *= input[i];
			values[2] /= input[i + 1] + 0x1b;
			values[2] += -0x1b - input[i - 1];
		}

		for (auto i = 1; i < input_size; i++)
		{
			values[3] -= input[i];
		}

		for (auto i = 1; i < input_size - 4; i += 4)
		{
			values[4] = (input[i] + 0xf) * values[4] * (input[i - 1] + 0xf) + 0x16;
		}

		for (auto i = 0; i < input_size - 2; i += 2)
		{
			values[5] -= input[i];
		}

		for (auto i = 0; i < input_size - 2; i += 2)
		{
			values[6] %= input[i];
		}

		for (auto i = 0; i < input_size - 1; i++)
		{
			values[7] = values[7] / input[i] + input[i + 1];
		}

		std::string result;
		for (auto v : values)
		{
			result += result.empty() ? create_key_byte(v) : "-" + create_key_byte(v);
		}
		return result;
	}

	static std::string read_string(std::ifstream& stream)
	{
		int32_t size;
		stream.read((char*)&size, sizeof size);
		if (!stream || size > 2000000)
		{
			// 2 MB limit for data files
			return "";
		}

		std::string result(size, '\0');
		stream.read(&result[0], size);
		return result;
	}

	static std::string load_dataacd_sections(const path& path, const utils::path& acd_path, const std::string& car_id)
	{
		std::ifstream file(acd_path.wstring(), std::ios::in | std::ios::binary);
		if (!file.is_open()) return std::string();

		int32_t first_int;
		file.read((char*)&first_int, sizeof first_int);
		file.seekg(first_int == -1111 ? 8 : 0, std::ios_base::beg);

		const auto filename = path.filename().string();
		while (!file.eof())
		{
			const auto section = read_string(file);
			if (section.empty()) break;

			int32_t section_size;
			file.read((char*)&section_size, sizeof section_size);
			section_size *= 4;

			if (section == filename)
			{
				const auto bytes = new char[section_size];
				file.read(bytes, section_size);

				std::string decrypted(section_size / 4, '\0');
				const auto key = create_key(car_id);
				const auto key_end = key.size() - 1;
				size_t key_pos = 0;

				for (int i = 0, size = section_size / 4; i < size; i++)
				{
					const int value = (int)bytes[i * 4] - (int)key[key_pos];
					decrypted[i] = (char)(value < 0 ? value + 256 : value);
					key_pos = key_pos == key_end ? 0 : key_pos + 1;
				}

				delete[] bytes;
				return decrypted;
			}

			file.seekg(section_size, std::ios_base::cur);
		}

		return std::string();
	}

	std::string read_ac_file(const path& filename, bool* is_encrypted)
	{
		std::string result;
		std::string car_id;
		path acd_path;
		if (is_dataacd_file(filename, acd_path, car_id))
		{
			if (is_encrypted != nullptr) *is_encrypted = true;
			result = load_dataacd_sections(filename, acd_path, car_id);
		}
		else if (!exists(filename))
		{
			result = std::string();
		}
		else
		{
			std::ifstream file(filename.wstring());
			std::stringstream buffer;
			buffer << file.rdbuf();
			result = buffer.str();
		}
		return result;
	}

	static void load_sections(const path& path, std::vector<std::string>& processed,
		std::unordered_map<std::string, section>& sections, bool allow_includes,
		std::vector<utils::path>& resolve_within, bool* is_encrypted)
	{
		parse_ini_values(sections, read_ac_file(path, is_encrypted));

		std::vector<std::string> includes;
		processed.push_back(path.filename().string());

		if ((is_encrypted != nullptr && *is_encrypted)
			|| !try_get(sections, "INCLUDE", "INCLUDE", includes))
		{
			return;
		}

		if (!allow_includes)
		{
			return;
		}

		std::reverse(includes.begin(), includes.end());
		for (auto& inc : includes)
		{
			std_ext::trim_self(inc);
			const auto name = utils::path(inc).filename().string();
			auto skip = false;
			for (auto& processed_value : processed)
			{
				if (_stricmp(processed_value.c_str(), name.c_str()) == 0)
				{
					skip = true;
					break;
				}
			}
			if (skip) continue;

			for (int i = -1, t = (int)resolve_within.size(); i < t; i++)
			{
				const auto filename = (i == -1 ? path.parent_path() : resolve_within[i]) / inc;
				load_sections(filename, processed, sections, allow_includes, resolve_within, is_encrypted);

				const auto new_resolve_within = filename.parent_path();
				auto add_new_resolve_within = true;
				for (auto& within : resolve_within)
				{
					if (within == new_resolve_within)
					{
						add_new_resolve_within = false;
						break;
					}
				}

				if (add_new_resolve_within)
				{
					resolve_within.push_back(new_resolve_within);
				}
			}
		}
	}

	bool ini_file::try_find_referenced_file(const std::string& file_name, path& result, const std::vector<path>& extra_resolve_within) const
	{
		if (file_name.empty())
		{
			return false;
		}

		{
			const auto candicate = filename.parent_path() / file_name;
			if (exists(candicate) || encrypted_)
			{
				result = candicate;
				return true;
			}
		}

		for (const auto& dir : resolve_within_)
		{
			const auto candicate = dir / file_name;
			if (exists(candicate))
			{
				result = candicate;
				return true;
			}
		}

		for (const auto& dir : extra_resolve_within)
		{
			const auto candicate = dir / file_name;
			if (exists(candicate))
			{
				result = candicate;
				return true;
			}
		}

		return false;
	}

	bool ini_file::contains(const std::string& section, const std::string& key) const
	{
		const auto it1 = sections.find(section);
		return it1 != sections.end() && it1->second.find(key) != it1->second.end();
	}

	ini_file::ini_file(path path, bool allow_includes, std::vector<utils::path> resolve_within,
		std::vector<utils::path> extensions, bool delay_loading)
		: filename(std::move(path)),
		  virtual_(false),
		  allow_includes_(allow_includes),
		  extensions_(std::move(extensions)),
		  resolve_within_(std::move(resolve_within))
	{
		if (!delay_loading)
		{
			load();
		}
	}

	ini_file::~ini_file()
	{
		save();
	}

	ini_file ini_file::parse(const std::string& data)
	{
		ini_file result;
		parse_ini_values(result.sections, data);
		resolve_sequential(result.sections);
		return result;
	}

	ini_file ini_file::parse(std::shared_ptr<data_chunk> chunk)
	{
		ini_file result;
		if (chunk)
		{
			parse_ini_values(result.sections, chunk->c_str, int(chunk->size));
			resolve_sequential(result.sections);
		}
		return result;
	}

	ini_file& ini_file::operator=(const ini_file& other) = default;

	void ini_file::load()
	{
		if (virtual_) return;
		std::vector<std::string> processed;
		sections.clear();

		load_sections(filename, processed, sections, allow_includes_, resolve_within_, &encrypted_);
		if (encrypted_) return;

		for (auto& extension : extensions_)
		{
			load_sections(extension, processed, sections, allow_includes_, resolve_within_, &encrypted_);
		}

		resolve_sequential(sections);
	}

	bool sort_sections(const std::pair<std::string, section>& a, const std::pair<std::string, section>& b)
	{
		return a.first < b.first;
	}

	bool sort_items(const std::pair<std::string, variant>& a, const std::pair<std::string, variant>& b)
	{
		return a.first < b.first;
	}

	std::string ini_file::to_string() const
	{
		std::stringstream stream;

		const auto it = sections.find("");
		if (it != sections.end())
		{
			for (const auto& section_line : it->second)
			{
				stream << section_line.first << '=';
				size_t i = 0;
				for (const auto& item : section_line.second.data())
				{
					if (i++ != 0)
					{
						stream << ',';
					}
					stream << item;
				}
				stream << std::endl;
			}
			stream << std::endl;
		}

		std::vector<std::pair<std::string, section>> elems(sections.begin(), sections.end());
		std::sort(elems.begin(), elems.end(), sort_sections);
		for (const auto& section : elems)
		{
			if (section.first.empty()) continue;
			stream << '[' << section.first << ']' << std::endl;

			std::vector<std::pair<std::string, variant>> items(section.second.begin(), section.second.end());
			std::sort(items.begin(), items.end(), sort_items);
			for (const auto& section_line : items)
			{
				stream << section_line.first << '=';
				size_t i = 0;
				for (const auto& item : section_line.second.data())
				{
					if (i++ != 0)
					{
						stream << ',';
					}
					stream << item;
				}
				stream << std::endl;
			}
			stream << std::endl;
		}

		return stream.str();
	}

	void ini_file::save()
	{
		if (!modified_ || virtual_) return;
		std::ofstream file(filename.wstring());
		file << to_string();
		modified_ = false;
	}

	void ini_file::create_and_save()
	{
		if (!modified_) return;

		const auto dir = filename.parent_path();
		if (!exists(dir))
		{
			create_dir(dir);
		}

		virtual_ = false;
		save();
	}

	std::vector<std::string> ini_file::iterate(const std::string& prefix, bool no_postfix_for_first, int start_with) const
	{
		std::vector<std::string> result;
		for (int index = start_with, stop = (int)sections.size(); stop > 0; stop--, index++)
		{
			const auto s = index == start_with && no_postfix_for_first ? prefix : prefix + "_" + std::to_string(index);
			if (sections.find(s) == sections.end()) break;
			result.push_back(s);
		}
		return result;
	}
}
