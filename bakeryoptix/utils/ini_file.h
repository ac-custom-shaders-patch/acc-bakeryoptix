/**
 * Copyright (C) 2014 Patrick Mours. All rights reserved.
 * License: https://github.com/crosire/reshade#license
 */

#pragma once

#include <unordered_map>
#include <utility>
#include "variant.h"
#include "filesystem.h"

// #include "iterator_tpl.h"

namespace utils
{
	class ini_file
	{
	public:
		ini_file()
			: virtual_(true), allow_includes_(false) {}
		explicit ini_file(path path, bool allow_includes = false, std::vector<utils::path> resolve_within = {},
		                  std::vector<utils::path> extensions = {}, bool delay_loading = false);
		~ini_file();

		bool is_blank() const
		{
			return sections.empty();
		}

		bool is_virtual() const
		{
			return virtual_;
		}

		static ini_file parse(const std::string& data);
		static ini_file parse(std::shared_ptr<data_chunk> chunk);

		ini_file& operator=(const ini_file& other);

		void load();
		void save();
		void create_and_save();

		std::string to_string() const;
		bool try_find_referenced_file(const std::string& file_name, path& result, const std::vector<path>& extra_resolve_within = {}) const;

		template <typename T>
		T get(const std::string& section, const std::string& key, const T& default_value) const
		{
			T result;
			return try_get(section, key, result) ? result : default_value;
		}

		template <typename T>
		T get_fb(const std::string& section, const std::string& fallback_section, const std::string& key, const T& default_value) const
		{
			T result;
			return try_get_fb(section, fallback_section, key, result) ? result : default_value;
		}

		/**
		 * \brief For cases where section contains values for various fallback_section.
		 */
		template <typename T>
		T get_fb_shared(const std::string& section, const std::string& fallback_section, const std::string& key, const T& default_value) const
		{
			T result;
			return try_get_fb_shared(section, fallback_section, key, result) ? result : default_value;
		}

		bool contains(const std::string& section, const std::string& key) const;

		template <typename T>
		bool try_get(const std::string& section, const std::string& key, T& value) const
		{
			const auto it1 = sections.find(section);
			if (it1 == sections.end()) return false;

			const auto it2 = it1->second.find(key);
			if (it2 == it1->second.end()) return false;

			value = it2->second.as<T>();
			return true;
		}

		template <typename T>
		bool try_get_fb(const std::string& section, const std::string& fallback_section, const std::string& key, T& value) const
		{
			return try_get(section, key, value) || try_get(fallback_section, key, value);
		}

		/**
		 * \brief For cases where section contains values for various fallback_section.
		 */
		template <typename T>
		bool try_get_fb_shared(const std::string& section, const std::string& fallback_section, const std::string& key, T& value) const
		{
			return try_get(section, fallback_section + "_" + key, value) || try_get(fallback_section, key, value);
		}

		template <typename T, size_t Size>
		bool try_get(const std::string& section, const std::string& key, T (&values)[Size]) const
		{
			const auto it1 = sections.find(section);
			if (it1 == sections.end()) return false;

			const auto it2 = it1->second.find(key);
			if (it2 == it1->second.end()) return false;

			for (size_t i = 0; i < Size; i++)
			{
				values[i] = it2->second.as<T>(i);
			}
			return true;
		}

		template <typename T>
		bool try_get(const std::string& section, const std::string& key, std::vector<T>& values) const
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

		template <typename T>
		void set(const std::string& section, const std::string& key, const T& value)
		{
			modified_ = true;
			sections[section][key] = value;
		}

		template <typename T, size_t SIZE>
		void set(const std::string& section, const std::string& key, const T (&values)[SIZE])
		{
			modified_ = true;
			sections[section][key] = values;
		}

		template <typename T, size_t SIZE>
		void set(const std::string& section, const std::string& key, const std::vector<T>& values)
		{
			modified_ = true;
			sections[section][key] = values;
		}

		using section = std::unordered_map<std::string, variant>;
		std::unordered_map<std::string, section> sections;
		path filename;

		std::vector<std::string> iterate_nobreak(const std::string& prefix) const;
		std::vector<std::string> iterate_break(const std::string& prefix, bool no_postfix_for_first = false, int start_with = 0) const;

		/*class it_sections
		{
		public:
			it_sections(ini_file* file, std::string prefix, int start_with = 0, bool no_postfix_for_first = true)
				: file_(file), prefix_(std::move(prefix)),
				  start_with_(start_with),
				  no_postfix_for_first_(no_postfix_for_first) {}

		private:
			const ini_file* file_;
			const std::string prefix_;
			int start_with_;
			bool no_postfix_for_first_;

			struct it_state
			{
				int pos;
				inline void next(const it_sections* ref) { ++pos; }
				inline void begin(const it_sections* ref) { pos = ref->start_with_; }
				inline void end(const it_sections* ref) { pos = ref->vec.size(); }
				inline std::string get(it_sections* ref) const { return get((const it_sections*)ref); }
				inline bool cmp(const it_state& s) const { return pos != s.pos; }
				inline std::string get(const it_sections* ref) const
				{
					return pos == ref->start_with_ && ref->no_postfix_for_first_ ?  ref->prefix_ :  ref->prefix_ + "_" + std::to_string(pos);
				}
			};

			SETUP_ITERATORS(it_sections, std::string, it_state);
		};

		it_sections iterate(std::string prefix, int start_with = 0, bool no_postfix_for_first = true)
		{
			return it_sections(this, prefix, start_with, no_postfix_for_first);
		}*/

	private:
		bool virtual_;
		bool encrypted_ = false;
		bool modified_ = false;
		bool allow_includes_;

		std::vector<path> extensions_;
		std::vector<path> resolve_within_;
	};
}
