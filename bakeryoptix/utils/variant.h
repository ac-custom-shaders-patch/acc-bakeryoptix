/**
 * Copyright (C) 2014 Patrick Mours. All rights reserved.
 * License: https://github.com/crosire/reshade#license
 */

#pragma once

#include <string>
#include <sstream>
#include <vector>
#include "filesystem.h"

#include <vector_types.h>
#include <optixu/optixu_matrix_namespace.h>

namespace utils
{
	class variant
	{
	public:
		variant() = default;
		variant(const char* value);
		variant(const wchar_t* value);
		variant(const std::vector<std::string>&& values);
		variant(const std::vector<std::wstring>&& values);

		template <typename T>
		variant(T value)
		{
			std::stringstream v;
			v << value;
			values_.push_back(v.str());
		}

		template <>
		variant(const bool& value)
			: variant(value ? "1" : "0") { }

		template <>
		variant(const std::string& value)
			: values_(1, value) { }

		template <>
		variant(const std::wstring& value)
			: values_(1, utf16_to_utf8(value)) { }

		template <>
		variant(std::string value)
			: values_(1, value) { }

		template <>
		variant(std::wstring value)
			: values_(1, utf16_to_utf8(value)) { }

		template <>
		variant(std::vector<std::string> values)
			: values_(std::move(values)) { }

		template <>
		variant(std::vector<std::wstring> values)
		{
			for (auto& v : values)
			{
				values_.push_back(utf16_to_utf8(v));
			}
		}

		template <class InputIt>
		variant(InputIt first, InputIt last)
			: values_(first, last) { }

		template <>
		variant(const path& value)
			: variant(value.string()) { }

		template <>
		variant(const std::vector<path>& values)
			: values_(values.size())
		{
			for (size_t i = 0; i < values.size(); i++) values_[i] = values[i].string();
		}

		template <typename T>
		variant(const T* values, size_t count)
			: values_(count)
		{
			for (size_t i = 0; i < count; i++) values_[i] = std::to_string(values[i]);
		}

		template <>
		variant(const bool* values, size_t count)
			: values_(count)
		{
			for (size_t i = 0; i < count; i++) values_[i] = values[i] ? "1" : "0";
		}

		template <typename T, size_t COUNT>
		variant(const T (&values)[COUNT])
			: variant(values, COUNT) { }

		template <typename T>
		variant(std::initializer_list<T> values)
			: variant(values.begin(), values.size()) { }

		std::vector<std::string>& data();
		const std::vector<std::string>& data() const;

		template <typename T>
		T as(size_t index = 0) const;
				
		template <> variant as(size_t i) const { return *this; }
		template <> bool as(size_t i) const { return as_bool(i); }
		template <> int32_t as(size_t i) const { return int32_t(as_int64_t(i)); }
		template <> uint32_t as(size_t i) const { return uint32_t(as_uint64_t(i)); }
		template <> long as(size_t i) const { return long(as_uint64_t(i)); }
		template <> unsigned long as(size_t i) const { return unsigned long(as_uint64_t(i)); }
		template <> int64_t as(size_t i) const { return as_int64_t(i); }
		template <> uint64_t as(size_t i) const { return as_uint64_t(i); }
		template <> float as(size_t i) const { return float(as_double(i)); }
		template <> double as(size_t i) const { return as_double(i); }
		template <> std::string as(size_t i) const { return as_string(i); }
		template <> std::wstring as(size_t i) const { return as_wstring(i); }
		template <> path as(size_t i) const { return as_string(i); }
		template <> float2 as(size_t i) const { return as_float2(i); }
		template <> float3 as(size_t i) const { return as_float3(i); }
		template <> float4 as(size_t i) const { return as_float4(i); }

	private:
		std::vector<std::string> values_;

		bool as_bool(size_t i) const;
		uint64_t as_uint64_t(size_t i) const;
		int64_t as_int64_t(size_t i) const;
		double as_double(size_t i) const;
		const std::string& as_string(size_t i) const;
		std::wstring as_wstring(size_t i) const;
		float2 as_float2(size_t i) const;
		float3 as_float3(size_t i) const;
		float4 as_float4(size_t i) const;
	};
}
