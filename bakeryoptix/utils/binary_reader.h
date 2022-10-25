#pragma once
#include <istream>
#include <utils/filesystem.h>

#include <vector_types.h>
#include <optixu/optixu_matrix_namespace.h>
#include <bake_api.h>

namespace utils
{
	using namespace optix;

	class binary_reader
	{
	public:
		explicit binary_reader(std::basic_istream<char>& stream, uint32_t buffer_size = 8192);
		explicit binary_reader(const path& path, uint32_t buffer_size = 8192);
		~binary_reader();

		void skip(uint32_t count);
		void skip_string();
		std::string read_string();
		std::string read_rest();
		std::string read_data(uint size);
		bool match(const char* str);
		char read_char();
		bool read_bool();

		int read_int() { return read_ref<int>(); }
		uint32_t read_uint() { return read_ref<uint32_t>(); }
		uint64_t read_uint64() { return read_ref<uint64_t>(); }
		uint16_t read_ushort() { return read_ref<uint16_t>(); }
		float read_float() { return read_ref<float>(); }
		float2 read_f2() { return read_ref<vec2>().optix(); }
		float3 read_f3() { return read_ref<vec3>().optix(); }
		float4 read_f4() { return read_ref<vec4>().optix(); }
		const bake::NodeTransformation& read_f4x4() { return read_ref<bake::NodeTransformation>(); }

		template <typename T>
		const T& read_ref()
		{
			return *reinterpret_cast<T*>(&buffer_[get_pos_and_move(sizeof(T))]);
		}

	private:
		struct vec2
		{
			float x, y;
			float2 optix() const { return float2{x, y}; }
		};

		struct vec3
		{
			float x, y, z;
			float3 optix() const { return float3{x, y, z}; }
		};

		struct vec4
		{
			float x, y, z, w;
			float4 optix() const { return float4{x, y, z, w}; }
		};

		std::basic_istream<char>* stream_;
		char* buffer_;
		uint32_t left_{}, total_{};
		bool own_;
		uint32_t buffer_size_;

		char next_char();
		int get_pos_and_move(uint32_t count);
		void require(uint32_t count);
	};
}
