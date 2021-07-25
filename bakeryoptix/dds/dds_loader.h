#pragma once
#include <memory>
#include <string>

struct dds_loader
{
	uint32_t width{}, height{};
	std::unique_ptr<std::string> data;

	dds_loader() = default;
	dds_loader(const char* data, size_t size);
};

