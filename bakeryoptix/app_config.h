#pragma once

#include <utils/filesystem.h>
#include <utils/load_scene.h>
#include <bake_wrap.h>

struct app_config
{
	utils::path scene_filename;
	std::string output_filename;

	bake_params bake{};
	load_params load{};
	save_params save{};

	app_config(int argc, const char** argv);
	static void usage_and_exit(const char* argv0);
	static void error_and_exit(const char* argv0, const std::string& flag, const char* arg);
};
