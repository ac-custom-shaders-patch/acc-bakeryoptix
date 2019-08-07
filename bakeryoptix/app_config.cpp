#include "app_config.h"

#include <cassert>
#include <iostream>
#include <utils/std_ext.h>

const size_t NUM_RAYS = 64;
const size_t SAMPLES_PER_FACE = 6;

app_config::app_config(int argc, const char** argv)
{
	bake.filter_mode = bake::VERTEX_FILTER_LEAST_SQUARES;
	bake.regularization_weight = 0.1f;

	// defaults for tracks
	bake.num_samples = 0; // default means determine from mesh
	bake.min_samples_per_face = SAMPLES_PER_FACE;
	bake.num_rays = NUM_RAYS;
	bake.ground_upaxis = 1;
	bake.ground_scale_factor = 10000.f;
	bake.ground_offset_factor = 0.0001f;
	bake.scene_offset_scale = 0.05f;
	bake.scene_maxdistance_scale = 200.f;
	bake.use_ground_plane_blocker = false;

	load.normals_bias = 0.5f;

	save.averaging_threshold = 0.05f;
	save.brightness = 1.1f;
	save.gamma = 0.9f;
	save.opacity = 0.9f;

	// parse arguments
	for (auto i = 1; i < argc; ++i)
	{
		std::string arg(argv[i]);
		if (arg.empty()) continue;

		if (arg == "-h" || arg == "--help")
		{
			usage_and_exit(argv[0]);
		}
		else if ((arg == "-f" || arg == "--file") && i + 1 < argc)
		{
			assert( scene_filename.empty() && "multiple -f (--file) flags found when parsing command line");
			scene_filename = argv[++i];
		}
		else if ((arg == "-o" || arg == "--outfile") && i + 1 < argc)
		{
			assert(output_filename.empty() && "multiple -o (--outfile) flags found when parsing command line");
			output_filename = argv[++i];
		}
		else if ((arg == "-s" || arg == "--samples") && i + 1 < argc)
		{
			if (sscanf_s(argv[++i], "%d", &bake.num_samples) != 1)
			{
				error_and_exit(argv[0], arg, argv[i]);
			}
		}
		else if ((arg == "-t" || arg == "--samples_per_face") && i + 1 < argc)
		{
			if (sscanf_s(argv[++i], "%d", &bake.min_samples_per_face) != 1)
			{
				error_and_exit(argv[0], arg, argv[i]);
			}
		}
		else if ((arg == "-d" || arg == "--ray_distance") && i + 1 < argc)
		{
			if (sscanf_s(argv[++i], "%f", &bake.scene_offset_scale) != 1)
			{
				error_and_exit(argv[0], arg, argv[i]);
			}
		}
		else if ((arg == "-m" || arg == "--hit_distance_scale") && i + 1 < argc)
		{
			if (sscanf_s(argv[++i], "%f", &bake.scene_maxdistance_scale) != 1)
			{
				error_and_exit(argv[0], arg, argv[i]);
			}
		}
		else if ((arg == "-r" || arg == "--rays") && i + 1 < argc)
		{
			if (sscanf_s(argv[++i], "%d", &bake.num_rays) != 1)
			{
				error_and_exit(argv[0], arg, argv[i]);
			}
		}
		else if ((arg == "-n" || arg == "--normals_bias") && i + 1 < argc)
		{
			if (sscanf_s(argv[++i], "%f", &load.normals_bias) != 1)
			{
				error_and_exit(argv[0], arg, argv[i]);
			}
		}
		else if ((arg == "-g" || arg == "--ground_setup") && i + 3 < argc)
		{
			bake.use_ground_plane_blocker = true;
			if (sscanf_s(argv[++i], "%d", &bake.ground_upaxis) != 1 || (bake.ground_upaxis < 0 || bake.ground_upaxis > 5))
			{
				error_and_exit(argv[0], arg, argv[i]);
			}
			if (sscanf_s(argv[++i], "%f", &bake.ground_scale_factor) != 1)
			{
				error_and_exit(argv[0], arg, argv[i]);
			}
			if (sscanf_s(argv[++i], "%f", &bake.ground_offset_factor) != 1)
			{
				error_and_exit(argv[0], arg, argv[i]);
			}
		}
		else if ((arg == "--no_ground_plane"))
		{
			bake.use_ground_plane_blocker = false;
		}
		else if ((arg == "--no_least_squares"))
		{
			bake.filter_mode = bake::VERTEX_FILTER_AREA_BASED;
		}
		else if ((arg == "-w" || arg == "--regularization_weight") && i + 1 < argc)
		{
			if (sscanf_s(argv[++i], "%f", &bake.regularization_weight) != 1)
			{
				error_and_exit(argv[0], arg, argv[i]);
			}
			bake.regularization_weight = std::max(bake.regularization_weight, 0.f);
		}
		else if ((arg == "-a" || arg == "--gamma") && i + 1 < argc)
		{
			if (sscanf_s(argv[++i], "%f", &save.gamma) != 1)
			{
				error_and_exit(argv[0], arg, argv[i]);
			}
		}
		else if ((arg == "-b" || arg == "--brightness") && i + 1 < argc)
		{
			if (sscanf_s(argv[++i], "%f", &save.brightness) != 1)
			{
				error_and_exit(argv[0], arg, argv[i]);
			}
		}
		else if ((arg == "-o" || arg == "--opacity") && i + 1 < argc)
		{
			if (sscanf_s(argv[++i], "%f", &save.opacity) != 1)
			{
				error_and_exit(argv[0], arg, argv[i]);
			}
		}
		else if ((arg == "-v" || arg == "--averaging") && i + 1 < argc)
		{
			if (sscanf_s(argv[++i], "%f", &save.averaging_threshold) != 1)
			{
				error_and_exit(argv[0], arg, argv[i]);
			}
		}
		else if ((arg == "-p" || arg == "--pass") && i + 1 < argc)
		{
			load.exclude_blockers = std_ext::split_string(argv[++i], ",", true, true);
		}
		else if ((arg == "-x" || arg == "--exclude") && i + 1 < argc)
		{
			load.exclude_patch = std_ext::split_string(argv[++i], ",", true, true);
		}
		else
		{
			scene_filename = arg;
			if (argc == 2 && (scene_filename.string().find("/cars/") != std::string::npos || scene_filename.string().find("\\cars\\") != std::string::npos))
			{
				std::cout << "Settings for baking cars are applied\n";
				bake.num_samples = 0;
				bake.min_samples_per_face = 12;
				bake.num_rays = 256;
				bake.ground_upaxis = 1;
				bake.ground_scale_factor = 10.f;
				bake.ground_offset_factor = 0.f;
				bake.scene_offset_scale = 0.01f;
				bake.scene_maxdistance_scale = 10.f;
				bake.use_ground_plane_blocker = true;
				load.normals_bias = 0.f;
				save.averaging_threshold = 0.0001f;
				save.brightness = 1.05f;
				save.gamma = 0.8f;
				save.opacity = 1.f;
			}
		}
	}

	if (scene_filename.empty())
	{
		std::cerr << "Use options to specify one." << std::endl;
		usage_and_exit(argv[0]);
	}
}

void app_config::usage_and_exit(const char* argv0)
{
	std::cerr
			<< "Usage  : " << argv0 << " [options] <model_file/ini_file>\n"
			<< "App options:\n"
			<< "  -h  | --help                          Print this usage message\n"
			<< "  -a  | --gamma                         AO gamma (default 0.9)\n"
			<< "  -b  | --brightness                    AO brightness (default 1.1)\n"
			<< "  -o  | --opacity                       AO opacity (default 0.9)\n"
			<< "  -r  | --rays    <n>                   Number of rays per sample point for gather (default " << NUM_RAYS << ")\n"
			<< "  -s  | --samples <n>                   Number of sample points on mesh (default " << SAMPLES_PER_FACE <<
			" per face; any extra samples are based on area)\n"
			<< "  -t  | --samples_per_face <n>          Minimum number of samples per face (default " << SAMPLES_PER_FACE << ")\n"
			<< "  -d  | --ray_distance <s>              Distance offset for ray from face (default 0.05)\n"
			<< "  -v  | --averaging <s>                 Distance threshold for merging and averaging AO (default 0.0)\n"
			<< "  -n  | --normals_bias <s>              Offset normals Y to adjust shadowing (default +0.5)\n"
			<< "  -m  | --hit_distance_scale <s>        Maximum hit distance to contribute: max distance = maximum scene extent * s (default 200.0)\n"
			<< "  -x  | --exclude <name>,<name>         Exclude meshes from resulting patch\n"
			<< "  -p  | --pass <name>,<name>            Exclude meshes from occluders list\n"
			<< "  -g  | --ground_setup <axis> <s> <o>   Ground plane setup: axis(int 0,1,2,3,4,5 = +x,+y,+z,-x,-y,-z) scale(float) offset(offset) (default is disabled)\n"
			<< "        --no_ground_plane               Disable virtual ground plane\n"
			<< "  -w  | --regularization_weight <w>     Regularization weight for least squares, positive range (default 0.1)\n"
			<< "        --no_least_squares              Disable least squares filtering\n"
			<< "\n"
			<< "If no custom options are set and car model is passed, default options will be used.\n"
			<< std::endl;
	exit(1);
}

void app_config::error_and_exit(const char* argv0, const std::string& flag, const char* arg)
{
	std::cerr << "Could not parse argument: " << flag << " " << arg << std::endl;
	usage_and_exit(argv0);
}
