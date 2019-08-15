#include "cout_progress.h"

#include <iostream>
#include <sstream>
#include <utils/std_ext.h>

cout_progress::cout_progress(const size_t total_steps, bool show_steps, bool use_moment_speed)
	: start(std::chrono::high_resolution_clock::now()), total_steps(total_steps), show_steps(show_steps), use_moment_speed(use_moment_speed) { }

cout_progress::~cout_progress()
{
	erase();
}

void out_time(std::stringstream& s, double seconds)
{
	const auto minutes = floor(seconds / 60.);
	seconds = round(seconds - minutes * 60.);

	if (minutes > 0)
	{
		s << " " << minutes << " min";
	}

	if (seconds > 0 || minutes == 0)
	{
		s << " " << seconds << " s";
	}
}

std::string cout_progress::get_msg(const std::string& comment, const std::chrono::steady_clock::time_point& now)
{
	#define BRACKET if (!bracket_opened) { s << " ("; bracket_opened = true; } else { s << ", "; }

	const auto progress = float(current_step) / float(total_steps);

	std::stringstream s;
	s << std_ext::format(" %.1f%%", 100.f * progress);

	auto bracket_opened = false;

	if (show_steps)
	{
		BRACKET
		s << current_step << "/" << total_steps;
	}

	const auto time_passed = double(std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count()) / 1000.0;
	if (shown_eta || time_passed > 1. + progress && progress > 0.005f)
	{
		BRACKET
		s << "time rem.:";
		if (!use_moment_speed)
		{
			avg_speed = avg_speed + (double(current_step) / time_passed - avg_speed) / (avg_speed == 0. ? 1. : 2.4);
		}
		out_time(s, double(total_steps - current_step) / avg_speed);
		shown_eta = true;
	}

	if (!comment.empty())
	{
		BRACKET
		s << comment;
	}

	if (bracket_opened)
	{
		s << ")";
	}

	return s.str();
}

void cout_progress::erase()
{
	if (!last_length) return;
	std::cout << std::string(last_length, '\b') << std::string(last_length, ' ') << std::string(last_length, '\b');
	last_length = 0;
}

void cout_progress::report(const std::string& comment)
{
	if (total_steps < 2) return;

	const auto now = std::chrono::high_resolution_clock::now();
	const auto time_passed = double(std::chrono::duration_cast<std::chrono::milliseconds>(now - last_report).count()) / 1000.0;

	if (use_moment_speed)
	{
		const auto moment_speed = double(current_step - last_report_step) / std::max(time_passed, 0.0001);
		avg_speed = avg_speed + (moment_speed - avg_speed) / (avg_speed == 0. ? 1. : moment_speed < avg_speed ? 12. : 40.);
	}

	if (time_passed > 0.1)
	{
		erase();
		const auto msg = get_msg(comment, now);
		std::cout << msg;
		last_length = msg.size();
		last_report = now;
		last_report_step = current_step;
	}

	current_step++;
}
