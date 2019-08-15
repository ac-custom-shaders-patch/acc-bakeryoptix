#pragma once
#include <chrono>
#include <string>

struct cout_progress
{
	std::chrono::steady_clock::time_point start;
	std::chrono::steady_clock::time_point last_report;
	size_t total_steps;
	size_t current_step{};
	size_t last_report_step{};
	size_t last_length{};
	double avg_speed{};
	bool show_steps{};
	bool use_moment_speed{};

	cout_progress(size_t total_steps, bool show_steps = false, bool use_moment_speed = false);
	~cout_progress();

	void erase();
	void report(const std::string& comment = "");

private:
	bool shown_eta{};
	std::string get_msg(const std::string& comment, const std::chrono::steady_clock::time_point& now);
};
