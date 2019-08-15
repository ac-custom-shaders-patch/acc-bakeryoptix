#pragma once
#include <chrono>
#include <string>
#include <iostream>
#include <sstream>

struct cout_progress
{
	std::chrono::steady_clock::time_point start;
	uint32_t total_steps;
	uint32_t current_step{};
	uint32_t last_length{};

	cout_progress(uint32_t total_steps);
	~cout_progress();

	void erase();
	void report(const std::string& comment = "");

private:
	std::string get_msg(const std::string& comment) const;
};
