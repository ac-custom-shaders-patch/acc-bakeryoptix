#pragma once
#include <vector>
#include <algorithm>
#include <iterator>

template <typename T>
std::vector<T> operator+(const std::vector<T>& A, const std::vector<T>& B)
{
	std::vector<T> AB;
	AB.reserve(A.size() + B.size());         // preallocate memory
	AB.insert(AB.end(), A.begin(), A.end()); // add A;
	AB.insert(AB.end(), B.begin(), B.end()); // add B;
	return AB;
}

template <typename T>
std::vector<T> operator-(const std::vector<T>& A, const std::vector<T>& B)
{
	std::vector<T> result;
	std::set_difference(
		A.begin(), A.end(),
		B.begin(), B.end(),
		std::back_inserter(result));
	return result;
}

template <typename T>
std::vector<T>& operator+=(std::vector<T>& A, const std::vector<T>& B)
{
	A.reserve(A.size() + B.size());        // preallocate memory without erase original data
	A.insert(A.end(), B.begin(), B.end()); // add B;
	return A;                              // here A could be named AB
}

template <typename T>
std::vector<T>& operator-=(std::vector<T>& A, const std::vector<T>& B)
{
	auto it = A.begin();
	auto it2 = B.begin();
	auto end = A.end();
	auto end2 = B.end();
	while (it != end)
	{
		while (it2 != end2)
		{
			if (*it == *it2)
			{
				it = A.erase(it);
				end = A.end();
				it2 = B.begin();
			}
			else ++it2;
		}
		++it;
		it2 = B.begin();
	}
	return A;
}

template <class Container, class Function>
auto apply(const Container& cont, Function fun)
{
	std::vector<typename
		std::result_of<Function(const typename Container::value_type&)>::type> ret;
	ret.reserve(cont.size());
	for (const auto& v : cont)
	{
		ret.push_back(fun(v));
	}
	return ret;
}