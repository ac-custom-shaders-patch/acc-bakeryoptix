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
std::vector<T> operator&(const std::vector<T>& A, const std::vector<T>& B)
{
	std::vector<T> ret;
	for (const auto& v : A)
	{
		if (std::find(B.begin(), B.end(), v) != B.end())
		{
			ret.push_back(v);
		}
	}
	return ret;
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
std::vector<T>& operator|=(std::vector<T>& A, const std::vector<T>& B)
{
	for (const auto& v : B)
	{
		if (std::find(A.begin(), A.end(), v) == A.end())
		{
			A.push_back(v);
		}
	}
	return A;
}

template <typename T>
std::vector<T>& operator-=(std::vector<T>& A, const std::vector<T>& B)
{
	auto it = A.begin();
	while (it != A.end())
	{
		if (std::find(B.begin(), B.end(), *it) != B.end())
		{
			it = A.erase(it);
		}
		else
		{
			++it;
		}
	}
	return A;
}

template <class Container, class Function>
auto apply(const Container& cont, Function fun)
{
	std::vector<typename std::result_of<Function(const typename Container::value_type&)>::type> ret;
	ret.reserve(cont.size());
	for (const auto& v : cont)
	{
		ret.push_back(fun(v));
	}
	return ret;
}

template <class Container, class Function>
auto where(const Container& cont, Function fun)
{
	std::vector<typename Container::value_type> ret;
	ret.reserve(cont.size());
	for (const auto& v : cont)
	{
		if (fun(v))
		{
			ret.push_back(v);
		}
	}
	return ret;
}
