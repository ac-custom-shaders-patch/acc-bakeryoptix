#include "alphanum.h"

namespace doj
{
	bool alphanum_isdigit(const char c)
	{
		return c >= '0' && c <= '9';
	}

	int alphanum_impl(const char* l, const char* r)
	{
		enum mode_t { STRING, NUMBER } mode = STRING;

		while (*l && *r)
		{
			if (mode == STRING)
			{
				char l_char, r_char;
				while ((l_char = *l) && (r_char = *r))
				{
					// check if this are digit characters
					const bool l_digit = alphanum_isdigit(l_char), r_digit = alphanum_isdigit(r_char);
					// if both characters are digits, we continue in NUMBER mode
					if (l_digit && r_digit)
					{
						mode = NUMBER;
						break;
					}
					// if only the left character is a digit, we have a result
					if (l_digit) return -1;
					// if only the right character is a digit, we have a result
					if (r_digit) return +1;
					// compute the difference of both characters
					const int diff = l_char - r_char;
					// if they differ we have a result
					if (diff != 0) return diff;
					// otherwise process the next characters
					++l;
					++r;
				}
			}
			else // mode==NUMBER
			{
				// get the left number
				unsigned long l_int = 0;
				while (*l && alphanum_isdigit(*l))
				{
					// TODO: this can overflow
					l_int = l_int * 10 + *l - '0';
					++l;
				}

				// get the right number
				unsigned long r_int = 0;
				while (*r && alphanum_isdigit(*r))
				{
					// TODO: this can overflow
					r_int = r_int * 10 + *r - '0';
					++r;
				}

				// if the difference is not equal to zero, we have a comparison result
				const long diff = l_int - r_int;
				if (diff != 0) return diff;

				// otherwise we process the next substring in STRING mode
				mode = STRING;
			}
		}

		if (*r) return -1;
		if (*l) return +1;
		return 0;
	}

	int alphanum_comp(char* l, char* r)
	{
		return alphanum_impl(l, r);
	}

	int alphanum_comp(const char* l, const char* r)
	{
		return alphanum_impl(l, r);
	}

	int alphanum_comp(char* l, const char* r)
	{
		return alphanum_impl(l, r);
	}

	int alphanum_comp(const char* l, char* r)
	{
		return alphanum_impl(l, r);
	}

	int alphanum_comp(const std::string& l, char* r)
	{
		return alphanum_impl(l.c_str(), r);
	}

	int alphanum_comp(char* l, const std::string& r)
	{
		return alphanum_impl(l, r.c_str());
	}

	int alphanum_comp(const std::string& l, const char* r)
	{
		return alphanum_impl(l.c_str(), r);
	}

	int alphanum_comp(const char* l, const std::string& r)
	{
		return alphanum_impl(l, r.c_str());
	}

	int alphanum_comp(const std::string& l, const std::string& r)
	{
		return alphanum_impl(l.c_str(), r.c_str());
	}
}
