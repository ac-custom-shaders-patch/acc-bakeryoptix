#include "std_ext.h"

#include <string>
#include <regex>
#include <cstdio>

namespace std_ext
{
#define UNICODE_MAX 0x10FFFFul

	static const char* const NAMED_ENTITIES[][2] = {
		{"AElig;", u8"Æ"},
		{"Aacute;", u8"Á"},
		{"Acirc;", u8"Â"},
		{"Agrave;", u8"À"},
		{"Alpha;", u8"Α"},
		{"Aring;", u8"Å"},
		{"Atilde;", u8"Ã"},
		{"Auml;", u8"Ä"},
		{"Beta;", u8"Β"},
		{"Ccedil;", u8"Ç"},
		{"Chi;", u8"Χ"},
		{"Dagger;", u8"‡"},
		{"Delta;", u8"Δ"},
		{"ETH;", u8"Ð"},
		{"Eacute;", u8"É"},
		{"Ecirc;", u8"Ê"},
		{"Egrave;", u8"È"},
		{"Epsilon;", u8"Ε"},
		{"Eta;", u8"Η"},
		{"Euml;", u8"Ë"},
		{"Gamma;", u8"Γ"},
		{"Iacute;", u8"Í"},
		{"Icirc;", u8"Î"},
		{"Igrave;", u8"Ì"},
		{"Iota;", u8"Ι"},
		{"Iuml;", u8"Ï"},
		{"Kappa;", u8"Κ"},
		{"Lambda;", u8"Λ"},
		{"Mu;", u8"Μ"},
		{"Ntilde;", u8"Ñ"},
		{"Nu;", u8"Ν"},
		{"OElig;", u8"Œ"},
		{"Oacute;", u8"Ó"},
		{"Ocirc;", u8"Ô"},
		{"Ograve;", u8"Ò"},
		{"Omega;", u8"Ω"},
		{"Omicron;", u8"Ο"},
		{"Oslash;", u8"Ø"},
		{"Otilde;", u8"Õ"},
		{"Ouml;", u8"Ö"},
		{"Phi;", u8"Φ"},
		{"Pi;", u8"Π"},
		{"Prime;", u8"″"},
		{"Psi;", u8"Ψ"},
		{"Rho;", u8"Ρ"},
		{"Scaron;", u8"Š"},
		{"Sigma;", u8"Σ"},
		{"THORN;", u8"Þ"},
		{"Tau;", u8"Τ"},
		{"Theta;", u8"Θ"},
		{"Uacute;", u8"Ú"},
		{"Ucirc;", u8"Û"},
		{"Ugrave;", u8"Ù"},
		{"Upsilon;", u8"Υ"},
		{"Uuml;", u8"Ü"},
		{"Xi;", u8"Ξ"},
		{"Yacute;", u8"Ý"},
		{"Yuml;", u8"Ÿ"},
		{"Zeta;", u8"Ζ"},
		{"aacute;", u8"á"},
		{"acirc;", u8"â"},
		{"acute;", u8"´"},
		{"aelig;", u8"æ"},
		{"agrave;", u8"à"},
		{"alefsym;", u8"ℵ"},
		{"alpha;", u8"α"},
		{"amp;", u8"&"},
		{"and;", u8"∧"},
		{"ang;", u8"∠"},
		{"apos;", u8"'"},
		{"aring;", u8"å"},
		{"asymp;", u8"≈"},
		{"atilde;", u8"ã"},
		{"auml;", u8"ä"},
		{"bdquo;", u8"„"},
		{"beta;", u8"β"},
		{"brvbar;", u8"¦"},
		{"bull;", u8"•"},
		{"cap;", u8"∩"},
		{"ccedil;", u8"ç"},
		{"cedil;", u8"¸"},
		{"cent;", u8"¢"},
		{"chi;", u8"χ"},
		{"circ;", u8"ˆ"},
		{"clubs;", u8"♣"},
		{"colon;", u8":"},
		{"commat;", u8"@"},
		{"cong;", u8"≅"},
		{"copy;", u8"©"},
		{"crarr;", u8"↵"},
		{"cup;", u8"∪"},
		{"curren;", u8"¤"},
		{"dArr;", u8"⇓"},
		{"dagger;", u8"†"},
		{"darr;", u8"↓"},
		{"deg;", u8"°"},
		{"delta;", u8"δ"},
		{"diams;", u8"♦"},
		{"divide;", u8"÷"},
		{"eacute;", u8"é"},
		{"ecirc;", u8"ê"},
		{"egrave;", u8"è"},
		{"empty;", u8"∅"},
		{"emsp;", u8" "},
		{"ensp;", u8" "},
		{"epsilon;", u8"ε"},
		{"equiv;", u8"≡"},
		{"eta;", u8"η"},
		{"eth;", u8"ð"},
		{"euml;", u8"ë"},
		{"euro;", u8"€"},
		{"exist;", u8"∃"},
		{"excl;", u8"!"},
		{"fnof;", u8"ƒ"},
		{"forall;", u8"∀"},
		{"frac12;", u8"½"},
		{"frac14;", u8"¼"},
		{"frac34;", u8"¾"},
		{"frasl;", u8"⁄"},
		{"gamma;", u8"γ"},
		{"ge;", u8"≥"},
		{"gt;", u8">"},
		{"hArr;", u8"⇔"},
		{"harr;", u8"↔"},
		{"hearts;", u8"♥"},
		{"hellip;", u8"…"},
		{"iacute;", u8"í"},
		{"icirc;", u8"î"},
		{"iexcl;", u8"¡"},
		{"igrave;", u8"ì"},
		{"image;", u8"ℑ"},
		{"infin;", u8"∞"},
		{"int;", u8"∫"},
		{"iota;", u8"ι"},
		{"iquest;", u8"¿"},
		{"isin;", u8"∈"},
		{"iuml;", u8"ï"},
		{"kappa;", u8"κ"},
		{"lArr;", u8"⇐"},
		{"lambda;", u8"λ"},
		{"lang;", u8"〈"},
		{"laquo;", u8"«"},
		{"larr;", u8"←"},
		{"lceil;", u8"⌈"},
		{"ldquo;", u8"“"},
		{"le;", u8"≤"},
		{"lfloor;", u8"⌊"},
		{"lowast;", u8"∗"},
		{"loz;", u8"◊"},
		{"lrm;", u8"\xE2\x80\x8E"},
		{"lsaquo;", u8"‹"},
		{"lsquo;", u8"‘"},
		{"lt;", u8"<"},
		{"macr;", u8"¯"},
		{"mdash;", u8"—"},
		{"micro;", u8"µ"},
		{"middot;", u8"·"},
		{"minus;", u8"−"},
		{"mu;", u8"μ"},
		{"nabla;", u8"∇"},
		{"nbsp;", u8" "},
		{"ndash;", u8"–"},
		{"ne;", u8"≠"},
		{"ni;", u8"∋"},
		{"not;", u8"¬"},
		{"notin;", u8"∉"},
		{"nsub;", u8"⊄"},
		{"ntilde;", u8"ñ"},
		{"nu;", u8"ν"},
		{"oacute;", u8"ó"},
		{"ocirc;", u8"ô"},
		{"oelig;", u8"œ"},
		{"ograve;", u8"ò"},
		{"oline;", u8"‾"},
		{"omega;", u8"ω"},
		{"omicron;", u8"ο"},
		{"oplus;", u8"⊕"},
		{"or;", u8"∨"},
		{"ordf;", u8"ª"},
		{"ordm;", u8"º"},
		{"oslash;", u8"ø"},
		{"otilde;", u8"õ"},
		{"otimes;", u8"⊗"},
		{"ouml;", u8"ö"},
		{"para;", u8"¶"},
		{"part;", u8"∂"},
		{"permil;", u8"‰"},
		{"perp;", u8"⊥"},
		{"phi;", u8"φ"},
		{"pi;", u8"π"},
		{"piv;", u8"ϖ"},
		{"plusmn;", u8"±"},
		{"pound;", u8"£"},
		{"prime;", u8"′"},
		{"prod;", u8"∏"},
		{"prop;", u8"∝"},
		{"psi;", u8"ψ"},
		{"quot;", u8"\""},
		{"rArr;", u8"⇒"},
		{"radic;", u8"√"},
		{"rang;", u8"〉"},
		{"raquo;", u8"»"},
		{"rarr;", u8"→"},
		{"rceil;", u8"⌉"},
		{"rdquo;", u8"”"},
		{"real;", u8"ℜ"},
		{"reg;", u8"®"},
		{"rfloor;", u8"⌋"},
		{"rho;", u8"ρ"},
		{"rlm;", u8"\xE2\x80\x8F"},
		{"rsaquo;", u8"›"},
		{"rsquo;", u8"’"},
		{"sbquo;", u8"‚"},
		{"scaron;", u8"š"},
		{"sdot;", u8"⋅"},
		{"sect;", u8"§"},
		{"shy;", u8"\xC2\xAD"},
		{"sigma;", u8"σ"},
		{"sigmaf;", u8"ς"},
		{"sim;", u8"∼"},
		{"spades;", u8"♠"},
		{"sub;", u8"⊂"},
		{"sube;", u8"⊆"},
		{"sum;", u8"∑"},
		{"sup;", u8"⊃"},
		{"sup1;", u8"¹"},
		{"sup2;", u8"²"},
		{"sup3;", u8"³"},
		{"supe;", u8"⊇"},
		{"szlig;", u8"ß"},
		{"tau;", u8"τ"},
		{"there4;", u8"∴"},
		{"theta;", u8"θ"},
		{"thetasym;", u8"ϑ"},
		{"thinsp;", u8" "},
		{"thorn;", u8"þ"},
		{"tilde;", u8"˜"},
		{"times;", u8"×"},
		{"trade;", u8"™"},
		{"uArr;", u8"⇑"},
		{"uacute;", u8"ú"},
		{"uarr;", u8"↑"},
		{"ucirc;", u8"û"},
		{"ugrave;", u8"ù"},
		{"uml;", u8"¨"},
		{"upsih;", u8"ϒ"},
		{"upsilon;", u8"υ"},
		{"uuml;", u8"ü"},
		{"weierp;", u8"℘"},
		{"xi;", u8"ξ"},
		{"yacute;", u8"ý"},
		{"yen;", u8"¥"},
		{"yuml;", u8"ÿ"},
		{"zeta;", u8"ζ"},
		{"zwj;", u8"\xE2\x80\x8D"},
		{"zwnj;", u8"\xE2\x80\x8C"}
	};

	static int cmp(const void* key, const void* value)
	{
		return strncmp((const char *)key, *(const char *const *)value,
			strlen(*(const char *const *)value));
	}

	static const char* get_named_entity(const char* name)
	{
		const auto* const* entity = (const char *const *)bsearch(name,
			NAMED_ENTITIES, sizeof NAMED_ENTITIES / sizeof *NAMED_ENTITIES,
			sizeof *NAMED_ENTITIES, cmp);
		return entity ? entity[1] : nullptr;
	}

	static size_t putc_utf8(unsigned long cp, char* buffer)
	{
		const auto bytes = (unsigned char *)buffer;

		if (cp <= 0x007Ful)
		{
			bytes[0] = (unsigned char)cp;
			return 1;
		}

		if (cp <= 0x07FFul)
		{
			bytes[1] = (unsigned char)((2 << 6) | (cp & 0x3F));
			bytes[0] = (unsigned char)((6 << 5) | (cp >> 6));
			return 2;
		}

		if (cp <= 0xFFFFul)
		{
			bytes[2] = (unsigned char)((2 << 6) | (cp & 0x3F));
			bytes[1] = (unsigned char)((2 << 6) | ((cp >> 6) & 0x3F));
			bytes[0] = (unsigned char)((14 << 4) | (cp >> 12));
			return 3;
		}

		if (cp <= 0x10FFFFul)
		{
			bytes[3] = (unsigned char)((2 << 6) | (cp & 0x3F));
			bytes[2] = (unsigned char)((2 << 6) | ((cp >> 6) & 0x3F));
			bytes[1] = (unsigned char)((2 << 6) | ((cp >> 12) & 0x3F));
			bytes[0] = (unsigned char)((30 << 3) | (cp >> 18));
			return 4;
		}

		return 0;
	}

	static bool parse_entity(const char* current, char** to, const char** from)
	{
		const auto end = strchr(current, ';');
		if (!end) return false;

		if (current[1] == '#')
		{
			char* tail = nullptr;
			const auto errno_save = errno;
			const auto hex = current[2] == 'x' || current[2] == 'X';

			errno = 0;
			const auto cp = strtoul(
				current + (hex ? 3 : 2), &tail, hex ? 16 : 10);

			const auto fail = errno || tail != end || cp > UNICODE_MAX;
			errno = errno_save;
			if (fail) return false;

			*to += putc_utf8(cp, *to);
			*from = end + 1;

			return true;
		}
		else
		{
			const auto entity = get_named_entity(&current[1]);
			if (!entity) return false;

			const auto len = strlen(entity);
			memcpy(*to, entity, len);

			*to += len;
			*from = end + 1;
			return true;
		}
	}

	size_t decode_html_entities(char* dest, const char* src)
	{
		if (!src) src = dest;

		char* to = dest;
		const char* from = src;

		for (const char* current; (current = strchr(from, '&'));)
		{
			memmove(to, from, (size_t)(current - from));
			to += current - from;

			if (parse_entity(current, &to, &from)) continue;

			from = current;
			*to++ = *from++;
		}

		size_t remaining = strlen(from);

		memmove(to, from, remaining);
		to += remaining;
		*to = 0;

		return (size_t)(to - dest);
	}

	std::string decode_html_entities(const std::string& src)
	{
		std::string result;
		result.resize(src.size() + 1);
		result.resize(decode_html_entities(&result[0], &src[0]));
		return result;
	}

	void decode_html_entities_self(std::string& src)
	{
		src.resize(decode_html_entities(&src[0], nullptr));
	}

	std::string join_to_string(const std::vector<std::string>& l, const std::string& separator)
	{
		if (l.size() == 1) return l[0];
		std::string ss;
		for (auto& v : l)
		{
			if (!ss.empty()) ss += separator;
			ss += v;
		}
		return ss;
	}

	std::string extract_token(std::string::const_iterator& s, std::string::const_iterator e)
	{
		std::string::const_iterator wcard;
		for (wcard = s; wcard != e; ++wcard) if ('?' == *wcard) break;

		std::string token(s, wcard);
		for (s = wcard; s != e; ++s) if ('?' != *s) break; // treat '??' as '?' in pattern
		return token;
	}

	bool match(std::string::const_iterator patb, std::string::const_iterator pate, const std::string& input)
	{
		while (patb != pate)
		{
			// get next token from pattern, advancing patb
			std::string token = extract_token(patb, pate); // updates patb

			if (!token.empty()) // could happen if pattern begins/ends with redundant '?'
			{
				size_t submatch = input.find(token); // first submatch please

				while (std::string::npos != submatch) // while we have a submatch
				{
					if (match(patb, pate, input.substr(token.size()))) return true; // match completed successfully

					// look for later potential submatches (*backtrack*)
					submatch = input.find(token, submatch + 1);
				}
				return false; // required token not found
			}
		}
		return true; // no (remaining) pattern, always match
	}

	template <typename T>
	bool match(const std::basic_string<T>& pattern, const std::basic_string<T>& candidate, int p, int c)
	{
		if (p >= int(pattern.size()))
		{
			for (; c < int(candidate.size()); c++)
			{
				const auto f = candidate[c];
				if (f != ' ' && f != '\t') return false;
			}
			return true;
		}

		if (pattern[p] == '?')
		{
			for (; c < int(candidate.size()); c++)
			{
				if (match(pattern, candidate, p + 1, c)) return true;
			}
			return match(pattern, candidate, p + 1, c);
		}
		return pattern[p] == candidate[c] && match(pattern, candidate, p + 1, c + 1);
	}

	bool match(const std::string& pattern, const std::string& candidate)
	{
		if (pattern.size() > 1 && pattern[0] == '`' && pattern[pattern.size() - 1] == '`')
		{
			return std::regex_match(candidate, std::regex(pattern.substr(1, pattern.size() - 1)));
		}
		return match(pattern, candidate, 0, 0);
	}

	bool match(const std::wstring& pattern, const std::wstring& candidate)
	{
		if (pattern.size() > 1 && pattern[0] == '`' && pattern[pattern.size() - 1] == '`')
		{
			return std::regex_match(candidate, std::wregex(pattern.substr(1, pattern.size() - 1)));
		}
		return match(pattern, candidate, 0, 0);
	}

	void trim_self(std::string& str, const char* chars)
	{
		const auto i = str.find_first_not_of(chars);
		if (i == std::string::npos)
		{
			str = std::string();
			return;
		}
		if (i > 0) str.erase(0, i);
		const auto l = str.find_last_not_of(chars);
		if (l < str.size() - 1) str.erase(l + 1);
	}

	std::string trim(const std::string& str, const char* chars)
	{
		std::string res(str);
		trim(res, chars);
		return res;
	}

	void trim_self(std::wstring& str, const wchar_t* chars)
	{
		const auto i = str.find_first_not_of(chars);
		if (i == std::wstring::npos)
		{
			str = std::wstring();
			return;
		}
		if (i > 0) str.erase(0, i);
		const auto l = str.find_last_not_of(chars);
		if (l < str.size() - 1) str.erase(l + 1);
	}

	std::wstring trim(const std::wstring& str, const wchar_t* chars)
	{
		std::wstring res(str);
		trim_self(res, chars);
		return res;
	}

	std::vector<std::string> split_string(const std::string& input, const std::string& separator, bool skip_empty, bool trim_result)
	{
		std::vector<std::string> result;
		auto index = 0;
		const auto size = int(input.size());

		while (index < size)
		{
			auto next = input.find_first_of(separator, index);
			if (next == std::string::npos) next = size_t(size);

			auto piece = input.substr(index, next - index);
			if (trim_result) trim_self(piece);
			if (!skip_empty || !piece.empty()) result.push_back(piece);
			index = int(next) + 1;
		}

		return result;
	}

	std::vector<std::string> split_string_spaces(const std::string& input)
	{
		std::vector<std::string> result;
		auto index = 0U;
		const auto size = uint32_t(input.size());

		while (index < size)
		{
			const auto next_a = input.find(' ', index);
			const auto next_b = input.find('\t', index);
			auto next = next_a == std::string::npos ? next_b
					: next_b == std::string::npos ? next_a : std::min(next_a, next_b);
			if (next == std::string::npos) next = size_t(size);
			if (next == index)
			{
				index++;
			}
			else
			{
				result.push_back(input.substr(index, next - index));
				index = uint32_t(next) + 1;
			}
		}

		return result;
	}

	std::vector<std::wstring> split_string(const std::wstring& input, const std::wstring& separator, bool skip_empty, bool trim_result)
	{
		std::vector<std::wstring> result;
		auto index = 0;
		const auto size = int(input.size());

		while (index < size)
		{
			auto next = input.find_first_of(separator, index);
			if (next == std::wstring::npos) next = size_t(size);

			auto piece = input.substr(index, next - index);
			if (trim_result) trim_self(piece);
			if (!skip_empty || !piece.empty()) result.push_back(piece);
			index = int(next) + 1;
		}

		return result;
	}
}
