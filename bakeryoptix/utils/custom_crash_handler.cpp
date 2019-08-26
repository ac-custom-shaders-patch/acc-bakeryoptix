#include "custom_crash_handler.h"

#include <Dbghelp.h>
#include <psapi.h>

#include "std_ext.h"
#include <iterator>
#include <iomanip>
#include <iostream>
#include <utils/filesystem.h>

#pragma comment(lib, "dbghelp.lib")

namespace utils
{
	class SymHandler
	{
		HANDLE p;
	public:
		SymHandler(HANDLE process, char const* path = NULL, bool intrude = false)
			: p(process)
		{
			if (!SymInitialize(p, path, intrude)) throw(std::logic_error("Unable to initialize symbol handler"));
		}
		~SymHandler() { SymCleanup(p); }
	};

#ifdef _M_X64
	STACKFRAME64 init_stack_frame(CONTEXT c)
	{
		STACKFRAME64 s;
		s.AddrPC.Offset = c.Rip;
		s.AddrPC.Mode = AddrModeFlat;
		s.AddrStack.Offset = c.Rsp;
		s.AddrStack.Mode = AddrModeFlat;
		s.AddrFrame.Offset = c.Rbp;
		s.AddrFrame.Mode = AddrModeFlat;
		return s;
	}
#else
	STACKFRAME64 init_stack_frame(CONTEXT c) {
		STACKFRAME64 s;
		s.AddrPC.Offset = c.Eip;
		s.AddrPC.Mode = AddrModeFlat;
		s.AddrStack.Offset = c.Esp;
		s.AddrStack.Mode = AddrModeFlat;    
		s.AddrFrame.Offset = c.Ebp;
		s.AddrFrame.Mode = AddrModeFlat;
		return s;
	}
#endif

	void sym_options(DWORD add, DWORD remove = 0)
	{
		DWORD symOptions = SymGetOptions();
		symOptions |= add;
		symOptions &= ~remove;
		SymSetOptions(symOptions);
	}

	struct module_data
	{
		std::string image_name;
		std::string module_name;
		void* base_address;
		DWORD load_size;
	};

	typedef std::vector<module_data> ModuleList;

	class symbol
	{
		typedef IMAGEHLP_SYMBOL64 sym_type;
		sym_type* sym;
		static const int max_name_len = 1024;
	public:
		symbol(HANDLE process, DWORD64 address)
			: sym((sym_type *)::operator new(sizeof(*sym) + max_name_len))
		{
			memset(sym, '\0', sizeof(*sym) + max_name_len);
			sym->SizeOfStruct = sizeof(*sym);
			sym->MaxNameLength = max_name_len;
			DWORD64 displacement;

			if (!SymGetSymFromAddr64(process, address, &displacement, sym)) throw(std::logic_error("Bad symbol"));
		}

		std::string name() { return std::string(sym->Name); }
		std::string undecorated_name()
		{
			std::vector<char> und_name(max_name_len);
			UnDecorateSymbolName(sym->Name, &und_name[0], max_name_len, UNDNAME_COMPLETE);
			return std::string(&und_name[0], strlen(&und_name[0]));
		}
	};

	class get_mod_info
	{
		HANDLE process;
		static const int buffer_length = 4096;
	public:
		get_mod_info(HANDLE h)
			: process(h) {}

		module_data operator()(HMODULE module)
		{
			module_data ret;
			char temp[buffer_length];
			MODULEINFO mi;

			GetModuleInformation(process, module, &mi, sizeof(mi));
			ret.base_address = mi.lpBaseOfDll;
			ret.load_size = mi.SizeOfImage;

			GetModuleFileNameExA(process, module, temp, sizeof(temp));
			ret.image_name = temp;
			GetModuleBaseNameA(process, module, temp, sizeof(temp));
			ret.module_name = temp;
			std::vector<char> img(ret.image_name.begin(), ret.image_name.end());
			std::vector<char> mod(ret.module_name.begin(), ret.module_name.end());
			SymLoadModule64(process, 0, &img[0], &mod[0], (DWORD64)ret.base_address, ret.load_size);
			return ret;
		}
	};

	void* load_modules_symbols(HANDLE process, DWORD pid)
	{
		ModuleList modules;

		DWORD cbNeeded;
		std::vector<HMODULE> module_handles(1);

		EnumProcessModules(process, &module_handles[0], DWORD(module_handles.size() * sizeof(HMODULE)), &cbNeeded);
		module_handles.resize(cbNeeded / sizeof(HMODULE));
		EnumProcessModules(process, &module_handles[0], DWORD(module_handles.size() * sizeof(HMODULE)), &cbNeeded);

		std::transform(module_handles.begin(), module_handles.end(), std::back_inserter(modules), get_mod_info(process));
		return modules[0].base_address;
	}

	void make_minidump(EXCEPTION_POINTERS* e)
	{
		std::cerr << "Crash! Trying to generate nice dump: " << e << std::endl;

		SYSTEMTIME t;
		GetSystemTime(&t);
		auto filename = utils::path(std_ext::format("crash_%4d%02d%02d_%02d%02d%02d.dmp", t.wYear, t.wMonth, t.wDay, t.wHour, t.wMinute, t.wSecond));
		std::cerr << "Create dump: " << filename << std::endl;

		const auto h_file = CreateFileA(filename.string().c_str(), GENERIC_ALL, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
		if (h_file == INVALID_HANDLE_VALUE)
		{
			std::cerr << "Dumping: failed to create file" << std::endl;
			return;
		}

		MINIDUMP_EXCEPTION_INFORMATION info;
		info.ThreadId = GetCurrentThreadId();
		info.ExceptionPointers = e;
		info.ClientPointers = FALSE;

		const auto dumped = MiniDumpWriteDump(
			GetCurrentProcess(),
			GetCurrentProcessId(),
			h_file,
			MINIDUMP_TYPE(MiniDumpWithIndirectlyReferencedMemory | MiniDumpScanMemory | MiniDumpWithProcessThreadData),
			e ? &info : nullptr,
			nullptr,
			nullptr);
		std::cerr << "Dumped: " << dumped << std::endl;
		CloseHandle(h_file);
	}
}
