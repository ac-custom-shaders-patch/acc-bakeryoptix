#include <d3d11.h>
#include <dxgiformat.h>

#define E_TRY_AGAIN _HRESULT_TYPEDEF_(0x89ABCDEFL)
#include "utils/blob.h"

static bool use_direct_loader(const utils::blob_view& blob)
{
	if (blob.size() > 127)
	{
		if (blob.match_unsafe(0, "DDS "))
		{
			if (blob.match_unsafe(84, "DX10"))
			{
				return true;
			}

			if (blob.read<uint32_t>(24) > 1)
			{
				return false;
			}

			// If MIPs are missing, treat as basic texture
			/*const auto mipless = blob.read<uint>(28) < 2 && blob.read<uint>(12) > 4 && blob.read<uint>(16) > 4;
			if (mipless)
			{
				return tex_type::base;
			}*/

			const auto height = blob.read<uint32_t>(12);
			const auto width = blob.read<uint32_t>(16);

			// One channel with alpha 
			if (blob.read<uint32_t>(76) == 32 && blob.read<uint32_t>(80) == 2
				&& blob.read<uint32_t>(84) == 0U && blob.read<uint32_t>(88) == 8)
			{
				return true;
			}

			if (blob.match_unsafe(84, "DXT1")
				|| blob.match_unsafe(84, "DXT3")
				|| blob.match_unsafe(84, "DXT5"))
			{
				if (height % 4 == 0 && width % 4 == 0)
				{
					return true;
				}
				return false;
			}

			// RGBA8888 from Photoshop
			if (blob.read<uint32_t>(84) == 0U
				&& (blob.read<uint32_t>(88) == 32U
					&& blob.read<uint32_t>(92) == 16711680U
					&& blob.read<uint32_t>(96) == 65280U
					&& blob.read<uint32_t>(100) == 255U
					&& blob.read<uint32_t>(104) == 4278190080U))
			{
				return true;
			}
		}
	}
	return false;
}

using uint = uint32_t;

#pragma pack(push, 1)

constexpr uint DDS_MAGIC = 0x20534444; // "DDS "

struct DDS_PIXELFORMAT
{
	uint size;
	uint flags;
	uint fourCC;
	uint RGBBitCount;
	uint RBitMask;
	uint GBitMask;
	uint BBitMask;
	uint ABitMask;
};

#define DDS_FOURCC      0x00000004  // DDPF_FOURCC
#define DDS_RGB         0x00000040  // DDPF_RGB
#define DDS_LUMINANCE   0x00020000  // DDPF_LUMINANCE
#define DDS_ALPHA       0x00000002  // DDPF_ALPHA
#define DDS_BUMPDUDV    0x00080000  // DDPF_BUMPDUDV

#define MAKEFOURCC_bc7(ch0, ch1, ch2, ch3) \
		(static_cast<uint>(static_cast<uint8_t>(ch0)) \
		| (static_cast<uint>(static_cast<uint8_t>(ch1)) << 8) \
		| (static_cast<uint>(static_cast<uint8_t>(ch2)) << 16) \
		| (static_cast<uint>(static_cast<uint8_t>(ch3)) << 24))

#define DDS_HEADER_FLAGS_VOLUME         0x00800000  // DDSD_DEPTH
#define DDS_HEIGHT 0x00000002 // DDSD_HEIGHT
#define DDS_CUBEMAP_POSITIVEX 0x00000600 // DDSCAPS2_CUBEMAP | DDSCAPS2_CUBEMAP_POSITIVEX
#define DDS_CUBEMAP_NEGATIVEX 0x00000a00 // DDSCAPS2_CUBEMAP | DDSCAPS2_CUBEMAP_NEGATIVEX
#define DDS_CUBEMAP_POSITIVEY 0x00001200 // DDSCAPS2_CUBEMAP | DDSCAPS2_CUBEMAP_POSITIVEY
#define DDS_CUBEMAP_NEGATIVEY 0x00002200 // DDSCAPS2_CUBEMAP | DDSCAPS2_CUBEMAP_NEGATIVEY
#define DDS_CUBEMAP_POSITIVEZ 0x00004200 // DDSCAPS2_CUBEMAP | DDSCAPS2_CUBEMAP_POSITIVEZ
#define DDS_CUBEMAP_NEGATIVEZ 0x00008200 // DDSCAPS2_CUBEMAP | DDSCAPS2_CUBEMAP_NEGATIVEZ
#define DDS_CUBEMAP_ALLFACES ( DDS_CUBEMAP_POSITIVEX | DDS_CUBEMAP_NEGATIVEX |\
		DDS_CUBEMAP_POSITIVEY | DDS_CUBEMAP_NEGATIVEY |\
		DDS_CUBEMAP_POSITIVEZ | DDS_CUBEMAP_NEGATIVEZ )
#define DDS_CUBEMAP 0x00000200 // DDSCAPS2_CUBEMAP

struct DDS_HEADER
{
	uint size;
	uint flags;
	uint height;
	uint width;
	[[maybe_unused]] uint pitch_or_linear_size;
	uint depth; // only if DDS_HEADER_FLAGS_VOLUME is set in flags
	uint mip_map_count;
	[[maybe_unused]] uint reserved1[11];
	DDS_PIXELFORMAT ddspf;
	[[maybe_unused]] uint caps;
	uint caps2;
	[[maybe_unused]] uint caps3;
	[[maybe_unused]] uint caps4;
	[[maybe_unused]] uint reserved2;
};

struct DDS_HEADER_DXT10
{
	DXGI_FORMAT dxgi_format;
	uint resource_dimension;
	uint misc_flag; // see D3D11_RESOURCE_MISC_FLAG
	uint array_size;
	[[maybe_unused]] uint misc_flags2; // see DDS_MISC_FLAGS2
};
#pragma pack(pop)

static HRESULT load_texture_data_from_memory(const uint8_t* data, size_t data_size, const DDS_HEADER** header, const DDS_HEADER_DXT10** dx10_header,
	utils::blob_view& out_data) noexcept
{
	if (!header)
	{
		return E_POINTER;
	}

	if (data_size > UINT32_MAX)
	{
		return E_FAIL;
	}

	if (data_size < sizeof(uint) + sizeof(DDS_HEADER))
	{
		return E_FAIL;
	}

	// DDS files always start with the same magic number ("DDS ")
	{
		const auto magic_number = *reinterpret_cast<const uint*>(data);
		if (magic_number != DDS_MAGIC)
		{
			return E_FAIL;
		}
	}

	const auto hdr = reinterpret_cast<const DDS_HEADER*>(data + sizeof(uint));

	// Verify header to validate DDS file
	if (hdr->size != sizeof(DDS_HEADER) || hdr->ddspf.size != sizeof(DDS_PIXELFORMAT))
	{
		return E_FAIL;
	}

	// Check for DX10 extension
	auto dxt10_header = false;
	if ((hdr->ddspf.flags & DDS_FOURCC) != 0 && MAKEFOURCC_bc7('D', 'X', '1', '0') == hdr->ddspf.fourCC)
	{
		// Must be long enough for both headers and magic value
		if (data_size < (sizeof(uint) + sizeof(DDS_HEADER) + sizeof(DDS_HEADER_DXT10)))
		{
			return E_FAIL;
		}

		dxt10_header = true;

		if (dx10_header)
		{
			*dx10_header = reinterpret_cast<const DDS_HEADER_DXT10*>(data + sizeof(uint) + sizeof(DDS_HEADER));
		}
	}
	else if (dx10_header)
	{
		*dx10_header = nullptr;
	}

	// setup the pointers in the process request
	*header = hdr;
	const auto offset = sizeof(uint) + sizeof(DDS_HEADER) + (dxt10_header ? sizeof(DDS_HEADER_DXT10) : 0U);
	out_data = utils::blob_view((const char*)data + offset, data_size - offset);
	return S_OK;
}

static size_t bits_per_pixel(DXGI_FORMAT fmt) noexcept
{
	switch (fmt)
	{
	case DXGI_FORMAT_R32G32B32A32_TYPELESS:
	case DXGI_FORMAT_R32G32B32A32_FLOAT:
	case DXGI_FORMAT_R32G32B32A32_UINT:
	case DXGI_FORMAT_R32G32B32A32_SINT:
		return 128;

	case DXGI_FORMAT_R32G32B32_TYPELESS:
	case DXGI_FORMAT_R32G32B32_FLOAT:
	case DXGI_FORMAT_R32G32B32_UINT:
	case DXGI_FORMAT_R32G32B32_SINT:
		return 96;

	case DXGI_FORMAT_R16G16B16A16_TYPELESS:
	case DXGI_FORMAT_R16G16B16A16_FLOAT:
	case DXGI_FORMAT_R16G16B16A16_UNORM:
	case DXGI_FORMAT_R16G16B16A16_UINT:
	case DXGI_FORMAT_R16G16B16A16_SNORM:
	case DXGI_FORMAT_R16G16B16A16_SINT:
	case DXGI_FORMAT_R32G32_TYPELESS:
	case DXGI_FORMAT_R32G32_FLOAT:
	case DXGI_FORMAT_R32G32_UINT:
	case DXGI_FORMAT_R32G32_SINT:
	case DXGI_FORMAT_R32G8X24_TYPELESS:
	case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:
	case DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS:
	case DXGI_FORMAT_X32_TYPELESS_G8X24_UINT:
	case DXGI_FORMAT_Y416:
	case DXGI_FORMAT_Y210:
	case DXGI_FORMAT_Y216:
		return 64;

	case DXGI_FORMAT_R10G10B10A2_TYPELESS:
	case DXGI_FORMAT_R10G10B10A2_UNORM:
	case DXGI_FORMAT_R10G10B10A2_UINT:
	case DXGI_FORMAT_R11G11B10_FLOAT:
	case DXGI_FORMAT_R8G8B8A8_TYPELESS:
	case DXGI_FORMAT_R8G8B8A8_UNORM:
	case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
	case DXGI_FORMAT_R8G8B8A8_UINT:
	case DXGI_FORMAT_R8G8B8A8_SNORM:
	case DXGI_FORMAT_R8G8B8A8_SINT:
	case DXGI_FORMAT_R16G16_TYPELESS:
	case DXGI_FORMAT_R16G16_FLOAT:
	case DXGI_FORMAT_R16G16_UNORM:
	case DXGI_FORMAT_R16G16_UINT:
	case DXGI_FORMAT_R16G16_SNORM:
	case DXGI_FORMAT_R16G16_SINT:
	case DXGI_FORMAT_R32_TYPELESS:
	case DXGI_FORMAT_D32_FLOAT:
	case DXGI_FORMAT_R32_FLOAT:
	case DXGI_FORMAT_R32_UINT:
	case DXGI_FORMAT_R32_SINT:
	case DXGI_FORMAT_R24G8_TYPELESS:
	case DXGI_FORMAT_D24_UNORM_S8_UINT:
	case DXGI_FORMAT_R24_UNORM_X8_TYPELESS:
	case DXGI_FORMAT_X24_TYPELESS_G8_UINT:
	case DXGI_FORMAT_R9G9B9E5_SHAREDEXP:
	case DXGI_FORMAT_R8G8_B8G8_UNORM:
	case DXGI_FORMAT_G8R8_G8B8_UNORM:
	case DXGI_FORMAT_B8G8R8A8_UNORM:
	case DXGI_FORMAT_B8G8R8X8_UNORM:
	case DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM:
	case DXGI_FORMAT_B8G8R8A8_TYPELESS:
	case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
	case DXGI_FORMAT_B8G8R8X8_TYPELESS:
	case DXGI_FORMAT_B8G8R8X8_UNORM_SRGB:
	case DXGI_FORMAT_AYUV:
	case DXGI_FORMAT_Y410:
	case DXGI_FORMAT_YUY2:
		return 32;

	case DXGI_FORMAT_P010:
	case DXGI_FORMAT_P016:
		#if (_WIN32_WINNT >= _WIN32_WINNT_WIN10)
	case DXGI_FORMAT_V408:
		#endif
		return 24;

	case DXGI_FORMAT_R8G8_TYPELESS:
	case DXGI_FORMAT_R8G8_UNORM:
	case DXGI_FORMAT_R8G8_UINT:
	case DXGI_FORMAT_R8G8_SNORM:
	case DXGI_FORMAT_R8G8_SINT:
	case DXGI_FORMAT_R16_TYPELESS:
	case DXGI_FORMAT_R16_FLOAT:
	case DXGI_FORMAT_D16_UNORM:
	case DXGI_FORMAT_R16_UNORM:
	case DXGI_FORMAT_R16_UINT:
	case DXGI_FORMAT_R16_SNORM:
	case DXGI_FORMAT_R16_SINT:
	case DXGI_FORMAT_B5G6R5_UNORM:
	case DXGI_FORMAT_B5G5R5A1_UNORM:
	case DXGI_FORMAT_A8P8:
	case DXGI_FORMAT_B4G4R4A4_UNORM:
		#if (_WIN32_WINNT >= _WIN32_WINNT_WIN10)
	case DXGI_FORMAT_P208:
	case DXGI_FORMAT_V208:
		#endif
		return 16;

	case DXGI_FORMAT_NV12:
	case DXGI_FORMAT_420_OPAQUE:
	case DXGI_FORMAT_NV11:
		return 12;

	case DXGI_FORMAT_R8_TYPELESS:
	case DXGI_FORMAT_R8_UNORM:
	case DXGI_FORMAT_R8_UINT:
	case DXGI_FORMAT_R8_SNORM:
	case DXGI_FORMAT_R8_SINT:
	case DXGI_FORMAT_A8_UNORM:
	case DXGI_FORMAT_BC2_TYPELESS:
	case DXGI_FORMAT_BC2_UNORM:
	case DXGI_FORMAT_BC2_UNORM_SRGB:
	case DXGI_FORMAT_BC3_TYPELESS:
	case DXGI_FORMAT_BC3_UNORM:
	case DXGI_FORMAT_BC3_UNORM_SRGB:
	case DXGI_FORMAT_BC5_TYPELESS:
	case DXGI_FORMAT_BC5_UNORM:
	case DXGI_FORMAT_BC5_SNORM:
	case DXGI_FORMAT_BC6H_TYPELESS:
	case DXGI_FORMAT_BC6H_UF16:
	case DXGI_FORMAT_BC6H_SF16:
	case DXGI_FORMAT_BC7_TYPELESS:
	case DXGI_FORMAT_BC7_UNORM:
	case DXGI_FORMAT_BC7_UNORM_SRGB:
	case DXGI_FORMAT_AI44:
	case DXGI_FORMAT_IA44:
	case DXGI_FORMAT_P8:
		return 8;

	case DXGI_FORMAT_R1_UNORM:
		return 1;

	case DXGI_FORMAT_BC1_TYPELESS:
	case DXGI_FORMAT_BC1_UNORM:
	case DXGI_FORMAT_BC1_UNORM_SRGB:
	case DXGI_FORMAT_BC4_TYPELESS:
	case DXGI_FORMAT_BC4_UNORM:
	case DXGI_FORMAT_BC4_SNORM:
		return 4;

	case DXGI_FORMAT_UNKNOWN:
	case DXGI_FORMAT_FORCE_UINT:
	default:
		return 0;
	}
}


#define ISBITMASK(r, g, b, a) (ddpf.RBitMask == r && ddpf.GBitMask == g && ddpf.BBitMask == b && ddpf.ABitMask == a)

static DXGI_FORMAT get_dxgi_format(const DDS_PIXELFORMAT& ddpf) noexcept
{
	if (ddpf.flags & DDS_RGB)
	{
		// Note that sRGB formats are written using the "DX10" extended header
		// ReSharper disable once CppDefaultCaseNotHandledInSwitchStatement
		switch (ddpf.RGBBitCount)
		{
		case 32:
			if (ISBITMASK(0x000000ff, 0x0000ff00, 0x00ff0000, 0xff000000))
			{
				return DXGI_FORMAT_R8G8B8A8_UNORM;
			}

			if (ISBITMASK(0x00ff0000, 0x0000ff00, 0x000000ff, 0xff000000))
			{
				return DXGI_FORMAT_B8G8R8A8_UNORM;
			}

			if (ISBITMASK(0x00ff0000, 0x0000ff00, 0x000000ff, 0))
			{
				return DXGI_FORMAT_B8G8R8X8_UNORM;
			}

		// No DXGI format maps to ISBITMASK(0x000000ff,0x0000ff00,0x00ff0000,0) aka D3DFMT_X8B8G8R8

		// Note that many common DDS reader/writers (including D3DX) swap the
		// the RED/BLUE masks for 10:10:10:2 formats. We assume
		// below that the 'backwards' header mask is being used since it is most
		// likely written by D3DX. The more robust solution is to use the 'DX10'
		// header extension and specify the DXGI_FORMAT_R10G10B10A2_UNORM format directly

		// For 'correct' writers, this should be 0x000003ff,0x000ffc00,0x3ff00000 for RGB data
			if (ISBITMASK(0x3ff00000, 0x000ffc00, 0x000003ff, 0xc0000000))
			{
				return DXGI_FORMAT_R10G10B10A2_UNORM;
			}

		// No DXGI format maps to ISBITMASK(0x000003ff,0x000ffc00,0x3ff00000,0xc0000000) aka D3DFMT_A2R10G10B10

			if (ISBITMASK(0x0000ffff, 0xffff0000, 0, 0))
			{
				return DXGI_FORMAT_R16G16_UNORM;
			}

			if (ISBITMASK(0xffffffff, 0, 0, 0))
			{
				// Only 32-bit color channel format in D3D9 was R32F
				return DXGI_FORMAT_R32_FLOAT; // D3DX writes this out as a FourCC of 114
			}
			break;

		case 24:
			// No 24bpp DXGI formats aka D3DFMT_R8G8B8
			break;

		case 16:
			if (ISBITMASK(0x7c00, 0x03e0, 0x001f, 0x8000))
			{
				return DXGI_FORMAT_B5G5R5A1_UNORM;
			}
			if (ISBITMASK(0xf800, 0x07e0, 0x001f, 0))
			{
				return DXGI_FORMAT_B5G6R5_UNORM;
			}

		// No DXGI format maps to ISBITMASK(0x7c00,0x03e0,0x001f,0) aka D3DFMT_X1R5G5B5

			if (ISBITMASK(0x0f00, 0x00f0, 0x000f, 0xf000))
			{
				return (DXGI_FORMAT)DXGI_FORMAT_B4G4R4A4_UNORM;
			}

		// NVTT versions 1.x wrote this as RGB instead of LUMINANCE
			if (ISBITMASK(0x00ff, 0, 0, 0xff00))
			{
				return DXGI_FORMAT_R8G8_UNORM;
			}
			if (ISBITMASK(0xffff, 0, 0, 0))
			{
				return DXGI_FORMAT_R16_UNORM;
			}

		// No DXGI format maps to ISBITMASK(0x0f00,0x00f0,0x000f,0) aka D3DFMT_X4R4G4B4

		// No 3:3:2:8 or paletted DXGI formats aka D3DFMT_A8R3G3B2, D3DFMT_A8P8, etc.
			break;

		case 8:
			// NVTT versions 1.x wrote this as RGB instead of LUMINANCE
			if (ISBITMASK(0xff, 0, 0, 0))
			{
				return DXGI_FORMAT_R8_UNORM;
			}

		// No 3:3:2 or paletted DXGI formats aka D3DFMT_R3G3B2, D3DFMT_P8
			break;
		}
	}
	else if (ddpf.flags & DDS_LUMINANCE)
	{
		// ReSharper disable once CppDefaultCaseNotHandledInSwitchStatement
		switch (ddpf.RGBBitCount)
		{
		case 16:
			if (ISBITMASK(0xffff, 0, 0, 0))
			{
				return DXGI_FORMAT_R16_UNORM; // D3DX10/11 writes this out as DX10 extension
			}
			if (ISBITMASK(0x00ff, 0, 0, 0xff00))
			{
				return DXGI_FORMAT_R8G8_UNORM; // D3DX10/11 writes this out as DX10 extension
			}
			break;

		case 8:
			if (ISBITMASK(0xff, 0, 0, 0))
			{
				return DXGI_FORMAT_R8_UNORM; // D3DX10/11 writes this out as DX10 extension
			}

		// No DXGI format maps to ISBITMASK(0x0f,0,0,0xf0) aka D3DFMT_A4L4

			if (ISBITMASK(0x00ff, 0, 0, 0xff00))
			{
				return DXGI_FORMAT_R8G8_UNORM; // Some DDS writers assume the bitcount should be 8 instead of 16
			}
			break;
		}
	}
	else if (ddpf.flags & DDS_ALPHA)
	{
		if (8 == ddpf.RGBBitCount)
		{
			return DXGI_FORMAT_A8_UNORM;
		}
	}
	else if (ddpf.flags & DDS_BUMPDUDV)
	{
		// ReSharper disable once CppDefaultCaseNotHandledInSwitchStatement
		switch (ddpf.RGBBitCount)
		{
		case 32:
			if (ISBITMASK(0x000000ff, 0x0000ff00, 0x00ff0000, 0xff000000))
			{
				return DXGI_FORMAT_R8G8B8A8_SNORM; // D3DX10/11 writes this out as DX10 extension
			}
			if (ISBITMASK(0x0000ffff, 0xffff0000, 0, 0))
			{
				return DXGI_FORMAT_R16G16_SNORM; // D3DX10/11 writes this out as DX10 extension
			}

		// No DXGI format maps to ISBITMASK(0x3ff00000, 0x000ffc00, 0x000003ff, 0xc0000000) aka D3DFMT_A2W10V10U10
			break;

		case 16:
			if (ISBITMASK(0x00ff, 0xff00, 0, 0))
			{
				return DXGI_FORMAT_R8G8_SNORM; // D3DX10/11 writes this out as DX10 extension
			}
			break;
		}

		// No DXGI format maps to DDPF_BUMPLUMINANCE aka D3DFMT_L6V5U5, D3DFMT_X8L8V8U8
	}
	else if (ddpf.flags & DDS_FOURCC)
	{
		if (MAKEFOURCC_bc7('D', 'X', 'T', '1') == ddpf.fourCC)
		{
			return DXGI_FORMAT_BC1_UNORM;
		}
		if (MAKEFOURCC_bc7('D', 'X', 'T', '3') == ddpf.fourCC)
		{
			return DXGI_FORMAT_BC2_UNORM;
		}
		if (MAKEFOURCC_bc7('D', 'X', 'T', '5') == ddpf.fourCC)
		{
			return DXGI_FORMAT_BC3_UNORM;
		}

		// While pre-multiplied alpha isn't directly supported by the DXGI formats,
		// they are basically the same as these BC formats so they can be mapped
		if (MAKEFOURCC_bc7('D', 'X', 'T', '2') == ddpf.fourCC)
		{
			return DXGI_FORMAT_BC2_UNORM;
		}
		if (MAKEFOURCC_bc7('D', 'X', 'T', '4') == ddpf.fourCC)
		{
			return DXGI_FORMAT_BC3_UNORM;
		}

		if (MAKEFOURCC_bc7('A', 'T', 'I', '1') == ddpf.fourCC)
		{
			return DXGI_FORMAT_BC4_UNORM;
		}
		if (MAKEFOURCC_bc7('B', 'C', '4', 'U') == ddpf.fourCC)
		{
			return DXGI_FORMAT_BC4_UNORM;
		}
		if (MAKEFOURCC_bc7('B', 'C', '4', 'S') == ddpf.fourCC)
		{
			return DXGI_FORMAT_BC4_SNORM;
		}

		if (MAKEFOURCC_bc7('A', 'T', 'I', '2') == ddpf.fourCC)
		{
			return DXGI_FORMAT_BC5_UNORM;
		}
		if (MAKEFOURCC_bc7('B', 'C', '5', 'U') == ddpf.fourCC)
		{
			return DXGI_FORMAT_BC5_UNORM;
		}
		if (MAKEFOURCC_bc7('B', 'C', '5', 'S') == ddpf.fourCC)
		{
			return DXGI_FORMAT_BC5_SNORM;
		}

		// BC6H and BC7 are written using the "DX10" extended header

		if (MAKEFOURCC_bc7('R', 'G', 'B', 'G') == ddpf.fourCC)
		{
			return DXGI_FORMAT_R8G8_B8G8_UNORM;
		}
		if (MAKEFOURCC_bc7('G', 'R', 'G', 'B') == ddpf.fourCC)
		{
			return DXGI_FORMAT_G8R8_G8B8_UNORM;
		}

		if (MAKEFOURCC_bc7('Y', 'U', 'Y', '2') == ddpf.fourCC)
		{
			return (DXGI_FORMAT)DXGI_FORMAT_YUY2;
		}

		// Check for D3DFORMAT enums being set here
		// ReSharper disable once CppDefaultCaseNotHandledInSwitchStatement
		switch (ddpf.fourCC)
		{
		case 36: // D3DFMT_A16B16G16R16
			return DXGI_FORMAT_R16G16B16A16_UNORM;

		case 110: // D3DFMT_Q16W16V16U16
			return DXGI_FORMAT_R16G16B16A16_SNORM;

		case 111: // D3DFMT_R16F
			return DXGI_FORMAT_R16_FLOAT;

		case 112: // D3DFMT_G16R16F
			return DXGI_FORMAT_R16G16_FLOAT;

		case 113: // D3DFMT_A16B16G16R16F
			return DXGI_FORMAT_R16G16B16A16_FLOAT;

		case 114: // D3DFMT_R32F
			return DXGI_FORMAT_R32_FLOAT;

		case 115: // D3DFMT_G32R32F
			return DXGI_FORMAT_R32G32_FLOAT;

		case 116: // D3DFMT_A32B32G32R32F
			return DXGI_FORMAT_R32G32B32A32_FLOAT;

		// No DXGI format maps to D3DFMT_CxV8U8
		}
	}
	return DXGI_FORMAT_UNKNOWN;
}
#undef ISBITMASK


HRESULT get_surface_info(size_t width, size_t height, DXGI_FORMAT fmt, size_t* out_num_bytes, size_t* out_row_bytes, size_t* out_num_rows) noexcept
{
	uint64_t num_bytes;
	uint64_t row_bytes;
	uint64_t num_rows;

	auto bc = false;
	auto packed = false;
	auto planar = false;
	size_t bpe = 0;
	switch (fmt)
	{
	case DXGI_FORMAT_BC1_TYPELESS:
	case DXGI_FORMAT_BC1_UNORM:
	case DXGI_FORMAT_BC1_UNORM_SRGB:
	case DXGI_FORMAT_BC4_TYPELESS:
	case DXGI_FORMAT_BC4_UNORM:
	case DXGI_FORMAT_BC4_SNORM:
		bc = true;
		bpe = 8;
		break;

	case DXGI_FORMAT_BC2_TYPELESS:
	case DXGI_FORMAT_BC2_UNORM:
	case DXGI_FORMAT_BC2_UNORM_SRGB:
	case DXGI_FORMAT_BC3_TYPELESS:
	case DXGI_FORMAT_BC3_UNORM:
	case DXGI_FORMAT_BC3_UNORM_SRGB:
	case DXGI_FORMAT_BC5_TYPELESS:
	case DXGI_FORMAT_BC5_UNORM:
	case DXGI_FORMAT_BC5_SNORM:
	case DXGI_FORMAT_BC6H_TYPELESS:
	case DXGI_FORMAT_BC6H_UF16:
	case DXGI_FORMAT_BC6H_SF16:
	case DXGI_FORMAT_BC7_TYPELESS:
	case DXGI_FORMAT_BC7_UNORM:
	case DXGI_FORMAT_BC7_UNORM_SRGB:
		bc = true;
		bpe = 16;
		break;

	case DXGI_FORMAT_R8G8_B8G8_UNORM:
	case DXGI_FORMAT_G8R8_G8B8_UNORM:
	case DXGI_FORMAT_YUY2:
		packed = true;
		bpe = 4;
		break;

	case DXGI_FORMAT_Y210:
	case DXGI_FORMAT_Y216:
		packed = true;
		bpe = 8;
		break;

	case DXGI_FORMAT_NV12:
	case DXGI_FORMAT_420_OPAQUE:
		if ((height % 2) != 0)
		{
			// Requires a height alignment of 2.
			return E_INVALIDARG;
		}
		planar = true;
		bpe = 2;
		break;

		#if (_WIN32_WINNT >= _WIN32_WINNT_WIN10)

	case DXGI_FORMAT_P208:
		planar = true;
		bpe = 2;
		break;

		#endif

	case DXGI_FORMAT_P010:
	case DXGI_FORMAT_P016:
		if ((height % 2) != 0)
		{
			// Requires a height alignment of 2.
			return E_INVALIDARG;
		}
		planar = true;
		bpe = 4;
		break;

	default:
		break;
	}

	if (bc)
	{
		uint64_t numBlocksWide = 0;
		if (width > 0)
		{
			numBlocksWide = std::max<uint64_t>(1u, (uint64_t(width) + 3u) / 4u);
		}
		uint64_t numBlocksHigh = 0;
		if (height > 0)
		{
			numBlocksHigh = std::max<uint64_t>(1u, (uint64_t(height) + 3u) / 4u);
		}
		row_bytes = numBlocksWide * bpe;
		num_rows = numBlocksHigh;
		num_bytes = row_bytes * numBlocksHigh;
	}
	else if (packed)
	{
		row_bytes = ((uint64_t(width) + 1u) >> 1) * bpe;
		num_rows = uint64_t(height);
		num_bytes = row_bytes * height;
	}
	else if (fmt == DXGI_FORMAT_NV11)
	{
		row_bytes = ((uint64_t(width) + 3u) >> 2) * 4u;
		num_rows = uint64_t(height) * 2u; // Direct3D makes this simplifying assumption, although it is larger than the 4:1:1 data
		num_bytes = row_bytes * num_rows;
	}
	else if (planar)
	{
		row_bytes = ((uint64_t(width) + 1u) >> 1) * bpe;
		num_bytes = (row_bytes * uint64_t(height)) + ((row_bytes * uint64_t(height) + 1u) >> 1);
		num_rows = height + ((uint64_t(height) + 1u) >> 1);
	}
	else
	{
		const size_t bpp = bits_per_pixel(fmt);
		if (!bpp) return E_INVALIDARG;
		row_bytes = (uint64_t(width) * bpp + 7u) / 8u; // round up to nearest byte
		num_rows = uint64_t(height);
		num_bytes = row_bytes * height;
	}

	if (out_num_bytes)
	{
		*out_num_bytes = num_bytes;
	}
	if (out_row_bytes)
	{
		*out_row_bytes = row_bytes;
	}
	if (out_num_rows)
	{
		*out_num_rows = num_rows;
	}
	return S_OK;
}

HRESULT fill_init_data(size_t width, size_t height, size_t depth, size_t final_mip_count, size_t mip_count, size_t array_size,
	DXGI_FORMAT format, size_t max_size, size_t bit_size, const uint8_t* bit_data, size_t& twidth, size_t& theight, size_t& tdepth,
	size_t& skip_mip, D3D11_SUBRESOURCE_DATA* init_data) noexcept
{
	if (!bit_data || !init_data)
	{
		return E_POINTER;
	}

	skip_mip = 0;
	twidth = 0;
	theight = 0;
	tdepth = 0;

	size_t num_bytes = 0;
	size_t row_bytes = 0;
	auto src_bits = bit_data;
	auto src_bits_prev = src_bits;
	const auto end_bits = bit_data + bit_size;

	size_t index = 0;
	for (size_t j = 0; j < array_size; j++)
	{
		auto w = width;
		auto h = height;
		auto d = depth;
		for (size_t i = 0; i < final_mip_count; i++)
		{
			if (i >= mip_count)
			{
				init_data[index].pSysMem = src_bits_prev;
				init_data[index].SysMemPitch = uint(row_bytes /= 2);
				init_data[index].SysMemSlicePitch = uint(num_bytes /= 4);
				++index;
				continue;
			}

			HRESULT hr = get_surface_info(w, h, format, &num_bytes, &row_bytes, nullptr);
			if (FAILED(hr)) return hr;

			if (num_bytes > UINT32_MAX || row_bytes > UINT32_MAX)
				return HRESULT_FROM_WIN32(ERROR_ARITHMETIC_OVERFLOW);

			if ((final_mip_count <= 1) || !max_size || (w <= max_size && h <= max_size && d <= max_size))
			{
				if (!twidth)
				{
					twidth = w;
					theight = h;
					tdepth = d;
				}

				_Analysis_assume_(index < mip_count* array_size);
				init_data[index].pSysMem = src_bits;
				init_data[index].SysMemPitch = uint(row_bytes);
				init_data[index].SysMemSlicePitch = uint(num_bytes);
				++index;
			}
			else if (!j)
			{
				// Count number of skipped mipmaps (first item only)
				++skip_mip;
			}

			if (src_bits + num_bytes * d > end_bits)
			{
				return HRESULT_FROM_WIN32(ERROR_HANDLE_EOF);
			}

			src_bits_prev = src_bits;
			src_bits += num_bytes * d;

			w = w >> 1;
			h = h >> 1;
			d = d >> 1;
			if (w == 0)
			{
				w = 1;
			}
			if (h == 0)
			{
				h = 1;
			}
			if (d == 0)
			{
				d = 1;
			}
		}
	}

	return index > 0 ? S_OK : E_FAIL;
}

HRESULT create_resources(ID3D11Device* device, uint res_dim, size_t width, size_t height, size_t depth, size_t mip_count, size_t array_size,
	DXGI_FORMAT format, D3D11_USAGE usage, uint bind_flags, uint cpu_access_flags, uint misc_flags,
	bool is_cube_map, const D3D11_SUBRESOURCE_DATA* init_data, ID3D11Resource** texture, ID3D11ShaderResourceView** view) noexcept
{
	auto hr = E_FAIL;

	// ReSharper disable once CppDefaultCaseNotHandledInSwitchStatement
	switch (res_dim) // NOLINT(hicpp-multiway-paths-covered)
	{
	case D3D11_RESOURCE_DIMENSION_TEXTURE1D:
	{
		D3D11_TEXTURE1D_DESC desc = {};
		desc.Width = uint(width);
		desc.MipLevels = uint(mip_count);
		desc.ArraySize = uint(array_size);
		desc.Format = format;
		desc.Usage = usage;
		desc.BindFlags = bind_flags;
		desc.CPUAccessFlags = cpu_access_flags;
		desc.MiscFlags = misc_flags & ~static_cast<uint>(D3D11_RESOURCE_MISC_TEXTURECUBE);

		ID3D11Texture1D* tex = nullptr;
		hr = device->CreateTexture1D(&desc, init_data, &tex);
		if (FAILED(hr) || !tex) return E_TRY_AGAIN;

		if (view)
		{
			D3D11_SHADER_RESOURCE_VIEW_DESC SRVDesc = {};
			SRVDesc.Format = format;
			if (array_size > 1)
			{
				SRVDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE1DARRAY;
				SRVDesc.Texture1DArray.MipLevels = mip_count == 0U ? uint(-1) : desc.MipLevels;
				SRVDesc.Texture1DArray.ArraySize = uint(array_size);
			}
			else
			{
				SRVDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE1D;
				SRVDesc.Texture1D.MipLevels = mip_count == 0U ? uint(-1) : desc.MipLevels;
			}

			hr = device->CreateShaderResourceView(tex, &SRVDesc, view);
			if (FAILED(hr))
			{
				tex->Release();
				return hr;
			}
		}

		if (texture)
		{
			*texture = tex;
		}
		else
		{
			tex->Release();
		}
	}
	break;

	case D3D11_RESOURCE_DIMENSION_TEXTURE2D:
	{
		D3D11_TEXTURE2D_DESC desc = {};
		desc.Width = uint(width);
		desc.Height = uint(height);
		desc.MipLevels = uint(mip_count);
		desc.ArraySize = uint(array_size);
		desc.Format = format;
		desc.SampleDesc.Count = 1;
		desc.SampleDesc.Quality = 0;
		desc.Usage = usage;
		desc.BindFlags = bind_flags;
		desc.CPUAccessFlags = cpu_access_flags;
		if (is_cube_map)
		{
			desc.MiscFlags = misc_flags | D3D11_RESOURCE_MISC_TEXTURECUBE;
		}
		else
		{
			desc.MiscFlags = misc_flags & ~static_cast<uint>(D3D11_RESOURCE_MISC_TEXTURECUBE);
		}

		ID3D11Texture2D* tex = nullptr;
		hr = device->CreateTexture2D(&desc, init_data, &tex);
		if (FAILED(hr) || !tex) return E_TRY_AGAIN;

		if (view)
		{
			D3D11_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
			srv_desc.Format = format;
			if (is_cube_map)
			{
				if (array_size > 6)
				{
					srv_desc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURECUBEARRAY;
					srv_desc.TextureCubeArray.MipLevels = mip_count == 0U ? uint(-1) : desc.MipLevels;

					// Earlier we set array_size to (NumCubes * 6)
					srv_desc.TextureCubeArray.NumCubes = uint(array_size / 6);
				}
				else
				{
					srv_desc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURECUBE;
					srv_desc.TextureCube.MipLevels = mip_count == 0U ? uint(-1) : desc.MipLevels;
				}
			}
			else if (array_size > 1)
			{
				srv_desc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2DARRAY;
				srv_desc.Texture2DArray.MipLevels = mip_count == 0U ? uint(-1) : desc.MipLevels;
				srv_desc.Texture2DArray.ArraySize = uint(array_size);
			}
			else
			{
				srv_desc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
				srv_desc.Texture2D.MipLevels = mip_count == 0U ? uint(-1) : desc.MipLevels;
			}

			hr = device->CreateShaderResourceView(tex, &srv_desc, view);
			if (FAILED(hr))
			{
				tex->Release();
				return hr;
			}
		}

		if (texture)
		{
			*texture = tex;
		}
		else
		{
			tex->Release();
		}
	}
	break;
	case D3D11_RESOURCE_DIMENSION_TEXTURE3D:
	{
		D3D11_TEXTURE3D_DESC desc = {};
		desc.Width = uint(width);
		desc.Height = uint(height);
		desc.Depth = uint(depth);
		desc.MipLevels = uint(mip_count);
		desc.Format = format;
		desc.Usage = usage;
		desc.BindFlags = bind_flags;
		desc.CPUAccessFlags = cpu_access_flags;
		desc.MiscFlags = misc_flags & ~uint(D3D11_RESOURCE_MISC_TEXTURECUBE);

		ID3D11Texture3D* tex = nullptr;
		hr = device->CreateTexture3D(&desc, init_data, &tex);
		if (FAILED(hr) || !tex) return E_TRY_AGAIN;

		if (view)
		{
			D3D11_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
			srv_desc.Format = format;
			srv_desc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE3D;
			srv_desc.Texture3D.MipLevels = mip_count == 0U ? uint(-1) : desc.MipLevels;
			hr = device->CreateShaderResourceView(tex, &srv_desc, view);
			if (FAILED(hr))
			{
				tex->Release();
				return hr;
			}
		}

		if (texture)
		{
			*texture = tex;
		}
		else
		{
			tex->Release();
		}
	}
	break;
	}

	return hr;
}

static HRESULT create_parsed_texture(ID3D11Device* device, uint width, uint height, uint depth, uint mip_count, uint res_dim, uint array_size,
	DXGI_FORMAT format, bool is_cubemap, bool is_dx10,
	const utils::blob_view& data, ID3D11Resource** texture, ID3D11ShaderResourceView** view) noexcept
{
	auto autogen = false;
	/*if (needs_mips_regenerated(width, height, uint(mip_count)) && view && !is_dx10) // Must have context and shader-view to auto generate mipmaps
	{
		auto format_support = 0U;
		if (SUCCEEDED(AC::d3d::device()->CheckFormatSupport(format, &format_support))
			&& (format_support & D3D11_FORMAT_SUPPORT_MIP_AUTOGEN) != 0)
		{
			autogen = true;
		}
		else
		{
			return E_TRY_AGAIN;
		}
	}*/

	D3D11_USAGE usage = D3D11_USAGE_DEFAULT;
	uint bind_flags = D3D11_BIND_SHADER_RESOURCE;
	constexpr uint cpu_access_flags = 0;
	uint misc_flags = 0;

	auto final_mip_count = mip_count;
	if (autogen)
	{
		final_mip_count = std::max(uint(mip_count), uint(floorf(log2f(float(std::min(width, height))))));
		bind_flags |= D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
		misc_flags |= D3D11_RESOURCE_MISC_GENERATE_MIPS;
	}
	else
	{
		usage = D3D11_USAGE_IMMUTABLE;
	}

	// Create the texture
	const std::unique_ptr<D3D11_SUBRESOURCE_DATA[]> init_data(new(std::nothrow) D3D11_SUBRESOURCE_DATA[final_mip_count * array_size]);
	size_t skip_mip = 0;
	size_t twidth = 0;
	size_t theight = 0;
	size_t tdepth = 0;

	auto hr = fill_init_data(width, height, depth, final_mip_count, mip_count, array_size, format,
		0, data.size(), (const uint8_t*)data.data(),
		twidth, theight, tdepth, skip_mip, init_data.get());
	if (FAILED(hr))
	{
		return E_TRY_AGAIN;
	}

	hr = create_resources(device, res_dim, twidth, theight, tdepth, final_mip_count - skip_mip, array_size, format,
		usage, bind_flags, cpu_access_flags, misc_flags, is_cubemap, init_data.get(), texture, view);
	// generate_mips(autogen && SUCCEEDED(hr) ? *view : nullptr);
	return hr;
}

static HRESULT create_texture_from_dds(ID3D11Device* device, const DDS_HEADER* header, const utils::blob_view& data, ID3D11Resource** texture,
	ID3D11ShaderResourceView** view) noexcept
{
	const auto width = header->width;
	auto height = header->height;
	auto depth = header->depth;

	uint res_dim;
	uint array_size = 1;
	DXGI_FORMAT format;
	auto is_cubemap = false;

	size_t mip_count = header->mip_map_count;
	if (0 == mip_count)
	{
		mip_count = 1;
	}

	const auto is_dx10 = (header->ddspf.flags & DDS_FOURCC) != 0 && MAKEFOURCC_bc7('D', 'X', '1', '0') == header->ddspf.fourCC;
	if (is_dx10)
	{
		const auto dxt10 = reinterpret_cast<const DDS_HEADER_DXT10*>(reinterpret_cast<const char*>(header) + sizeof(DDS_HEADER));
		array_size = dxt10->array_size;
		if (array_size == 0)
		{
			return HRESULT_FROM_WIN32(ERROR_INVALID_DATA);
		}

		switch (dxt10->dxgi_format)
		{
		case DXGI_FORMAT_NV12:
		case DXGI_FORMAT_P010:
		case DXGI_FORMAT_P016:
		case DXGI_FORMAT_420_OPAQUE:
			if (dxt10->resource_dimension != D3D11_RESOURCE_DIMENSION_TEXTURE2D || width % 2 != 0 || height % 2 != 0)
			{
				return E_TRY_AGAIN;
			}
			break;

		case DXGI_FORMAT_YUY2:
		case DXGI_FORMAT_Y210:
		case DXGI_FORMAT_Y216:
		case DXGI_FORMAT_P208:
			if (width % 2 != 0)
			{
				return E_TRY_AGAIN;
			}
			break;

		case DXGI_FORMAT_NV11:
			if (width % 4 != 0)
			{
				return E_TRY_AGAIN;
			}
			break;

		case DXGI_FORMAT_AI44:
		case DXGI_FORMAT_IA44:
		case DXGI_FORMAT_P8:
		case DXGI_FORMAT_A8P8:
			return E_TRY_AGAIN;

		default:
			if (bits_per_pixel(dxt10->dxgi_format) == 0)
			{
				return E_TRY_AGAIN;
			}
		}

		format = dxt10->dxgi_format;

		switch (dxt10->resource_dimension)
		{
		case D3D11_RESOURCE_DIMENSION_TEXTURE1D:
			// D3DX writes 1D textures with a fixed Height of 1
			if ((header->flags & DDS_HEIGHT) && height != 1)
			{
				return HRESULT_FROM_WIN32(ERROR_INVALID_DATA);
			}
			height = depth = 1;
			break;

		case D3D11_RESOURCE_DIMENSION_TEXTURE2D:
			if (dxt10->misc_flag & D3D11_RESOURCE_MISC_TEXTURECUBE)
			{
				array_size *= 6;
				is_cubemap = true;
			}
			depth = 1;
			break;

		case D3D11_RESOURCE_DIMENSION_TEXTURE3D:
			if (!(header->flags & DDS_HEADER_FLAGS_VOLUME))
			{
				return HRESULT_FROM_WIN32(ERROR_INVALID_DATA);
			}

			if (array_size > 1)
			{
				return E_TRY_AGAIN;
			}
			break;

		case D3D11_RESOURCE_DIMENSION_BUFFER:
			return E_TRY_AGAIN;

		case D3D11_RESOURCE_DIMENSION_UNKNOWN:
		default:
			return E_TRY_AGAIN;
		}

		res_dim = dxt10->resource_dimension;
	}
	else
	{
		format = get_dxgi_format(header->ddspf);

		if (format == DXGI_FORMAT_UNKNOWN)
		{
			return E_TRY_AGAIN;
		}

		if (header->flags & DDS_HEADER_FLAGS_VOLUME)
		{
			res_dim = D3D11_RESOURCE_DIMENSION_TEXTURE3D;
		}
		else
		{
			if (header->caps2 & DDS_CUBEMAP)
			{
				// We require all six faces to be defined
				if ((header->caps2 & DDS_CUBEMAP_ALLFACES) != DDS_CUBEMAP_ALLFACES)
				{
					return E_TRY_AGAIN;
				}

				array_size = 6;
				is_cubemap = true;
			}

			depth = 1;
			res_dim = D3D11_RESOURCE_DIMENSION_TEXTURE2D;
			// Note there's no way for a legacy Direct3D 9 DDS to express a '1D' texture
		}
	}

	/*if (misc_flags & D3D11_RESOURCE_MISC_TEXTURECUBE && res_dim == D3D11_RESOURCE_DIMENSION_TEXTURE2D && array_size % 6 == 0)
	{
		is_cubemap = true;
	}*/

	// Bound sizes (for security purposes we don't trust DDS file metadata larger than the Direct3D hardware requirements)
	if (mip_count > D3D11_REQ_MIP_LEVELS)
	{
		return E_TRY_AGAIN;
	}

	switch (res_dim)
	{
	case D3D11_RESOURCE_DIMENSION_TEXTURE1D:
		if (array_size > D3D11_REQ_TEXTURE1D_ARRAY_AXIS_DIMENSION ||
			width > D3D11_REQ_TEXTURE1D_U_DIMENSION)
		{
			return E_TRY_AGAIN;
		}
		break;

	case D3D11_RESOURCE_DIMENSION_TEXTURE2D:
		if (is_cubemap)
		{
			// This is the right bound because we set array_size to (NumCubes*6) above
			if (array_size > D3D11_REQ_TEXTURE2D_ARRAY_AXIS_DIMENSION ||
				width > D3D11_REQ_TEXTURECUBE_DIMENSION ||
				height > D3D11_REQ_TEXTURECUBE_DIMENSION)
			{
				return E_TRY_AGAIN;
			}
		}
		else if (array_size > D3D11_REQ_TEXTURE2D_ARRAY_AXIS_DIMENSION ||
			width > D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION ||
			height > D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION)
		{
			return E_TRY_AGAIN;
		}
		break;

	case D3D11_RESOURCE_DIMENSION_TEXTURE3D:
		if (array_size > 1 ||
			width > D3D11_REQ_TEXTURE3D_U_V_OR_W_DIMENSION ||
			height > D3D11_REQ_TEXTURE3D_U_V_OR_W_DIMENSION ||
			depth > D3D11_REQ_TEXTURE3D_U_V_OR_W_DIMENSION)
		{
			return E_TRY_AGAIN;
		}
		break;

	case D3D11_RESOURCE_DIMENSION_BUFFER:
		return E_TRY_AGAIN;

	case D3D11_RESOURCE_DIMENSION_UNKNOWN:
	default:
		return E_TRY_AGAIN;
	}

	return create_parsed_texture(device, width, height, depth, uint(mip_count), res_dim, array_size, format, is_cubemap, is_dx10, data, texture, view);
}

HRESULT create_srv_from(ID3D11Device* device, const utils::blob_view& data, ID3D11ShaderResourceView** ret)
{
	const DDS_HEADER* header{};
	utils::blob_view bit_view;
	if (!use_direct_loader(data)
		|| FAILED(load_texture_data_from_memory((uint8_t*)data.data(), data.size(), &header, nullptr, bit_view)))
	{
		return E_TRY_AGAIN;
	}

	ID3D11Resource* tex{};
	const auto hr = create_texture_from_dds(device, header, bit_view, &tex, ret);
	if (tex) tex->Release();
	return hr;
}
