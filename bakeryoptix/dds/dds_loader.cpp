#include "dds_loader.h"
#include <D3D11.h>
#include <D3DX11tex.h>
#include <dx_shaders.h>
#include <utils/base64.h>
#include <utils/filesystem.h>
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "d3dx11.lib")

static struct
{
	ID3D11Device* device{};
	ID3D11DeviceContext* context{};

	ID3D11BlendState* state_blend{};
	ID3D11RasterizerState* state_cull{};
	ID3D11DepthStencilState* state_depth{};
	ID3D11VertexShader* sh_copy_vs{};
	ID3D11PixelShader* sh_copy_ps{};

	void ensure_initialized()
	{
		if (device && context) return;
		if (FAILED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0U, nullptr, 0U, D3D11_SDK_VERSION,
			&device, nullptr, &context)))
		{
			throw std::runtime_error("Failed to initialize D3D11");
		}

		{
			D3D11_BLEND_DESC desc{};
			desc.RenderTarget[0].BlendEnable = false;
			desc.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
			device->CreateBlendState(&desc, &state_blend);
		}

		{
			D3D11_RASTERIZER_DESC desc{};
			desc.FillMode = D3D11_FILL_SOLID;
			desc.CullMode = D3D11_CULL_NONE;
			desc.ScissorEnable = false;
			desc.DepthClipEnable = false;
			device->CreateRasterizerState(&desc, &state_cull);
		}

		{
			D3D11_DEPTH_STENCIL_DESC desc{};
			desc.DepthEnable = false;
			desc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
			desc.DepthFunc = D3D11_COMPARISON_ALWAYS;
			desc.StencilEnable = false;
			desc.FrontFace.StencilFailOp = desc.FrontFace.StencilDepthFailOp = desc.FrontFace.StencilPassOp = D3D11_STENCIL_OP_KEEP;
			desc.FrontFace.StencilFunc = D3D11_COMPARISON_ALWAYS;
			desc.BackFace = desc.FrontFace;
			device->CreateDepthStencilState(&desc, &state_depth);
		}

		const auto sh_copy_vs_data = utils::base64::decode(dx_shader_copy_vs());
		if (FAILED(device->CreateVertexShader(sh_copy_vs_data.data(), sh_copy_vs_data.size(), nullptr, &sh_copy_vs)))
		{
			throw std::exception("Failed to load vertex shader");
		}
		
		const auto sh_copy_ps_data = utils::base64::decode(dx_shader_copy_ps());
		if (FAILED(device->CreatePixelShader(sh_copy_ps_data.data(), sh_copy_ps_data.size(), nullptr, &sh_copy_ps)))
		{
			throw std::exception("Failed to load pixel shader");
		}
	}

	void set_state() const
	{
		const float blend_factor[4] = {0.f, 0.f, 0.f, 0.f};
		context->OMSetBlendState(state_blend, blend_factor, 0xffffffff);
		context->RSSetState(state_cull);
		context->OMSetDepthStencilState(state_depth, 0);
		
		context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
		context->IASetInputLayout(nullptr);
		context->IASetIndexBuffer(nullptr, DXGI_FORMAT(0), 0);
		context->IASetVertexBuffers(0, 0, nullptr, nullptr, nullptr);
		
		context->VSSetShader(sh_copy_vs, nullptr, 0U);
		context->PSSetShader(sh_copy_ps, nullptr, 0U);
	}

	ID3D11Texture2D* create_texture(uint32_t width, uint32_t height, bool cpu_readable) const
	{
		D3D11_TEXTURE2D_DESC desc;
		desc.Width = width;
		desc.Height = height;
		desc.MipLevels = 1U;
		desc.ArraySize = 1U;
		desc.SampleDesc.Count = 1U;
		desc.SampleDesc.Quality = 0U;
		desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
		desc.CPUAccessFlags = cpu_readable ? D3D11_CPU_ACCESS_READ : 0;
		desc.BindFlags = cpu_readable ? 0 : D3D11_BIND_RENDER_TARGET;
		desc.MiscFlags = 0;
		desc.Usage = cpu_readable ? D3D11_USAGE_STAGING : D3D11_USAGE_DEFAULT;

		ID3D11Texture2D* ret;
		if (FAILED(device->CreateTexture2D(&desc, nullptr, &ret)))
		{
			throw std::runtime_error("Failed to create texture");
		}
		return ret;
	}

	std::unique_ptr<std::string> load_data(const char* data, size_t size, uint32_t& width, uint32_t& height)
	{		
		ensure_initialized();
		ID3D11ShaderResourceView* view;
		if (FAILED(D3DX11CreateShaderResourceViewFromMemory(device, data, size, nullptr, nullptr, &view, nullptr)))
		{
			throw std::runtime_error("Failed to load texture");
		}

		{
			ID3D11Texture2D* tex;
			view->GetResource((ID3D11Resource**)&tex);
			D3D11_TEXTURE2D_DESC tex_desc;
			tex->GetDesc(&tex_desc);
			tex->Release();
			width = tex_desc.Width;
			height = tex_desc.Height;

			if (width > 16) width /= 4;
			if (height > 16) height /= 4;
		}

		const auto tex_rt = create_texture(width, height, false);
		ID3D11RenderTargetView* rt;
		if (FAILED(device->CreateRenderTargetView(tex_rt, nullptr, &rt)))
		{
			throw std::runtime_error("Failed to create RT");
		}
		
		set_state();
		D3D11_VIEWPORT viewport{0.f, 0.f, float(width), float(height), 0.f, 1.f};
		context->RSSetViewports(1, &viewport);
		context->OMSetRenderTargets(1, &rt, nullptr);
		context->PSSetShaderResources(0, 1, &view);
		context->Draw(3, 0);
		
		context->OMSetRenderTargets(0, nullptr, nullptr);
		ID3D11ShaderResourceView* view_null{};
		context->PSSetShaderResources(0, 1, &view_null);
		view->Release();
		rt->Release();
		
		const auto tex_cpu = create_texture(width, height, true);
		context->CopyResource(tex_cpu, tex_rt);
		tex_rt->Release();

		D3D11_MAPPED_SUBRESOURCE res;
		if (FAILED(context->Map(tex_cpu, 0, D3D11_MAP_READ, 0, &res)))
		{
			throw std::runtime_error("Failed to map texture to CPU memory");
		}

		auto ret = std::make_unique<std::string>();
		ret->resize(4 * width * height);
		for (auto y = 0U; y < height; ++y)
		{
			const auto row_d = &(*ret)[4 * width * y];
			const auto row_s = (int*)&((char*)res.pData)[res.RowPitch];
			memcpy(row_d, row_s, 4 * width);
		}
		context->Unmap(tex_cpu, 0);
		
		//static int i;
		//D3DX11SaveTextureToFileA(context, tex_cpu, D3DX11_IFF_DDS, ("H:/0/a_" + std::to_string(i++) + ".dds").c_str());

		tex_cpu->Release();
		return ret;
	}
} dds_loader_data;

dds_loader::dds_loader(const char* data, size_t size)
{
	this->data = dds_loader_data.load_data(data, size, width, height);
}
