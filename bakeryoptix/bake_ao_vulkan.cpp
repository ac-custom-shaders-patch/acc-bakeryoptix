#include <bake_ao_vulkan.h>
#include <bake_ao_embree.h>
#include <bake_api.h>
#include <optix_compat.h>
#include <dds/dds_loader.h>
#include <utils/cout_progress.h>

// Windows must come before Vulkan
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

// Vulkan + VMA
#include <vulkan/vulkan.h>

#define VMA_IMPLEMENTATION
#define VMA_STATIC_VULKAN_FUNCTIONS 1
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0
#include <vulkan/vk_mem_alloc.h>

#include <iostream>
#include <vector>
#include <cstring>
#include <algorithm>
#include <cassert>

// Embedded SPIR-V
#include "shaders/ao_trace_spv.h"

// Link vulkan-1.lib (delay-loaded so the app still starts without Vulkan runtime)
#pragma comment(lib, "vulkan-1.lib")
#pragma comment(lib, "delayimp.lib")

using namespace bake;

// ════════════════════════════════════════════════════════════════
// Helper types
// ════════════════════════════════════════════════════════════════

namespace {

struct VkCtx
{
	VkInstance       instance      = VK_NULL_HANDLE;
	VkPhysicalDevice gpu           = VK_NULL_HANDLE;
	VkDevice         device        = VK_NULL_HANDLE;
	VkQueue          queue         = VK_NULL_HANDLE;
	uint32_t         queueFamily   = 0;
	VkCommandPool    cmdPool       = VK_NULL_HANDLE;
	VmaAllocator     vma           = VK_NULL_HANDLE;

	// Extension function pointers (loaded after device creation)
	PFN_vkCreateAccelerationStructureKHR           pfnCreateAS          = nullptr;
	PFN_vkDestroyAccelerationStructureKHR          pfnDestroyAS         = nullptr;
	PFN_vkGetAccelerationStructureBuildSizesKHR    pfnGetASBuildSizes   = nullptr;
	PFN_vkCmdBuildAccelerationStructuresKHR        pfnCmdBuildAS        = nullptr;
	PFN_vkGetAccelerationStructureDeviceAddressKHR pfnGetASDeviceAddr   = nullptr;

	void loadExtFunctions()
	{
		pfnCreateAS        = (PFN_vkCreateAccelerationStructureKHR)          vkGetDeviceProcAddr(device, "vkCreateAccelerationStructureKHR");
		pfnDestroyAS       = (PFN_vkDestroyAccelerationStructureKHR)         vkGetDeviceProcAddr(device, "vkDestroyAccelerationStructureKHR");
		pfnGetASBuildSizes = (PFN_vkGetAccelerationStructureBuildSizesKHR)   vkGetDeviceProcAddr(device, "vkGetAccelerationStructureBuildSizesKHR");
		pfnCmdBuildAS      = (PFN_vkCmdBuildAccelerationStructuresKHR)       vkGetDeviceProcAddr(device, "vkCmdBuildAccelerationStructuresKHR");
		pfnGetASDeviceAddr = (PFN_vkGetAccelerationStructureDeviceAddressKHR)vkGetDeviceProcAddr(device, "vkGetAccelerationStructureDeviceAddressKHR");
	}
};

struct GpuBuf
{
	VkBuffer       buf   = VK_NULL_HANDLE;
	VmaAllocation  alloc = VK_NULL_HANDLE;
	VkDeviceSize   size  = 0;
};

struct AccelStruct
{
	VkAccelerationStructureKHR as  = VK_NULL_HANDLE;
	GpuBuf                     buf;
};

// GPU-side structs matching shader layout (std430)
struct GeomInfoGpu
{
	uint32_t matType;
	float    alphaRef;
	float    passChance;
	float    albedo;
	float    emissive;
	uint32_t vertexOffset;
	uint32_t indexOffset;
	uint32_t textureIndex;
};

struct TextureInfoGpu
{
	uint32_t width;
	uint32_t height;
	uint32_t dataOffset;
	uint32_t _pad;
};

struct PushConstants
{
	int32_t  px;
	int32_t  py;
	int32_t  sqrtPasses;
	uint32_t numSamples;
	float    sceneOffsetH;
	float    sceneOffsetV;
	uint32_t bounceCounts;
	uint32_t useOcclusionShortcut;
};

// ════════════════════════════════════════════════════════════════
// Buffer helpers
// ════════════════════════════════════════════════════════════════

GpuBuf createBuffer(VkCtx& ctx, VkDeviceSize size, VkBufferUsageFlags usage, VmaMemoryUsage memUsage)
{
	GpuBuf b;
	b.size = size;

	VkBufferCreateInfo bufCI{};
	bufCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufCI.size  = size;
	bufCI.usage = usage;

	VmaAllocationCreateInfo allocCI{};
	allocCI.usage = memUsage;

	vmaCreateBuffer(ctx.vma, &bufCI, &allocCI, &b.buf, &b.alloc, nullptr);
	return b;
}

void destroyBuffer(VkCtx& ctx, GpuBuf& b)
{
	if (b.buf) vmaDestroyBuffer(ctx.vma, b.buf, b.alloc);
	b.buf = VK_NULL_HANDLE;
	b.alloc = VK_NULL_HANDLE;
}

VkDeviceAddress getBufferAddress(VkCtx& ctx, VkBuffer buf)
{
	VkBufferDeviceAddressInfo info{};
	info.sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
	info.buffer = buf;
	return vkGetBufferDeviceAddress(ctx.device, &info);
}

// ════════════════════════════════════════════════════════════════
// One-shot command buffer helpers
// ════════════════════════════════════════════════════════════════

VkCommandBuffer beginCmd(VkCtx& ctx)
{
	VkCommandBufferAllocateInfo ai{};
	ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	ai.commandPool        = ctx.cmdPool;
	ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	ai.commandBufferCount = 1;
	VkCommandBuffer cmd;
	vkAllocateCommandBuffers(ctx.device, &ai, &cmd);

	VkCommandBufferBeginInfo bi{};
	bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	vkBeginCommandBuffer(cmd, &bi);
	return cmd;
}

void endCmd(VkCtx& ctx, VkCommandBuffer cmd)
{
	vkEndCommandBuffer(cmd);
	VkSubmitInfo si{};
	si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	si.commandBufferCount = 1;
	si.pCommandBuffers    = &cmd;
	vkQueueSubmit(ctx.queue, 1, &si, VK_NULL_HANDLE);
	vkQueueWaitIdle(ctx.queue);
	vkFreeCommandBuffers(ctx.device, ctx.cmdPool, 1, &cmd);
}

// Upload data to a device-local buffer via staging. Returns empty buffer if size==0.
GpuBuf uploadToGpu(VkCtx& ctx, const void* data, VkDeviceSize size, VkBufferUsageFlags extraUsage)
{
	if (size == 0)
		return createBuffer(ctx, 4, extraUsage, VMA_MEMORY_USAGE_GPU_ONLY);

	GpuBuf staging = createBuffer(ctx, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);
	void* mapped;
	vmaMapMemory(ctx.vma, staging.alloc, &mapped);
	memcpy(mapped, data, (size_t)size);
	vmaUnmapMemory(ctx.vma, staging.alloc);

	GpuBuf device = createBuffer(ctx, size,
		extraUsage | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

	VkCommandBuffer cmd = beginCmd(ctx);
	VkBufferCopy region{0, 0, size};
	vkCmdCopyBuffer(cmd, staging.buf, device.buf, 1, &region);
	endCmd(ctx, cmd);

	destroyBuffer(ctx, staging);
	return device;
}

// ════════════════════════════════════════════════════════════════
// Vulkan initialisation
// ════════════════════════════════════════════════════════════════

bool initVulkan(VkCtx& ctx)
{
	// --- Instance ---
	VkApplicationInfo appInfo{};
	appInfo.sType      = VK_STRUCTURE_TYPE_APPLICATION_INFO;
	appInfo.pApplicationName = "bakeryoptix";
	appInfo.apiVersion = VK_API_VERSION_1_2;

	VkInstanceCreateInfo instCI{};
	instCI.sType            = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	instCI.pApplicationInfo = &appInfo;
	if (vkCreateInstance(&instCI, nullptr, &ctx.instance) != VK_SUCCESS) return false;

	// --- Physical device ---
	uint32_t devCount = 0;
	vkEnumeratePhysicalDevices(ctx.instance, &devCount, nullptr);
	if (devCount == 0) return false;
	std::vector<VkPhysicalDevice> devs(devCount);
	vkEnumeratePhysicalDevices(ctx.instance, &devCount, devs.data());

	// Pick first device with required extensions
	const char* requiredExts[] = {
		VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
		VK_KHR_RAY_QUERY_EXTENSION_NAME,
		VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME
	};
	constexpr uint32_t numRequired = 3;

	for (auto dev : devs)
	{
		uint32_t extCount = 0;
		vkEnumerateDeviceExtensionProperties(dev, nullptr, &extCount, nullptr);
		std::vector<VkExtensionProperties> exts(extCount);
		vkEnumerateDeviceExtensionProperties(dev, nullptr, &extCount, exts.data());

		uint32_t found = 0;
		for (auto& e : exts)
			for (uint32_t r = 0; r < numRequired; ++r)
				if (strcmp(e.extensionName, requiredExts[r]) == 0) ++found;

		if (found >= numRequired)
		{
			ctx.gpu = dev;
			break;
		}
	}
	if (ctx.gpu == VK_NULL_HANDLE) return false;

	// --- Check feature support ---
	VkPhysicalDeviceRayQueryFeaturesKHR            rqFeats{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
	VkPhysicalDeviceAccelerationStructureFeaturesKHR asFeats{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
	asFeats.pNext = &rqFeats;
	VkPhysicalDeviceVulkan12Features               v12Feats{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
	v12Feats.pNext = &asFeats;
	VkPhysicalDeviceFeatures2                      feats2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
	feats2.pNext = &v12Feats;
	vkGetPhysicalDeviceFeatures2(ctx.gpu, &feats2);

	if (!rqFeats.rayQuery || !asFeats.accelerationStructure || !v12Feats.bufferDeviceAddress)
		return false;

	// --- Find compute queue ---
	uint32_t qfCount = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(ctx.gpu, &qfCount, nullptr);
	std::vector<VkQueueFamilyProperties> qfProps(qfCount);
	vkGetPhysicalDeviceQueueFamilyProperties(ctx.gpu, &qfCount, qfProps.data());
	ctx.queueFamily = UINT32_MAX;
	for (uint32_t i = 0; i < qfCount; ++i)
	{
		if (qfProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT)
		{
			ctx.queueFamily = i;
			break;
		}
	}
	if (ctx.queueFamily == UINT32_MAX) return false;

	// --- Logical device ---
	float queuePriority = 1.0f;
	VkDeviceQueueCreateInfo queueCI{};
	queueCI.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
	queueCI.queueFamilyIndex = ctx.queueFamily;
	queueCI.queueCount       = 1;
	queueCI.pQueuePriorities = &queuePriority;

	// Enable features via pNext chain
	VkPhysicalDeviceRayQueryFeaturesKHR             enableRQ{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
	enableRQ.rayQuery = VK_TRUE;
	VkPhysicalDeviceAccelerationStructureFeaturesKHR enableAS{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
	enableAS.accelerationStructure = VK_TRUE;
	enableAS.pNext = &enableRQ;
	VkPhysicalDeviceVulkan12Features                 enableV12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
	enableV12.bufferDeviceAddress = VK_TRUE;
	enableV12.pNext = &enableAS;

	VkDeviceCreateInfo devCI{};
	devCI.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
	devCI.pNext                   = &enableV12;
	devCI.queueCreateInfoCount    = 1;
	devCI.pQueueCreateInfos       = &queueCI;
	devCI.enabledExtensionCount   = numRequired;
	devCI.ppEnabledExtensionNames = requiredExts;

	if (vkCreateDevice(ctx.gpu, &devCI, nullptr, &ctx.device) != VK_SUCCESS) return false;

	vkGetDeviceQueue(ctx.device, ctx.queueFamily, 0, &ctx.queue);
	ctx.loadExtFunctions();

	// --- Command pool ---
	VkCommandPoolCreateInfo poolCI{};
	poolCI.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	poolCI.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	poolCI.queueFamilyIndex = ctx.queueFamily;
	vkCreateCommandPool(ctx.device, &poolCI, nullptr, &ctx.cmdPool);

	// --- VMA ---
	VmaAllocatorCreateInfo vmaCI{};
	vmaCI.vulkanApiVersion = VK_API_VERSION_1_2;
	vmaCI.physicalDevice   = ctx.gpu;
	vmaCI.device           = ctx.device;
	vmaCI.instance         = ctx.instance;
	vmaCI.flags            = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
	vmaCreateAllocator(&vmaCI, &ctx.vma);

	return true;
}

void cleanupVulkan(VkCtx& ctx)
{
	if (ctx.vma)     { vmaDestroyAllocator(ctx.vma); ctx.vma = VK_NULL_HANDLE; }
	if (ctx.cmdPool) { vkDestroyCommandPool(ctx.device, ctx.cmdPool, nullptr); ctx.cmdPool = VK_NULL_HANDLE; }
	if (ctx.device)  { vkDestroyDevice(ctx.device, nullptr); ctx.device = VK_NULL_HANDLE; }
	if (ctx.instance){ vkDestroyInstance(ctx.instance, nullptr); ctx.instance = VK_NULL_HANDLE; }
}

// ════════════════════════════════════════════════════════════════
// Acceleration structure helpers
// ════════════════════════════════════════════════════════════════

AccelStruct buildBLAS(VkCtx& ctx, VkDeviceAddress vertAddr, uint32_t vertCount,
	VkDeviceSize vertStride, VkDeviceAddress idxAddr, uint32_t triCount, bool isOpaque)
{
	VkAccelerationStructureGeometryKHR geom{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
	geom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
	geom.flags        = isOpaque ? (VkGeometryFlagsKHR)VK_GEOMETRY_OPAQUE_BIT_KHR : 0;

	auto& tris                 = geom.geometry.triangles;
	tris.sType                 = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
	tris.vertexFormat          = VK_FORMAT_R32G32B32_SFLOAT;
	tris.vertexData.deviceAddress = vertAddr;
	tris.vertexStride          = vertStride;
	tris.maxVertex             = vertCount - 1;
	tris.indexType             = VK_INDEX_TYPE_UINT32;
	tris.indexData.deviceAddress  = idxAddr;

	VkAccelerationStructureBuildGeometryInfoKHR buildInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
	buildInfo.type          = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
	buildInfo.flags         = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
	buildInfo.mode          = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
	buildInfo.geometryCount = 1;
	buildInfo.pGeometries   = &geom;

	VkAccelerationStructureBuildSizesInfoKHR sizes{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
	ctx.pfnGetASBuildSizes(ctx.device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
		&buildInfo, &triCount, &sizes);

	AccelStruct as;
	as.buf = createBuffer(ctx, sizes.accelerationStructureSize,
		VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
		VMA_MEMORY_USAGE_GPU_ONLY);

	VkAccelerationStructureCreateInfoKHR asCI{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
	asCI.buffer = as.buf.buf;
	asCI.size   = sizes.accelerationStructureSize;
	asCI.type   = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
	ctx.pfnCreateAS(ctx.device, &asCI, nullptr, &as.as);

	GpuBuf scratch = createBuffer(ctx, sizes.buildScratchSize,
		VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
		VMA_MEMORY_USAGE_GPU_ONLY);

	buildInfo.dstAccelerationStructure  = as.as;
	buildInfo.scratchData.deviceAddress = getBufferAddress(ctx, scratch.buf);

	VkAccelerationStructureBuildRangeInfoKHR range{};
	range.primitiveCount = triCount;
	const auto* pRange = &range;

	VkCommandBuffer cmd = beginCmd(ctx);
	ctx.pfnCmdBuildAS(cmd, 1, &buildInfo, &pRange);
	endCmd(ctx, cmd);

	destroyBuffer(ctx, scratch);
	return as;
}

AccelStruct buildTLAS(VkCtx& ctx, const std::vector<VkAccelerationStructureInstanceKHR>& instances)
{
	GpuBuf instanceBuf = uploadToGpu(ctx, instances.data(),
		instances.size() * sizeof(VkAccelerationStructureInstanceKHR),
		VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
		| VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

	VkAccelerationStructureGeometryKHR geom{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
	geom.geometryType                          = VK_GEOMETRY_TYPE_INSTANCES_KHR;
	geom.geometry.instances.sType              = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
	geom.geometry.instances.arrayOfPointers    = VK_FALSE;
	geom.geometry.instances.data.deviceAddress = getBufferAddress(ctx, instanceBuf.buf);

	uint32_t instanceCount = (uint32_t)instances.size();

	VkAccelerationStructureBuildGeometryInfoKHR buildInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
	buildInfo.type          = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
	buildInfo.flags         = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
	buildInfo.mode          = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
	buildInfo.geometryCount = 1;
	buildInfo.pGeometries   = &geom;

	VkAccelerationStructureBuildSizesInfoKHR sizes{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
	ctx.pfnGetASBuildSizes(ctx.device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
		&buildInfo, &instanceCount, &sizes);

	AccelStruct tlas;
	tlas.buf = createBuffer(ctx, sizes.accelerationStructureSize,
		VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
		VMA_MEMORY_USAGE_GPU_ONLY);

	VkAccelerationStructureCreateInfoKHR asCI{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
	asCI.buffer = tlas.buf.buf;
	asCI.size   = sizes.accelerationStructureSize;
	asCI.type   = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
	ctx.pfnCreateAS(ctx.device, &asCI, nullptr, &tlas.as);

	GpuBuf scratch = createBuffer(ctx, sizes.buildScratchSize,
		VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
		VMA_MEMORY_USAGE_GPU_ONLY);

	buildInfo.dstAccelerationStructure  = tlas.as;
	buildInfo.scratchData.deviceAddress = getBufferAddress(ctx, scratch.buf);

	VkAccelerationStructureBuildRangeInfoKHR range{};
	range.primitiveCount = instanceCount;
	const auto* pRange = &range;

	VkCommandBuffer cmd = beginCmd(ctx);
	ctx.pfnCmdBuildAS(cmd, 1, &buildInfo, &pRange);
	endCmd(ctx, cmd);

	destroyBuffer(ctx, scratch);
	destroyBuffer(ctx, instanceBuf);
	return tlas;
}

// ════════════════════════════════════════════════════════════════
// Transform helpers (same as Embree version)
// ════════════════════════════════════════════════════════════════

static float3 xform_point(const float* m, const Vec3& p)
{
	return {
		p.x * m[0] + p.y * m[1] + p.z * m[2]  + m[3],
		p.x * m[4] + p.y * m[5] + p.z * m[6]  + m[7],
		p.x * m[8] + p.y * m[9] + p.z * m[10] + m[11]
	};
}

} // anonymous namespace

// ════════════════════════════════════════════════════════════════
// vulkan_available()
// ════════════════════════════════════════════════════════════════

static VkCtx _vulcan_ready_ctx;

const char* bake::vulkan_configure()
{
	// Check if the Vulkan loader DLL exists (prevents delay-load crash)
	HMODULE hVk = LoadLibraryA("vulkan-1.dll");
	if (!hVk)
	{
		return "couldn’t find “vulkan-1.dll”";
	}
	FreeLibrary(hVk);

	// Create temp instance
	VkApplicationInfo appInfo{};
	appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
	appInfo.apiVersion = VK_API_VERSION_1_2;

	VkInstanceCreateInfo ci{};
	ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	ci.pApplicationInfo = &appInfo;

	VkInstance inst;
	if (vkCreateInstance(&ci, nullptr, &inst) != VK_SUCCESS)
	{
		return "couldn’t create Vulkan instance";
	}

	uint32_t devCount = 0;
	vkEnumeratePhysicalDevices(inst, &devCount, nullptr);
	std::vector<VkPhysicalDevice> devs(devCount);
	if (devCount) vkEnumeratePhysicalDevices(inst, &devCount, devs.data());

	const char* need[] = {
		VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
		VK_KHR_RAY_QUERY_EXTENSION_NAME,
		VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME
	};

	bool found = false;
	for (auto dev : devs)
	{
		uint32_t extCount = 0;
		vkEnumerateDeviceExtensionProperties(dev, nullptr, &extCount, nullptr);
		std::vector<VkExtensionProperties> exts(extCount);
		vkEnumerateDeviceExtensionProperties(dev, nullptr, &extCount, exts.data());

		int hits = 0;
		for (auto& e : exts)
			for (int r = 0; r < 3; ++r)
				if (strcmp(e.extensionName, need[r]) == 0) ++hits;

		if (hits >= 3)
		{
			found = true;
			break;
		}
	}

	vkDestroyInstance(inst, nullptr);
	if (!found)
	{
		return "couldn’t find a Vulkan device with raycasting capabilities";
	}
	else if (!initVulkan(_vulcan_ready_ctx))
	{
		return "failed to set up Vulkan context";
	}
	else
	{
		return nullptr;
	}
}

bool bake::vulkan_available()
{
	return _vulcan_ready_ctx.instance != nullptr;
}

// ════════════════════════════════════════════════════════════════
// ao_vulkan() — main entry point
// ════════════════════════════════════════════════════════════════

void bake::ao_vulkan(const std::vector<Mesh*>& blockers,
	const AOSamples& ao_samples, const int rays_per_sample, const float albedo, const uint32_t bounce_counts,
	const float scene_offset_scale_horizontal, const float scene_offset_scale_vertical,
	const float trees_light_pass_chance, const bool debug_mode, float* ao_values)
{
	// We concatenate all meshes' data into single flat arrays.
	// Each mesh becomes one BLAS, one TLAS instance.
	struct MeshEntry
	{
		uint32_t vertexOffset;
		uint32_t indexOffset;
		uint32_t vertexCount;
		uint32_t triangleCount;
		bool     isOpaque;     // for TLAS instance flag
		GeomInfoGpu info;
	};

	std::vector<MeshEntry> entries;
	entries.reserve(blockers.size());

	// Accumulate totals for pre-allocation
	uint32_t totalVerts = 0, totalIndices = 0;
	bool hasEmissive = false;

	for (const auto* mesh : blockers)
	{
		if (mesh->triangles.empty() || mesh->vertices.empty()) continue;
		totalVerts  += (uint32_t)mesh->vertices.size();
		totalIndices += (uint32_t)mesh->triangles.size() * 3;
	}

	std::vector<float>    allPositions; allPositions.reserve(totalVerts * 4);
	std::vector<float>    allUVs;       allUVs.reserve(totalVerts * 2);
	std::vector<float>    allNormals;   allNormals.reserve(totalVerts * 4);
	std::vector<uint32_t> allIndices;   allIndices.reserve(totalIndices);
	std::vector<GeomInfoGpu> allGeomInfos;
	std::vector<float>    allMatrices;  // 16 floats (column-major mat4) per mesh

	// Textures
	std::vector<TextureInfoGpu> allTexInfos;
	std::vector<uint32_t>       allTexData;  // packed RGBA8 pixels
	// Map: texture raw pointer → texture index (dedup)
	struct TexKey { const dds_loader* ptr; uint32_t index; };
	std::vector<TexKey> texMap;

	auto getOrAddTexture = [&](const dds_loader* tex) -> uint32_t {
		if (!tex || !tex->data) return 0xFFFFFFFFu;
		for (auto& tk : texMap)
			if (tk.ptr == tex) return tk.index;
		uint32_t idx = (uint32_t)allTexInfos.size();
		TextureInfoGpu ti{};
		ti.width      = tex->width;
		ti.height     = tex->height;
		ti.dataOffset = (uint32_t)allTexData.size();
		allTexInfos.push_back(ti);
		// Copy pixel data as uint32 per pixel
		const auto* src = reinterpret_cast<const uint32_t*>(tex->data->data());
		allTexData.insert(allTexData.end(), src, src + (size_t)tex->width * tex->height);
		texMap.push_back({tex, idx});
		return idx;
	};

	uint32_t vertOff = 0, idxOff = 0;

	for (const auto* mesh : blockers)
	{
		if (mesh->triangles.empty() || mesh->vertices.empty()) continue;

		MeshEntry me{};
		me.vertexOffset  = vertOff;
		me.indexOffset   = idxOff;
		me.vertexCount   = (uint32_t)mesh->vertices.size();
		me.triangleCount = (uint32_t)mesh->triangles.size();

		// ── Material setup (mirrors Embree version) ──
		GeomInfoGpu& gi = me.info;
		gi.vertexOffset = vertOff;
		gi.indexOffset  = idxOff;
		gi.albedo       = albedo;
		gi.emissive     = 0.f;
		gi.alphaRef     = 0.5f * 255.f;
		gi.passChance   = trees_light_pass_chance;
		gi.textureIndex = 0xFFFFFFFFu;

		if (mesh->material && mesh->material->texture && mesh->material->texture->data)
		{
			const bool is_foliage = mesh->material->shader.find("ksTree") == 0;
			gi.matType = is_foliage ? 2u : 1u; // foliage : alpha_test
			const auto* var = mesh->material->get_var_or_null("ksAlphaRef");
			gi.alphaRef = (var ? var->v1 : 0.5f) * 255.f;
			gi.textureIndex = getOrAddTexture(mesh->material->texture.get());
		}
		else if (mesh->material)
		{
			gi.matType = 0u; // opaque
		}
		else
		{
			if (mesh->name == "blocker")      { gi.matType = 0u; gi.albedo = 0.f; }
			else if (mesh->name == "trees")   { gi.matType = 3u; gi.albedo = 0.2f; } // proc_foliage
			else if (mesh->name == "emissive"){ gi.matType = 4u; gi.albedo = 1.f; gi.emissive = mesh->lod_out; }
			else                              { gi.matType = 0u; gi.albedo = albedo; }
		}

		me.isOpaque = (gi.matType == 0u || gi.matType == 4u);
		if (gi.matType == 4u) hasEmissive = true;

		// ── Vertices (world-space) ──
		const float* mat = mesh->matrix.data();
		for (size_t i = 0; i < mesh->vertices.size(); ++i)
		{
			float3 wp = xform_point(mat, mesh->vertices[i].pos);
			allPositions.push_back(wp.x);
			allPositions.push_back(wp.y);
			allPositions.push_back(wp.z);
			allPositions.push_back(0.f); // padding for vec4

			allUVs.push_back(mesh->vertices[i].tex.x);
			allUVs.push_back(mesh->vertices[i].tex.y);

			if (!mesh->normals.empty())
			{
				allNormals.push_back(mesh->normals[i].x);
				allNormals.push_back(mesh->normals[i].y);
				allNormals.push_back(mesh->normals[i].z);
			}
			else
			{
				allNormals.push_back(0.f);
				allNormals.push_back(1.f);
				allNormals.push_back(0.f);
			}
			allNormals.push_back(0.f); // padding for vec4
		}

		// ── Indices (local to this mesh) ──
		for (size_t i = 0; i < mesh->triangles.size(); ++i)
		{
			allIndices.push_back(mesh->triangles[i].a);
			allIndices.push_back(mesh->triangles[i].b);
			allIndices.push_back(mesh->triangles[i].c);
		}

		// ── Matrix (column-major for GLSL mat4) ──
		// The mesh matrix is row-major in C++. GLSL mat4 is column-major.
		// Row-major m[row*4+col] → column-major: m[col*4+row]
		for (int col = 0; col < 4; ++col)
			for (int row = 0; row < 4; ++row)
				allMatrices.push_back(mat[row * 4 + col]);

		allGeomInfos.push_back(gi);
		entries.push_back(me);

		vertOff += me.vertexCount;
		idxOff  += me.triangleCount * 3;
	}

	if (entries.empty())
	{
		return;
	}

	// ── 3. Upload geometry to GPU ──
	const VkBufferUsageFlags asInput =
		VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
		| VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
		| VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

	auto& ctx = _vulcan_ready_ctx;
	GpuBuf posBuf  = uploadToGpu(ctx, allPositions.data(), allPositions.size() * sizeof(float), asInput);
	GpuBuf idxBuf  = uploadToGpu(ctx, allIndices.data(),   allIndices.size()   * sizeof(uint32_t), asInput);
	GpuBuf uvBuf   = uploadToGpu(ctx, allUVs.data(),       allUVs.size()       * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
	GpuBuf normBuf = uploadToGpu(ctx, allNormals.data(),   allNormals.size()   * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
	GpuBuf geomInfoBuf = uploadToGpu(ctx, allGeomInfos.data(), allGeomInfos.size() * sizeof(GeomInfoGpu), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
	GpuBuf matBuf  = uploadToGpu(ctx, allMatrices.data(),  allMatrices.size()  * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

	// Texture data (may be empty)
	if (allTexInfos.empty()) { allTexInfos.push_back({1, 1, 0, 0}); allTexData.push_back(0xFFFFFFFFu); }
	GpuBuf texInfoBuf = uploadToGpu(ctx, allTexInfos.data(), allTexInfos.size() * sizeof(TextureInfoGpu), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
	GpuBuf texDataBuf = uploadToGpu(ctx, allTexData.data(),  allTexData.size()  * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

	// ── 4. Build BLASes ──
	VkDeviceAddress posAddr = getBufferAddress(ctx, posBuf.buf);
	VkDeviceAddress idxAddr = getBufferAddress(ctx, idxBuf.buf);

	std::vector<AccelStruct> blases(entries.size());
	for (size_t i = 0; i < entries.size(); ++i)
	{
		const auto& e = entries[i];
		blases[i] = buildBLAS(ctx,
			posAddr + (VkDeviceSize)e.vertexOffset * 16,   // 4 floats per vertex
			e.vertexCount, 16,                              // stride = sizeof(vec4)
			idxAddr + (VkDeviceSize)e.indexOffset * 4,      // uint32 per index
			e.triangleCount, e.isOpaque);
	}

	// ── 5. Build TLAS ──
	std::vector<VkAccelerationStructureInstanceKHR> tlasInstances(entries.size());
	for (size_t i = 0; i < entries.size(); ++i)
	{
		auto& inst = tlasInstances[i];
		memset(&inst, 0, sizeof(inst));
		// Identity transform (vertices are already in world space)
		inst.transform.matrix[0][0] = 1.f;
		inst.transform.matrix[1][1] = 1.f;
		inst.transform.matrix[2][2] = 1.f;
		inst.instanceCustomIndex                    = (uint32_t)i;
		inst.mask                                   = 0xFF;
		inst.instanceShaderBindingTableRecordOffset = 0;
		inst.flags = entries[i].isOpaque
			? VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR
			: VK_GEOMETRY_INSTANCE_FORCE_NO_OPAQUE_BIT_KHR;

		VkAccelerationStructureDeviceAddressInfoKHR addrInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
		addrInfo.accelerationStructure = blases[i].as;
		inst.accelerationStructureReference = ctx.pfnGetASDeviceAddr(ctx.device, &addrInfo);
	}

	AccelStruct tlas = buildTLAS(ctx, tlasInstances);

	// ── 6. Upload sample data ──
	const size_t numSamples = ao_samples.num_samples;

	// Pad float3 → vec4
	std::vector<float> padSamplePos(numSamples * 4);
	std::vector<float> padSampleNorm(numSamples * 4);
	std::vector<float> padSampleFaceNorm(numSamples * 4);
	for (size_t i = 0; i < numSamples; ++i)
	{
		padSamplePos[i*4+0] = ao_samples.sample_positions[i*3+0];
		padSamplePos[i*4+1] = ao_samples.sample_positions[i*3+1];
		padSamplePos[i*4+2] = ao_samples.sample_positions[i*3+2];
		padSamplePos[i*4+3] = 0.f;

		padSampleNorm[i*4+0] = ao_samples.sample_normals[i*3+0];
		padSampleNorm[i*4+1] = ao_samples.sample_normals[i*3+1];
		padSampleNorm[i*4+2] = ao_samples.sample_normals[i*3+2];
		padSampleNorm[i*4+3] = 0.f;

		padSampleFaceNorm[i*4+0] = ao_samples.sample_face_normals[i*3+0];
		padSampleFaceNorm[i*4+1] = ao_samples.sample_face_normals[i*3+1];
		padSampleFaceNorm[i*4+2] = ao_samples.sample_face_normals[i*3+2];
		padSampleFaceNorm[i*4+3] = 0.f;
	}

	GpuBuf samplePosBuf      = uploadToGpu(ctx, padSamplePos.data(),      padSamplePos.size()      * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
	GpuBuf sampleNormBuf     = uploadToGpu(ctx, padSampleNorm.data(),     padSampleNorm.size()     * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
	GpuBuf sampleFaceNormBuf = uploadToGpu(ctx, padSampleFaceNorm.data(), padSampleFaceNorm.size() * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

	// AO output buffer (device-local, initialised to zero)
	GpuBuf aoBuf = createBuffer(ctx, numSamples * sizeof(float),
		VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		VMA_MEMORY_USAGE_GPU_ONLY);
	{
		VkCommandBuffer cmd = beginCmd(ctx);
		vkCmdFillBuffer(cmd, aoBuf.buf, 0, VK_WHOLE_SIZE, 0);
		endCmd(ctx, cmd);
	}

	// ── 7. Create compute pipeline ──
	VkShaderModuleCreateInfo smCI{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
	smCI.codeSize = ao_trace_spv_size;
	smCI.pCode    = ao_trace_spv;
	VkShaderModule shaderModule;
	vkCreateShaderModule(ctx.device, &smCI, nullptr, &shaderModule);

	// Descriptor set layout (13 bindings)
	VkDescriptorSetLayoutBinding bindings[13]{};
	for (int i = 0; i < 13; ++i)
	{
		bindings[i].binding         = i;
		bindings[i].descriptorCount = 1;
		bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
	}
	bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
	for (int i = 1; i <= 12; ++i)
		bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;

	VkDescriptorSetLayoutCreateInfo dslCI{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
	dslCI.bindingCount = 13;
	dslCI.pBindings    = bindings;
	VkDescriptorSetLayout dsLayout;
	vkCreateDescriptorSetLayout(ctx.device, &dslCI, nullptr, &dsLayout);

	VkPushConstantRange pcRange{};
	pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	pcRange.offset     = 0;
	pcRange.size       = sizeof(PushConstants);

	VkPipelineLayoutCreateInfo plCI{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
	plCI.setLayoutCount         = 1;
	plCI.pSetLayouts            = &dsLayout;
	plCI.pushConstantRangeCount = 1;
	plCI.pPushConstantRanges    = &pcRange;
	VkPipelineLayout pipeLayout;
	vkCreatePipelineLayout(ctx.device, &plCI, nullptr, &pipeLayout);

	VkComputePipelineCreateInfo cpCI{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
	cpCI.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	cpCI.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
	cpCI.stage.module = shaderModule;
	cpCI.stage.pName  = "main";
	cpCI.layout       = pipeLayout;
	VkPipeline pipeline;
	vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &cpCI, nullptr, &pipeline);

	// ── 8. Descriptor set ──
	VkDescriptorPoolSize poolSizes[] = {
		{ VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1 },
		{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 12 }
	};
	VkDescriptorPoolCreateInfo dpCI{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
	dpCI.maxSets       = 1;
	dpCI.poolSizeCount = 2;
	dpCI.pPoolSizes    = poolSizes;
	VkDescriptorPool descPool;
	vkCreateDescriptorPool(ctx.device, &dpCI, nullptr, &descPool);

	VkDescriptorSetAllocateInfo dsAI{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
	dsAI.descriptorPool     = descPool;
	dsAI.descriptorSetCount = 1;
	dsAI.pSetLayouts        = &dsLayout;
	VkDescriptorSet descSet;
	vkAllocateDescriptorSets(ctx.device, &dsAI, &descSet);

	// Write descriptors
	VkWriteDescriptorSetAccelerationStructureKHR asWrite{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
	asWrite.accelerationStructureCount = 1;
	asWrite.pAccelerationStructures    = &tlas.as;

	VkDescriptorBufferInfo bufInfos[12]{};
	auto setBI = [](VkDescriptorBufferInfo& bi, VkBuffer buf, VkDeviceSize sz) {
		bi.buffer = buf; bi.offset = 0; bi.range = sz;
	};
	setBI(bufInfos[0],  samplePosBuf.buf,      samplePosBuf.size);
	setBI(bufInfos[1],  sampleNormBuf.buf,     sampleNormBuf.size);
	setBI(bufInfos[2],  sampleFaceNormBuf.buf, sampleFaceNormBuf.size);
	setBI(bufInfos[3],  aoBuf.buf,             aoBuf.size);
	setBI(bufInfos[4],  geomInfoBuf.buf,       geomInfoBuf.size);
	setBI(bufInfos[5],  uvBuf.buf,             uvBuf.size);
	setBI(bufInfos[6],  normBuf.buf,           normBuf.size);
	setBI(bufInfos[7],  idxBuf.buf,            idxBuf.size);
	setBI(bufInfos[8],  posBuf.buf,            posBuf.size);
	setBI(bufInfos[9],  matBuf.buf,            matBuf.size);
	setBI(bufInfos[10], texInfoBuf.buf,        texInfoBuf.size);
	setBI(bufInfos[11], texDataBuf.buf,        texDataBuf.size);

	VkWriteDescriptorSet writes[13]{};
	// Binding 0: TLAS
	writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writes[0].pNext           = &asWrite;
	writes[0].dstSet          = descSet;
	writes[0].dstBinding      = 0;
	writes[0].descriptorCount = 1;
	writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
	// Bindings 1–12: storage buffers
	for (int i = 1; i <= 12; ++i)
	{
		writes[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writes[i].dstSet          = descSet;
		writes[i].dstBinding      = i;
		writes[i].descriptorCount = 1;
		writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		writes[i].pBufferInfo     = &bufInfos[i - 1];
	}
	vkUpdateDescriptorSets(ctx.device, 13, writes, 0, nullptr);

	// ── 9. Dispatch passes ──
	const int sqrtPasses = (int)lroundf(sqrtf((float)rays_per_sample));
	const size_t totalPasses = (size_t)sqrtPasses * sqrtPasses;
	const bool useOcclusionShortcut = (bounce_counts == 0) && !hasEmissive;
	const uint32_t workgroupSize = 128;
	const uint32_t numGroups = ((uint32_t)numSamples + workgroupSize - 1) / workgroupSize;

	memset(ao_values, 0, sizeof(float) * numSamples);
	cout_progress progress{totalPasses};

	for (int px = 0; px < sqrtPasses; ++px)
	{
		for (int py = 0; py < sqrtPasses; ++py)
		{
			PushConstants pc{};
			pc.px                   = px;
			pc.py                   = py;
			pc.sqrtPasses           = sqrtPasses;
			pc.numSamples           = (uint32_t)numSamples;
			pc.sceneOffsetH         = scene_offset_scale_horizontal;
			pc.sceneOffsetV         = scene_offset_scale_vertical;
			pc.bounceCounts         = bounce_counts;
			pc.useOcclusionShortcut = useOcclusionShortcut ? 1u : 0u;

			VkCommandBuffer cmd = beginCmd(ctx);

			// Memory barrier: ensure previous pass writes are visible
			if (px != 0 || py != 0)
			{
				VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
				mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
				mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
				vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
					VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
			}

			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeLayout, 0, 1, &descSet, 0, nullptr);
			vkCmdPushConstants(cmd, pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
			vkCmdDispatch(cmd, numGroups, 1, 1);

			endCmd(ctx, cmd);

			progress.report();
		}
	}

	// ── 10. Read back results ──
	GpuBuf readback = createBuffer(ctx, numSamples * sizeof(float),
		VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_CPU_ONLY);
	{
		VkCommandBuffer cmd = beginCmd(ctx);
		VkBufferCopy region{0, 0, numSamples * sizeof(float)};
		vkCmdCopyBuffer(cmd, aoBuf.buf, readback.buf, 1, &region);
		endCmd(ctx, cmd);
	}
	{
		void* mapped;
		vmaMapMemory(ctx.vma, readback.alloc, &mapped);
		memcpy(ao_values, mapped, numSamples * sizeof(float));
		vmaUnmapMemory(ctx.vma, readback.alloc);
	}

	// Normalize
	const float invTotal = 1.0f / float(rays_per_sample);
	for (size_t i = 0; i < numSamples; ++i)
		ao_values[i] *= invTotal;

	// ── 11. Cleanup ──
	destroyBuffer(ctx, readback);
	vkDestroyPipeline(ctx.device, pipeline, nullptr);
	vkDestroyPipelineLayout(ctx.device, pipeLayout, nullptr);
	vkDestroyDescriptorPool(ctx.device, descPool, nullptr);
	vkDestroyDescriptorSetLayout(ctx.device, dsLayout, nullptr);
	vkDestroyShaderModule(ctx.device, shaderModule, nullptr);

	destroyBuffer(ctx, aoBuf);
	destroyBuffer(ctx, sampleFaceNormBuf);
	destroyBuffer(ctx, sampleNormBuf);
	destroyBuffer(ctx, samplePosBuf);
	destroyBuffer(ctx, texDataBuf);
	destroyBuffer(ctx, texInfoBuf);
	destroyBuffer(ctx, matBuf);
	destroyBuffer(ctx, geomInfoBuf);
	destroyBuffer(ctx, normBuf);
	destroyBuffer(ctx, uvBuf);
	destroyBuffer(ctx, idxBuf);
	destroyBuffer(ctx, posBuf);

	// Destroy TLAS
	ctx.pfnDestroyAS(ctx.device, tlas.as, nullptr);
	destroyBuffer(ctx, tlas.buf);

	// Destroy BLASes
	for (auto& b : blases)
	{
		ctx.pfnDestroyAS(ctx.device, b.as, nullptr);
		destroyBuffer(ctx, b.buf);
	}
}
