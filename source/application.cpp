#include "application.hpp"

#include <imgui.h>

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>

namespace application {

namespace {

constexpr float kPi = 3.14159265359f;
constexpr uint32_t kMaxPointLights = 32;
constexpr uint32_t kMaxSpotLights = 32;
constexpr uint32_t kMaxLineVertices = 4096;

struct Vec3 {
	float x;
	float y;
	float z;
};

struct Vertex {
	float position[3];
	float normal[3];
};

struct LineVertex {
	float position[3];
	float pad;
};

struct Mat4 {
	float data[16]{};
};

struct SceneUniform {
	float viewProjection[16];
	float cameraPosition[4];
	float ambient[4];
	float directionalDirection[4];
	float directionalColor[4];
	int32_t lightCounts[4];
};

struct GpuPointLight {
	float position[4];
	float colorIntensity[4];
};

struct GpuSpotLight {
	float position[4];
	float direction[4];
	float colorIntensity[4];
	float angles[4];
};

struct ModelUniform {
	float model[16];
	float normalMatrix[16];
	float albedo[4];
	float specular[4];
};

struct PushConstants {
	float mvp[16];
	float color[4];
};

struct PointLight {
	float position[3];
	float color[3];
	float intensity;
};

struct SpotLight {
	float position[3];
	float direction[3];
	float color[3];
	float intensity;
	float innerAngle;
	float outerAngle;
};

struct Buffer {
	VkBuffer buffer = VK_NULL_HANDLE;
	VmaAllocation allocation = VK_NULL_HANDLE;
	void* mapped = nullptr;
};

struct Mesh {
	Buffer vertices;
	Buffer indices;
	uint32_t indexCount = 0;
};

static_assert(sizeof(SceneUniform) == 144, "SceneUniform must match std140");
static_assert(sizeof(GpuPointLight) == 32, "GpuPointLight must match std430");
static_assert(sizeof(GpuSpotLight) == 64, "GpuSpotLight must match std430");
static_assert(sizeof(ModelUniform) == 160, "ModelUniform must match std140");
static_assert(sizeof(PushConstants) == 80, "PushConstants size");

VkPipeline litPipeline = VK_NULL_HANDLE;
VkPipeline unlitMeshPipeline = VK_NULL_HANDLE;
VkPipeline unlitLinePipeline = VK_NULL_HANDLE;
VkPipelineLayout litPipelineLayout = VK_NULL_HANDLE;
VkPipelineLayout unlitPipelineLayout = VK_NULL_HANDLE;

VkDescriptorSetLayout sceneSetLayout = VK_NULL_HANDLE;
VkDescriptorSetLayout modelSetLayout = VK_NULL_HANDLE;
VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
VkDescriptorSet sceneSet = VK_NULL_HANDLE;
VkDescriptorSet sphereSet = VK_NULL_HANDLE;
VkDescriptorSet floorSet = VK_NULL_HANDLE;

VkShaderModule sphereVertexShader = VK_NULL_HANDLE;
VkShaderModule sphereFragmentShader = VK_NULL_HANDLE;
VkShaderModule unlitVertexShader = VK_NULL_HANDLE;
VkShaderModule unlitFragmentShader = VK_NULL_HANDLE;

Buffer sceneBuffer;
Buffer pointLightBuffer;
Buffer spotLightBuffer;
Buffer sphereModelBuffer;
Buffer floorModelBuffer;
Buffer lineBuffer;

Mesh sphereMesh;
Mesh floorMesh;

float cameraPosition[3] = {0.0f, 0.6f, 3.4f};
float cameraYaw = 0.0f;
float cameraPitch = -0.18f;
float cameraFov = 60.0f;

float ambientColor[3] = {1.0f, 1.0f, 1.0f};
float ambientIntensity = 0.08f;

float directionalDirection[3] = {-0.35f, -1.0f, -0.25f};
float directionalColor[3] = {1.0f, 0.98f, 0.95f};
float directionalIntensity = 0.75f;

float spherePosition[3] = {0.0f, 0.0f, 0.0f};
float angleX = 0.0f;
float angleY = 0.0f;
float angleZ = 0.0f;
float objectScale = 1.0f;

float albedo[3] = {0.2f, 0.55f, 1.0f};
float specularColor[3] = {1.0f, 1.0f, 1.0f};
float shininess = 48.0f;

float floorAlbedo[3] = {0.45f, 0.45f, 0.48f};
float floorSpecular[3] = {0.25f, 0.25f, 0.25f};
float floorShininess = 24.0f;

std::vector<PointLight> pointLights;
std::vector<SpotLight> spotLights;

bool showLightGizmos = true;
bool mouseLooking = false;
double lastMouseX = 0.0;
double lastMouseY = 0.0;
double lastTime = 0.0;

Vec3 vec3(float x, float y, float z)
{
	return {x, y, z};
}

Vec3 add(Vec3 a, Vec3 b)
{
	return {a.x + b.x, a.y + b.y, a.z + b.z};
}

Vec3 sub(Vec3 a, Vec3 b)
{
	return {a.x - b.x, a.y - b.y, a.z - b.z};
}

Vec3 mul(Vec3 v, float s)
{
	return {v.x * s, v.y * s, v.z * s};
}

float dot(Vec3 a, Vec3 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

float length(Vec3 v)
{
	return std::sqrt(dot(v, v));
}

Vec3 cross(Vec3 a, Vec3 b)
{
	return {
		a.y * b.z - a.z * b.y,
		a.z * b.x - a.x * b.z,
		a.x * b.y - a.y * b.x,
	};
}

Vec3 normalize(Vec3 v)
{
	const float len = length(v);
	if (len < 1.0e-6f) {
		return {0.0f, 1.0f, 0.0f};
	}
	return mul(v, 1.0f / len);
}

Mat4 identity()
{
	Mat4 result{};
	result.data[0] = 1.0f;
	result.data[5] = 1.0f;
	result.data[10] = 1.0f;
	result.data[15] = 1.0f;
	return result;
}

Mat4 multiply(const Mat4& a, const Mat4& b)
{
	Mat4 result{};
	for (int column = 0; column < 4; ++column) {
		for (int row = 0; row < 4; ++row) {
			result.data[column * 4 + row] =
				a.data[0 * 4 + row] * b.data[column * 4 + 0] +
				a.data[1 * 4 + row] * b.data[column * 4 + 1] +
				a.data[2 * 4 + row] * b.data[column * 4 + 2] +
				a.data[3 * 4 + row] * b.data[column * 4 + 3];
		}
	}
	return result;
}

Mat4 translation(float x, float y, float z)
{
	Mat4 result = identity();
	result.data[12] = x;
	result.data[13] = y;
	result.data[14] = z;
	return result;
}

Mat4 scaling(float scale)
{
	Mat4 result = identity();
	result.data[0] = scale;
	result.data[5] = scale;
	result.data[10] = scale;
	return result;
}

Mat4 rotationX(float angle)
{
	Mat4 result = identity();
	const float c = std::cos(angle);
	const float s = std::sin(angle);
	result.data[5] = c;
	result.data[6] = s;
	result.data[9] = -s;
	result.data[10] = c;
	return result;
}

Mat4 rotationY(float angle)
{
	Mat4 result = identity();
	const float c = std::cos(angle);
	const float s = std::sin(angle);
	result.data[0] = c;
	result.data[2] = -s;
	result.data[8] = s;
	result.data[10] = c;
	return result;
}

Mat4 rotationZ(float angle)
{
	Mat4 result = identity();
	const float c = std::cos(angle);
	const float s = std::sin(angle);
	result.data[0] = c;
	result.data[1] = s;
	result.data[4] = -s;
	result.data[5] = c;
	return result;
}

Mat4 perspective(float fov, float aspect, float nearPlane, float farPlane)
{
	Mat4 result{};
	const float f = 1.0f / std::tan(fov * 0.5f);
	result.data[0] = f / aspect;
	result.data[5] = -f;
	result.data[10] = farPlane / (nearPlane - farPlane);
	result.data[11] = -1.0f;
	result.data[14] = (farPlane * nearPlane) / (nearPlane - farPlane);
	return result;
}

Mat4 lookAt(Vec3 eye, Vec3 center, Vec3 up)
{
	const Vec3 f = normalize(sub(center, eye));
	const Vec3 s = normalize(cross(f, up));
	const Vec3 u = cross(s, f);

	Mat4 result = identity();
	result.data[0] = s.x;
	result.data[1] = u.x;
	result.data[2] = -f.x;
	result.data[4] = s.y;
	result.data[5] = u.y;
	result.data[6] = -f.y;
	result.data[8] = s.z;
	result.data[9] = u.z;
	result.data[10] = -f.z;
	result.data[12] = -dot(s, eye);
	result.data[13] = -dot(u, eye);
	result.data[14] = dot(f, eye);
	return result;
}

Mat4 normalMatrix(const Mat4& model)
{
	const float m00 = model.data[0];
	const float m10 = model.data[1];
	const float m20 = model.data[2];
	const float m01 = model.data[4];
	const float m11 = model.data[5];
	const float m21 = model.data[6];
	const float m02 = model.data[8];
	const float m12 = model.data[9];
	const float m22 = model.data[10];

	const float c00 = m11 * m22 - m12 * m21;
	const float c01 = m12 * m20 - m10 * m22;
	const float c02 = m10 * m21 - m11 * m20;
	const float c10 = m02 * m21 - m01 * m22;
	const float c11 = m00 * m22 - m02 * m20;
	const float c12 = m01 * m20 - m00 * m21;
	const float c20 = m01 * m12 - m02 * m11;
	const float c21 = m02 * m10 - m00 * m12;
	const float c22 = m00 * m11 - m01 * m10;

	const float det = m00 * c00 + m01 * c01 + m02 * c02;
	const float invDet = 1.0f / det;

	Mat4 result = identity();
	result.data[0] = c00 * invDet;
	result.data[1] = c10 * invDet;
	result.data[2] = c20 * invDet;
	result.data[4] = c01 * invDet;
	result.data[5] = c11 * invDet;
	result.data[6] = c21 * invDet;
	result.data[8] = c02 * invDet;
	result.data[9] = c12 * invDet;
	result.data[10] = c22 * invDet;
	return result;
}

Vec3 cameraForward()
{
	const float cp = std::cos(cameraPitch);
	const float sp = std::sin(cameraPitch);
	const float cy = std::cos(cameraYaw);
	const float sy = std::sin(cameraYaw);
	return {sy * cp, sp, -cy * cp};
}

float radians(float degrees)
{
	return degrees * kPi / 180.0f;
}

std::vector<uint8_t> readShader(const char* filename)
{
	FILE* file = std::fopen(filename, "rb");
	if (file == nullptr) {
		std::cerr << "Cannot open shader: " << filename << '\n';
		return {};
	}

	std::fseek(file, 0, SEEK_END);
	const long size = std::ftell(file);
	std::fseek(file, 0, SEEK_SET);

	if (size <= 0) {
		std::fclose(file);
		return {};
	}

	std::vector<uint8_t> buffer(static_cast<size_t>(size));
	if (std::fread(buffer.data(), 1, buffer.size(), file) != buffer.size()) {
		std::cerr << "Cannot read shader: " << filename << '\n';
		buffer.clear();
	}
	std::fclose(file);
	return buffer;
}

VkShaderModule createShaderModule(const char* filename)
{
	const auto code = readShader(filename);
	if (code.empty()) {
		return VK_NULL_HANDLE;
	}

	VkShaderModuleCreateInfo createInfo{};
	createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	createInfo.codeSize = code.size();
	createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

	VkShaderModule shaderModule = VK_NULL_HANDLE;
	if (vkCreateShaderModule(graphics::internal::context.device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
		std::cerr << "Failed to create shader module: " << filename << '\n';
		return VK_NULL_HANDLE;
	}
	return shaderModule;
}

bool createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, Buffer& buffer)
{
	VkBufferCreateInfo info{};
	info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	info.size = size;
	info.usage = usage;
	info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	VmaAllocationCreateInfo allocationInfo{};
	allocationInfo.usage = VMA_MEMORY_USAGE_AUTO;
	allocationInfo.flags =
		VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
		VMA_ALLOCATION_CREATE_MAPPED_BIT;

	VmaAllocationInfo mapped{};
	if (vmaCreateBuffer(graphics::internal::context.allocator, &info, &allocationInfo,
						&buffer.buffer, &buffer.allocation, &mapped) != VK_SUCCESS) {
		std::cerr << "Failed to create buffer\n";
		return false;
	}

	buffer.mapped = mapped.pMappedData;
	return true;
}

void destroyBuffer(Buffer& buffer)
{
	if (buffer.buffer == VK_NULL_HANDLE) {
		return;
	}

	vmaDestroyBuffer(graphics::internal::context.allocator, buffer.buffer, buffer.allocation);
	buffer.buffer = VK_NULL_HANDLE;
	buffer.allocation = VK_NULL_HANDLE;
	buffer.mapped = nullptr;
}

void upload(const Buffer& buffer, const void* data, size_t size)
{
	if (size == 0 || buffer.mapped == nullptr) {
		return;
	}
	std::memcpy(buffer.mapped, data, size);
	vmaFlushAllocation(graphics::internal::context.allocator, buffer.allocation, 0, size);
}

bool createMesh(const std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices, Mesh& mesh)
{
	mesh.indexCount = static_cast<uint32_t>(indices.size());
	if (!createBuffer(vertices.size() * sizeof(Vertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, mesh.vertices)) {
		return false;
	}
	if (!createBuffer(indices.size() * sizeof(uint32_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, mesh.indices)) {
		return false;
	}
	upload(mesh.vertices, vertices.data(), vertices.size() * sizeof(Vertex));
	upload(mesh.indices, indices.data(), indices.size() * sizeof(uint32_t));
	return true;
}

bool createSphereMesh()
{
	constexpr uint32_t segments = 10;
	constexpr uint32_t rings = 9;

	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;
	vertices.reserve((segments + 1) * (rings + 1));
	indices.reserve(segments * rings * 6);

	for (uint32_t ring = 0; ring <= rings; ++ring) {
		const float v = static_cast<float>(ring) / static_cast<float>(rings);
		const float phi = v * kPi;

		for (uint32_t segment = 0; segment <= segments; ++segment) {
			const float u = static_cast<float>(segment) / static_cast<float>(segments);
			const float theta = u * 2.0f * kPi;
			const float x = std::sin(phi) * std::cos(theta);
			const float y = std::cos(phi);
			const float z = std::sin(phi) * std::sin(theta);

			Vertex vertex{};
			vertex.position[0] = x;
			vertex.position[1] = y;
			vertex.position[2] = z;
			vertex.normal[0] = x;
			vertex.normal[1] = y;
			vertex.normal[2] = z;
			vertices.push_back(vertex);
		}
	}

	for (uint32_t ring = 0; ring < rings; ++ring) {
		for (uint32_t segment = 0; segment < segments; ++segment) {
			const uint32_t current = ring * (segments + 1) + segment;
			const uint32_t next = current + segments + 1;

			indices.push_back(current);
			indices.push_back(next);
			indices.push_back(current + 1);

			indices.push_back(current + 1);
			indices.push_back(next);
			indices.push_back(next + 1);
		}
	}

	std::cout << "Sphere created: " << vertices.size() << " vertices, " << indices.size() << " indices\n";
	return createMesh(vertices, indices, sphereMesh);
}

bool createFloorMesh()
{
	const Vertex vertices[] = {
		{{-1.0f, 0.0f, -1.0f}, {0.0f, 1.0f, 0.0f}},
		{{1.0f, 0.0f, -1.0f}, {0.0f, 1.0f, 0.0f}},
		{{1.0f, 0.0f, 1.0f}, {0.0f, 1.0f, 0.0f}},
		{{-1.0f, 0.0f, 1.0f}, {0.0f, 1.0f, 0.0f}},
	};
	const uint32_t indices[] = {0, 1, 2, 0, 2, 3};
	return createMesh({vertices, vertices + 4}, {indices, indices + 6}, floorMesh);
}

bool createDescriptors()
{
	auto& device = graphics::internal::context.device;

	VkDescriptorSetLayoutBinding sceneBindings[3]{};
	sceneBindings[0].binding = 0;
	sceneBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	sceneBindings[0].descriptorCount = 1;
	sceneBindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

	sceneBindings[1].binding = 1;
	sceneBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	sceneBindings[1].descriptorCount = 1;
	sceneBindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

	sceneBindings[2].binding = 2;
	sceneBindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	sceneBindings[2].descriptorCount = 1;
	sceneBindings[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

	VkDescriptorSetLayoutCreateInfo sceneLayoutInfo{};
	sceneLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	sceneLayoutInfo.bindingCount = 3;
	sceneLayoutInfo.pBindings = sceneBindings;
	if (vkCreateDescriptorSetLayout(device, &sceneLayoutInfo, nullptr, &sceneSetLayout) != VK_SUCCESS) {
		std::cerr << "Failed to create scene descriptor set layout\n";
		return false;
	}

	VkDescriptorSetLayoutBinding modelBinding{};
	modelBinding.binding = 0;
	modelBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	modelBinding.descriptorCount = 1;
	modelBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

	VkDescriptorSetLayoutCreateInfo modelLayoutInfo{};
	modelLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	modelLayoutInfo.bindingCount = 1;
	modelLayoutInfo.pBindings = &modelBinding;
	if (vkCreateDescriptorSetLayout(device, &modelLayoutInfo, nullptr, &modelSetLayout) != VK_SUCCESS) {
		std::cerr << "Failed to create model descriptor set layout\n";
		return false;
	}

	VkDescriptorPoolSize poolSizes[2]{};
	poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	poolSizes[0].descriptorCount = 3;
	poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	poolSizes[1].descriptorCount = 2;

	VkDescriptorPoolCreateInfo poolInfo{};
	poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	poolInfo.maxSets = 3;
	poolInfo.poolSizeCount = 2;
	poolInfo.pPoolSizes = poolSizes;
	if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
		std::cerr << "Failed to create descriptor pool\n";
		return false;
	}

	const VkDescriptorSetLayout layouts[] = {sceneSetLayout, modelSetLayout, modelSetLayout};
	VkDescriptorSet sets[3]{};

	VkDescriptorSetAllocateInfo allocateInfo{};
	allocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	allocateInfo.descriptorPool = descriptorPool;
	allocateInfo.descriptorSetCount = 3;
	allocateInfo.pSetLayouts = layouts;
	if (vkAllocateDescriptorSets(device, &allocateInfo, sets) != VK_SUCCESS) {
		std::cerr << "Failed to allocate descriptor sets\n";
		return false;
	}

	sceneSet = sets[0];
	sphereSet = sets[1];
	floorSet = sets[2];

	VkDescriptorBufferInfo sceneInfo{sceneBuffer.buffer, 0, sizeof(SceneUniform)};
	VkDescriptorBufferInfo pointInfo{pointLightBuffer.buffer, 0, VK_WHOLE_SIZE};
	VkDescriptorBufferInfo spotInfo{spotLightBuffer.buffer, 0, VK_WHOLE_SIZE};
	VkDescriptorBufferInfo sphereInfo{sphereModelBuffer.buffer, 0, sizeof(ModelUniform)};
	VkDescriptorBufferInfo floorInfo{floorModelBuffer.buffer, 0, sizeof(ModelUniform)};

	VkWriteDescriptorSet writes[5]{};
	writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writes[0].dstSet = sceneSet;
	writes[0].dstBinding = 0;
	writes[0].descriptorCount = 1;
	writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	writes[0].pBufferInfo = &sceneInfo;

	writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writes[1].dstSet = sceneSet;
	writes[1].dstBinding = 1;
	writes[1].descriptorCount = 1;
	writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	writes[1].pBufferInfo = &pointInfo;

	writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writes[2].dstSet = sceneSet;
	writes[2].dstBinding = 2;
	writes[2].descriptorCount = 1;
	writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	writes[2].pBufferInfo = &spotInfo;

	writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writes[3].dstSet = sphereSet;
	writes[3].dstBinding = 0;
	writes[3].descriptorCount = 1;
	writes[3].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	writes[3].pBufferInfo = &sphereInfo;

	writes[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writes[4].dstSet = floorSet;
	writes[4].dstBinding = 0;
	writes[4].descriptorCount = 1;
	writes[4].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	writes[4].pBufferInfo = &floorInfo;

	vkUpdateDescriptorSets(device, 5, writes, 0, nullptr);
	return true;
}

bool createLitPipeline()
{
	VkPipelineShaderStageCreateInfo stages[2]{};
	stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
	stages[0].module = sphereVertexShader;
	stages[0].pName = "main";
	stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
	stages[1].module = sphereFragmentShader;
	stages[1].pName = "main";

	VkVertexInputBindingDescription binding{};
	binding.binding = 0;
	binding.stride = sizeof(Vertex);
	binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

	VkVertexInputAttributeDescription attributes[2]{};
	attributes[0].location = 0;
	attributes[0].binding = 0;
	attributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
	attributes[0].offset = offsetof(Vertex, position);
	attributes[1].location = 1;
	attributes[1].binding = 0;
	attributes[1].format = VK_FORMAT_R32G32B32_SFLOAT;
	attributes[1].offset = offsetof(Vertex, normal);

	VkPipelineVertexInputStateCreateInfo vertexInput{};
	vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	vertexInput.vertexBindingDescriptionCount = 1;
	vertexInput.pVertexBindingDescriptions = &binding;
	vertexInput.vertexAttributeDescriptionCount = 2;
	vertexInput.pVertexAttributeDescriptions = attributes;

	VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
	inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
	inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

	VkPipelineViewportStateCreateInfo viewportState{};
	viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	viewportState.viewportCount = 1;
	viewportState.scissorCount = 1;

	VkPipelineRasterizationStateCreateInfo rasterizer{};
	rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
	rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
	rasterizer.cullMode = VK_CULL_MODE_NONE;
	rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
	rasterizer.lineWidth = 1.0f;

	VkPipelineMultisampleStateCreateInfo multisampling{};
	multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
	multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

	VkPipelineDepthStencilStateCreateInfo depthStencil{};
	depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
	depthStencil.depthTestEnable = VK_TRUE;
	depthStencil.depthWriteEnable = VK_TRUE;
	depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;

	VkPipelineColorBlendAttachmentState colorBlendAttachment{};
	colorBlendAttachment.colorWriteMask =
		VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
		VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

	VkPipelineColorBlendStateCreateInfo colorBlending{};
	colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	colorBlending.attachmentCount = 1;
	colorBlending.pAttachments = &colorBlendAttachment;

	VkDynamicState dynamicStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
	VkPipelineDynamicStateCreateInfo dynamicState{};
	dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
	dynamicState.dynamicStateCount = 2;
	dynamicState.pDynamicStates = dynamicStates;

	const VkDescriptorSetLayout setLayouts[] = {sceneSetLayout, modelSetLayout};
	VkPipelineLayoutCreateInfo layoutInfo{};
	layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	layoutInfo.setLayoutCount = 2;
	layoutInfo.pSetLayouts = setLayouts;
	if (vkCreatePipelineLayout(graphics::internal::context.device, &layoutInfo, nullptr, &litPipelineLayout) != VK_SUCCESS) {
		std::cerr << "Failed to create lit pipeline layout\n";
		return false;
	}

	VkGraphicsPipelineCreateInfo pipelineInfo{};
	pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	pipelineInfo.stageCount = 2;
	pipelineInfo.pStages = stages;
	pipelineInfo.pVertexInputState = &vertexInput;
	pipelineInfo.pInputAssemblyState = &inputAssembly;
	pipelineInfo.pViewportState = &viewportState;
	pipelineInfo.pRasterizationState = &rasterizer;
	pipelineInfo.pMultisampleState = &multisampling;
	pipelineInfo.pDepthStencilState = &depthStencil;
	pipelineInfo.pColorBlendState = &colorBlending;
	pipelineInfo.pDynamicState = &dynamicState;
	pipelineInfo.layout = litPipelineLayout;
	pipelineInfo.renderPass = graphics::internal::context.render_pass;
	pipelineInfo.subpass = 0;

	if (vkCreateGraphicsPipelines(graphics::internal::context.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &litPipeline) != VK_SUCCESS) {
		std::cerr << "Failed to create lit pipeline\n";
		return false;
	}
	return true;
}

bool createUnlitPipeline(VkPrimitiveTopology topology, uint32_t stride, VkPipeline& pipeline)
{
	VkPipelineShaderStageCreateInfo stages[2]{};
	stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
	stages[0].module = unlitVertexShader;
	stages[0].pName = "main";
	stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
	stages[1].module = unlitFragmentShader;
	stages[1].pName = "main";

	VkVertexInputBindingDescription binding{};
	binding.binding = 0;
	binding.stride = stride;
	binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

	VkVertexInputAttributeDescription attribute{};
	attribute.location = 0;
	attribute.binding = 0;
	attribute.format = VK_FORMAT_R32G32B32_SFLOAT;
	attribute.offset = 0;

	VkPipelineVertexInputStateCreateInfo vertexInput{};
	vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	vertexInput.vertexBindingDescriptionCount = 1;
	vertexInput.pVertexBindingDescriptions = &binding;
	vertexInput.vertexAttributeDescriptionCount = 1;
	vertexInput.pVertexAttributeDescriptions = &attribute;

	VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
	inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
	inputAssembly.topology = topology;

	VkPipelineViewportStateCreateInfo viewportState{};
	viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	viewportState.viewportCount = 1;
	viewportState.scissorCount = 1;

	VkPipelineRasterizationStateCreateInfo rasterizer{};
	rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
	rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
	rasterizer.cullMode = VK_CULL_MODE_NONE;
	rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
	rasterizer.lineWidth = 1.0f;

	VkPipelineMultisampleStateCreateInfo multisampling{};
	multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
	multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

	VkPipelineDepthStencilStateCreateInfo depthStencil{};
	depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
	depthStencil.depthTestEnable = VK_TRUE;
	depthStencil.depthWriteEnable = VK_TRUE;
	depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;

	VkPipelineColorBlendAttachmentState colorBlendAttachment{};
	colorBlendAttachment.colorWriteMask =
		VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
		VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

	VkPipelineColorBlendStateCreateInfo colorBlending{};
	colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	colorBlending.attachmentCount = 1;
	colorBlending.pAttachments = &colorBlendAttachment;

	VkDynamicState dynamicStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
	VkPipelineDynamicStateCreateInfo dynamicState{};
	dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
	dynamicState.dynamicStateCount = 2;
	dynamicState.pDynamicStates = dynamicStates;

	VkGraphicsPipelineCreateInfo pipelineInfo{};
	pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	pipelineInfo.stageCount = 2;
	pipelineInfo.pStages = stages;
	pipelineInfo.pVertexInputState = &vertexInput;
	pipelineInfo.pInputAssemblyState = &inputAssembly;
	pipelineInfo.pViewportState = &viewportState;
	pipelineInfo.pRasterizationState = &rasterizer;
	pipelineInfo.pMultisampleState = &multisampling;
	pipelineInfo.pDepthStencilState = &depthStencil;
	pipelineInfo.pColorBlendState = &colorBlending;
	pipelineInfo.pDynamicState = &dynamicState;
	pipelineInfo.layout = unlitPipelineLayout;
	pipelineInfo.renderPass = graphics::internal::context.render_pass;
	pipelineInfo.subpass = 0;

	if (vkCreateGraphicsPipelines(graphics::internal::context.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline) != VK_SUCCESS) {
		std::cerr << "Failed to create unlit pipeline\n";
		return false;
	}
	return true;
}

bool createPipelines()
{
	sphereVertexShader = createShaderModule("shaders/sphere.vert.spv");
	sphereFragmentShader = createShaderModule("shaders/sphere.frag.spv");
	unlitVertexShader = createShaderModule("shaders/unlit.vert.spv");
	unlitFragmentShader = createShaderModule("shaders/unlit.frag.spv");
	if (sphereVertexShader == VK_NULL_HANDLE || sphereFragmentShader == VK_NULL_HANDLE ||
		unlitVertexShader == VK_NULL_HANDLE || unlitFragmentShader == VK_NULL_HANDLE) {
		return false;
	}

	if (!createLitPipeline()) {
		return false;
	}

	VkPushConstantRange pushConstantRange{};
	pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
	pushConstantRange.offset = 0;
	pushConstantRange.size = sizeof(PushConstants);

	VkPipelineLayoutCreateInfo layoutInfo{};
	layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	layoutInfo.pushConstantRangeCount = 1;
	layoutInfo.pPushConstantRanges = &pushConstantRange;
	if (vkCreatePipelineLayout(graphics::internal::context.device, &layoutInfo, nullptr, &unlitPipelineLayout) != VK_SUCCESS) {
		std::cerr << "Failed to create unlit pipeline layout\n";
		return false;
	}

	if (!createUnlitPipeline(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, sizeof(Vertex), unlitMeshPipeline)) {
		return false;
	}
	if (!createUnlitPipeline(VK_PRIMITIVE_TOPOLOGY_LINE_LIST, sizeof(LineVertex), unlitLinePipeline)) {
		return false;
	}
	return true;
}

void destroyMesh(Mesh& mesh)
{
	destroyBuffer(mesh.vertices);
	destroyBuffer(mesh.indices);
	mesh.indexCount = 0;
}

void cleanup()
{
	auto& context = graphics::internal::context;
	if (context.device == VK_NULL_HANDLE) {
		return;
	}

	vkQueueWaitIdle(context.graphics_queue);

	if (litPipeline != VK_NULL_HANDLE) {
		vkDestroyPipeline(context.device, litPipeline, nullptr);
		litPipeline = VK_NULL_HANDLE;
	}
	if (unlitMeshPipeline != VK_NULL_HANDLE) {
		vkDestroyPipeline(context.device, unlitMeshPipeline, nullptr);
		unlitMeshPipeline = VK_NULL_HANDLE;
	}
	if (unlitLinePipeline != VK_NULL_HANDLE) {
		vkDestroyPipeline(context.device, unlitLinePipeline, nullptr);
		unlitLinePipeline = VK_NULL_HANDLE;
	}
	if (litPipelineLayout != VK_NULL_HANDLE) {
		vkDestroyPipelineLayout(context.device, litPipelineLayout, nullptr);
		litPipelineLayout = VK_NULL_HANDLE;
	}
	if (unlitPipelineLayout != VK_NULL_HANDLE) {
		vkDestroyPipelineLayout(context.device, unlitPipelineLayout, nullptr);
		unlitPipelineLayout = VK_NULL_HANDLE;
	}

	if (descriptorPool != VK_NULL_HANDLE) {
		vkDestroyDescriptorPool(context.device, descriptorPool, nullptr);
		descriptorPool = VK_NULL_HANDLE;
		sceneSet = VK_NULL_HANDLE;
		sphereSet = VK_NULL_HANDLE;
		floorSet = VK_NULL_HANDLE;
	}
	if (sceneSetLayout != VK_NULL_HANDLE) {
		vkDestroyDescriptorSetLayout(context.device, sceneSetLayout, nullptr);
		sceneSetLayout = VK_NULL_HANDLE;
	}
	if (modelSetLayout != VK_NULL_HANDLE) {
		vkDestroyDescriptorSetLayout(context.device, modelSetLayout, nullptr);
		modelSetLayout = VK_NULL_HANDLE;
	}

	VkShaderModule* shaders[] = {&sphereVertexShader, &sphereFragmentShader, &unlitVertexShader, &unlitFragmentShader};
	for (VkShaderModule* shader : shaders) {
		if (*shader != VK_NULL_HANDLE) {
			vkDestroyShaderModule(context.device, *shader, nullptr);
			*shader = VK_NULL_HANDLE;
		}
	}

	destroyBuffer(sceneBuffer);
	destroyBuffer(pointLightBuffer);
	destroyBuffer(spotLightBuffer);
	destroyBuffer(sphereModelBuffer);
	destroyBuffer(floorModelBuffer);
	destroyBuffer(lineBuffer);
	destroyMesh(sphereMesh);
	destroyMesh(floorMesh);
}

void packColor(const float color[3], float intensity, float out[4])
{
	out[0] = color[0];
	out[1] = color[1];
	out[2] = color[2];
	out[3] = intensity;
}

void writeModelUniform(Buffer& buffer, const Mat4& model, const float albedoRgb[3], const float specularRgb[3], float specularPower)
{
	ModelUniform uniform{};
	const Mat4 normals = normalMatrix(model);
	std::memcpy(uniform.model, model.data, sizeof(uniform.model));
	std::memcpy(uniform.normalMatrix, normals.data, sizeof(uniform.normalMatrix));
	uniform.albedo[0] = albedoRgb[0];
	uniform.albedo[1] = albedoRgb[1];
	uniform.albedo[2] = albedoRgb[2];
	uniform.albedo[3] = 1.0f;
	uniform.specular[0] = specularRgb[0];
	uniform.specular[1] = specularRgb[1];
	uniform.specular[2] = specularRgb[2];
	uniform.specular[3] = specularPower;
	upload(buffer, &uniform, sizeof(uniform));
}

Mat4 sphereModelMatrix()
{
	Mat4 model = translation(spherePosition[0], spherePosition[1], spherePosition[2]);
	model = multiply(model, rotationZ(angleZ));
	model = multiply(model, rotationY(angleY));
	model = multiply(model, rotationX(angleX));
	model = multiply(model, scaling(objectScale));
	return model;
}

void appendLine(std::vector<LineVertex>& lines, Vec3 a, Vec3 b)
{
	if (lines.size() + 2 > kMaxLineVertices) {
		return;
	}
	lines.push_back({{a.x, a.y, a.z}, 0.0f});
	lines.push_back({{b.x, b.y, b.z}, 0.0f});
}

std::vector<LineVertex> buildSpotCones()
{
	std::vector<LineVertex> lines;
	lines.reserve(spotLights.size() * 64);

	constexpr int segments = 24;
	constexpr float coneLength = 1.7f;

	for (const SpotLight& spot : spotLights) {
		const Vec3 direction = normalize(vec3(spot.direction[0], spot.direction[1], spot.direction[2]));
		const Vec3 origin = vec3(spot.position[0], spot.position[1], spot.position[2]);
		const Vec3 helper = std::abs(direction.y) < 0.9f ? vec3(0.0f, 1.0f, 0.0f) : vec3(1.0f, 0.0f, 0.0f);
		const Vec3 right = normalize(cross(helper, direction));
		const Vec3 up = cross(direction, right);
		const float radius = std::tan(radians(spot.outerAngle)) * coneLength;
		const Vec3 center = add(origin, mul(direction, coneLength));

		Vec3 previous{};
		for (int i = 0; i <= segments; ++i) {
			const float angle = 2.0f * kPi * static_cast<float>(i % segments) / static_cast<float>(segments);
			const Vec3 point = add(center, add(mul(right, std::cos(angle) * radius), mul(up, std::sin(angle) * radius)));
			if (i > 0) {
				appendLine(lines, previous, point);
			}
			if (i < segments && i % 3 == 0) {
				appendLine(lines, origin, point);
			}
			previous = point;
		}
	}

	return lines;
}

void drawIndexed(VkCommandBuffer commandBuffer, const Mesh& mesh)
{
	VkDeviceSize offset = 0;
	vkCmdBindVertexBuffers(commandBuffer, 0, 1, &mesh.vertices.buffer, &offset);
	vkCmdBindIndexBuffer(commandBuffer, mesh.indices.buffer, 0, VK_INDEX_TYPE_UINT32);
	vkCmdDrawIndexed(commandBuffer, mesh.indexCount, 1, 0, 0, 0);
}

void drawUnlitSphere(VkCommandBuffer commandBuffer, const Mat4& viewProjection, Vec3 position, float scale, const float color[3])
{
	const Mat4 model = multiply(translation(position.x, position.y, position.z), scaling(scale));
	const Mat4 mvp = multiply(viewProjection, model);

	PushConstants push{};
	std::memcpy(push.mvp, mvp.data, sizeof(push.mvp));
	push.color[0] = color[0];
	push.color[1] = color[1];
	push.color[2] = color[2];
	push.color[3] = 1.0f;

	vkCmdPushConstants(commandBuffer, unlitPipelineLayout,
					   VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
					   0, sizeof(push), &push);
	drawIndexed(commandBuffer, sphereMesh);
}

} // namespace

bool initialize()
{
	pointLights.clear();
	spotLights.clear();

	PointLight point{};
	point.position[0] = 1.7f;
	point.position[1] = 1.5f;
	point.position[2] = 1.1f;
	point.color[0] = 1.0f;
	point.color[1] = 0.62f;
	point.color[2] = 0.28f;
	point.intensity = 14.0f;
	pointLights.push_back(point);

	SpotLight spot{};
	spot.position[0] = -2.1f;
	spot.position[1] = 2.5f;
	spot.position[2] = 1.3f;
	spot.direction[0] = 2.1f;
	spot.direction[1] = -2.5f;
	spot.direction[2] = -1.3f;
	spot.color[0] = 0.65f;
	spot.color[1] = 0.82f;
	spot.color[2] = 1.0f;
	spot.intensity = 36.0f;
	spot.innerAngle = 12.0f;
	spot.outerAngle = 26.0f;
	spotLights.push_back(spot);

	if (!createSphereMesh() || !createFloorMesh()) {
		cleanup();
		return false;
	}

	const bool buffersOk =
		createBuffer(sizeof(SceneUniform), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, sceneBuffer) &&
		createBuffer(sizeof(GpuPointLight) * kMaxPointLights, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, pointLightBuffer) &&
		createBuffer(sizeof(GpuSpotLight) * kMaxSpotLights, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, spotLightBuffer) &&
		createBuffer(sizeof(ModelUniform), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, sphereModelBuffer) &&
		createBuffer(sizeof(ModelUniform), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, floorModelBuffer) &&
		createBuffer(sizeof(LineVertex) * kMaxLineVertices, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, lineBuffer);

	if (!buffersOk || !createDescriptors() || !createPipelines()) {
		cleanup();
		return false;
	}

	return true;
}

void shutdown()
{
	cleanup();
}

void update(double time)
{
	float dt = static_cast<float>(time - lastTime);
	lastTime = time;
	if (dt < 0.0f || dt > 0.1f) {
		dt = 0.0f;
	}

	ImGuiIO& io = ImGui::GetIO();

	if (!io.WantCaptureMouse) {
		if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
			mouseLooking = true;
			lastMouseX = io.MousePos.x;
			lastMouseY = io.MousePos.y;
		}
		if (ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
			mouseLooking = false;
		}
		if (mouseLooking) {
			const float deltaX = static_cast<float>(io.MousePos.x - lastMouseX);
			const float deltaY = static_cast<float>(io.MousePos.y - lastMouseY);
			cameraYaw += deltaX * 0.005f;
			cameraPitch -= deltaY * 0.005f;
			const float limit = radians(89.0f);
			cameraPitch = std::clamp(cameraPitch, -limit, limit);
			lastMouseX = io.MousePos.x;
			lastMouseY = io.MousePos.y;
		}
	} else if (!ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
		mouseLooking = false;
	}

	if (!io.WantCaptureKeyboard) {
		const Vec3 forward = cameraForward();
		const Vec3 right = normalize(cross(forward, vec3(0.0f, 1.0f, 0.0f)));
		Vec3 move{};
		if (ImGui::IsKeyDown(ImGuiKey_W)) {
			move = add(move, forward);
		}
		if (ImGui::IsKeyDown(ImGuiKey_S)) {
			move = sub(move, forward);
		}
		if (ImGui::IsKeyDown(ImGuiKey_D)) {
			move = add(move, right);
		}
		if (ImGui::IsKeyDown(ImGuiKey_A)) {
			move = sub(move, right);
		}
		if (ImGui::IsKeyDown(ImGuiKey_E)) {
			move.y += 1.0f;
		}
		if (ImGui::IsKeyDown(ImGuiKey_Q)) {
			move.y -= 1.0f;
		}
		if (dot(move, move) > 0.0f) {
			move = mul(normalize(move), 3.0f * dt);
			cameraPosition[0] += move.x;
			cameraPosition[1] += move.y;
			cameraPosition[2] += move.z;
		}
	}

	ImGui::SetNextWindowSize(ImVec2(400.0f, 680.0f), ImGuiCond_FirstUseEver);
	ImGui::Begin("Lab 2 - Camera and lighting");
	ImGui::TextUnformatted("Variant 5: sphere, 110 vertices");
	ImGui::TextUnformatted("Look: left drag. Move: WASD, Q/E.");
	ImGui::Checkbox("Show light gizmos", &showLightGizmos);

	if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::DragFloat3("Position##camera", cameraPosition, 0.05f, -20.0f, 20.0f);
		ImGui::SliderAngle("Yaw", &cameraYaw, -180.0f, 180.0f);
		ImGui::SliderAngle("Pitch", &cameraPitch, -89.0f, 89.0f);
		ImGui::SliderFloat("Field of view", &cameraFov, 20.0f, 100.0f);
	}

	if (ImGui::CollapsingHeader("Ambient", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::ColorEdit3("Ambient color", ambientColor);
		ImGui::SliderFloat("Ambient intensity", &ambientIntensity, 0.0f, 1.0f);
	}

	if (ImGui::CollapsingHeader("Directional light", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::DragFloat3("Direction", directionalDirection, 0.01f, -1.0f, 1.0f);
		ImGui::ColorEdit3("Directional color", directionalColor);
		ImGui::SliderFloat("Directional intensity", &directionalIntensity, 0.0f, 3.0f);
	}

	if (ImGui::CollapsingHeader("Sphere", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::DragFloat3("Position##sphere", spherePosition, 0.02f, -5.0f, 5.0f);
		ImGui::SliderAngle("Rotation X", &angleX, -180.0f, 180.0f);
		ImGui::SliderAngle("Rotation Y", &angleY, -180.0f, 180.0f);
		ImGui::SliderAngle("Rotation Z", &angleZ, -180.0f, 180.0f);
		ImGui::SliderFloat("Scale", &objectScale, 0.1f, 3.0f);
		ImGui::ColorEdit3("Albedo", albedo);
		ImGui::ColorEdit3("Specular color", specularColor);
		ImGui::SliderFloat("Shininess", &shininess, 1.0f, 128.0f);
	}

	if (ImGui::CollapsingHeader("Point lights", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::BeginDisabled(pointLights.size() >= kMaxPointLights);
		if (ImGui::Button("Add point light")) {
			PointLight light{};
			light.position[0] = 0.0f;
			light.position[1] = 1.4f;
			light.position[2] = 0.0f;
			light.color[0] = 1.0f;
			light.color[1] = 1.0f;
			light.color[2] = 1.0f;
			light.intensity = 10.0f;
			pointLights.push_back(light);
		}
		ImGui::EndDisabled();

		for (int i = 0; i < static_cast<int>(pointLights.size()); ++i) {
			ImGui::PushID(i);
			ImGui::Separator();
			ImGui::Text("Point %d", i);
			if (ImGui::Button("Remove##point")) {
				pointLights.erase(pointLights.begin() + i);
				ImGui::PopID();
				break;
			}
			ImGui::DragFloat3("Position", pointLights[i].position, 0.02f, -8.0f, 8.0f);
			ImGui::ColorEdit3("Color", pointLights[i].color);
			ImGui::SliderFloat("Intensity", &pointLights[i].intensity, 0.0f, 80.0f);
			ImGui::PopID();
		}
	}

	if (ImGui::CollapsingHeader("Spot lights", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::BeginDisabled(spotLights.size() >= kMaxSpotLights);
		if (ImGui::Button("Add spot light")) {
			SpotLight light{};
			light.position[0] = 0.0f;
			light.position[1] = 2.2f;
			light.position[2] = 0.0f;
			light.direction[0] = 0.0f;
			light.direction[1] = -1.0f;
			light.direction[2] = 0.0f;
			light.color[0] = 1.0f;
			light.color[1] = 1.0f;
			light.color[2] = 1.0f;
			light.intensity = 30.0f;
			light.innerAngle = 12.0f;
			light.outerAngle = 24.0f;
			spotLights.push_back(light);
		}
		ImGui::EndDisabled();

		for (int i = 0; i < static_cast<int>(spotLights.size()); ++i) {
			ImGui::PushID(1000 + i);
			ImGui::Separator();
			ImGui::Text("Spot %d", i);
			if (ImGui::Button("Remove##spot")) {
				spotLights.erase(spotLights.begin() + i);
				ImGui::PopID();
				break;
			}
			ImGui::DragFloat3("Position", spotLights[i].position, 0.02f, -8.0f, 8.0f);
			ImGui::DragFloat3("Direction", spotLights[i].direction, 0.02f, -5.0f, 5.0f);
			ImGui::ColorEdit3("Color", spotLights[i].color);
			ImGui::SliderFloat("Intensity", &spotLights[i].intensity, 0.0f, 80.0f);
			ImGui::SliderFloat("Inner angle", &spotLights[i].innerAngle, 1.0f, 70.0f);
			ImGui::SliderFloat("Outer angle", &spotLights[i].outerAngle, 1.0f, 80.0f);
			if (spotLights[i].outerAngle < spotLights[i].innerAngle) {
				spotLights[i].outerAngle = spotLights[i].innerAngle;
			}
			ImGui::PopID();
		}
	}

	ImGui::End();
}

void render(const graphics::internal::FrameData& fd)
{
	auto& context = graphics::internal::context;
	const float width = static_cast<float>(context.swapchain_extent.width);
	const float height = static_cast<float>(context.swapchain_extent.height);
	const float aspect = height > 0.0f ? width / height : 1.0f;

	const Vec3 eye = vec3(cameraPosition[0], cameraPosition[1], cameraPosition[2]);
	const Mat4 view = lookAt(eye, add(eye, cameraForward()), vec3(0.0f, 1.0f, 0.0f));
	const Mat4 projection = perspective(radians(cameraFov), aspect, 0.1f, 100.0f);
	const Mat4 viewProjection = multiply(projection, view);

	SceneUniform scene{};
	std::memcpy(scene.viewProjection, viewProjection.data, sizeof(scene.viewProjection));
	scene.cameraPosition[0] = cameraPosition[0];
	scene.cameraPosition[1] = cameraPosition[1];
	scene.cameraPosition[2] = cameraPosition[2];
	scene.cameraPosition[3] = 1.0f;
	packColor(ambientColor, ambientIntensity, scene.ambient);

	const Vec3 lightTravel = normalize(vec3(directionalDirection[0], directionalDirection[1], directionalDirection[2]));
	scene.directionalDirection[0] = lightTravel.x;
	scene.directionalDirection[1] = lightTravel.y;
	scene.directionalDirection[2] = lightTravel.z;
	packColor(directionalColor, directionalIntensity, scene.directionalColor);
	scene.lightCounts[0] = static_cast<int32_t>(pointLights.size());
	scene.lightCounts[1] = static_cast<int32_t>(spotLights.size());
	upload(sceneBuffer, &scene, sizeof(scene));

	std::vector<GpuPointLight> gpuPoints(pointLights.size());
	for (size_t i = 0; i < pointLights.size(); ++i) {
		gpuPoints[i].position[0] = pointLights[i].position[0];
		gpuPoints[i].position[1] = pointLights[i].position[1];
		gpuPoints[i].position[2] = pointLights[i].position[2];
		gpuPoints[i].position[3] = 1.0f;
		packColor(pointLights[i].color, pointLights[i].intensity, gpuPoints[i].colorIntensity);
	}
	if (!gpuPoints.empty()) {
		upload(pointLightBuffer, gpuPoints.data(), gpuPoints.size() * sizeof(GpuPointLight));
	}

	std::vector<GpuSpotLight> gpuSpots(spotLights.size());
	for (size_t i = 0; i < spotLights.size(); ++i) {
		const Vec3 direction = normalize(vec3(spotLights[i].direction[0], spotLights[i].direction[1], spotLights[i].direction[2]));
		gpuSpots[i].position[0] = spotLights[i].position[0];
		gpuSpots[i].position[1] = spotLights[i].position[1];
		gpuSpots[i].position[2] = spotLights[i].position[2];
		gpuSpots[i].position[3] = 1.0f;
		gpuSpots[i].direction[0] = direction.x;
		gpuSpots[i].direction[1] = direction.y;
		gpuSpots[i].direction[2] = direction.z;
		packColor(spotLights[i].color, spotLights[i].intensity, gpuSpots[i].colorIntensity);
		gpuSpots[i].angles[0] = std::cos(radians(spotLights[i].innerAngle));
		gpuSpots[i].angles[1] = std::cos(radians(spotLights[i].outerAngle));
	}
	if (!gpuSpots.empty()) {
		upload(spotLightBuffer, gpuSpots.data(), gpuSpots.size() * sizeof(GpuSpotLight));
	}

	writeModelUniform(sphereModelBuffer, sphereModelMatrix(), albedo, specularColor, shininess);
	const Mat4 floorModel = multiply(translation(0.0f, -1.15f, 0.0f), scaling(6.0f));
	writeModelUniform(floorModelBuffer, floorModel, floorAlbedo, floorSpecular, floorShininess);

	std::vector<LineVertex> coneLines;
	if (showLightGizmos) {
		coneLines = buildSpotCones();
		if (!coneLines.empty()) {
			upload(lineBuffer, coneLines.data(), coneLines.size() * sizeof(LineVertex));
		}
	}

	VkCommandBuffer commandBuffer = fd.command_buffer;
	VkCommandBufferBeginInfo beginInfo{};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	vkBeginCommandBuffer(commandBuffer, &beginInfo);

	VkClearValue clearValues[2]{};
	clearValues[0].color = {{0.02f, 0.025f, 0.03f, 1.0f}};
	clearValues[1].depthStencil = {1.0f, 0};

	VkRenderPassBeginInfo renderPassBegin{};
	renderPassBegin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	renderPassBegin.renderPass = context.render_pass;
	renderPassBegin.framebuffer = fd.framebuffer;
	renderPassBegin.renderArea.extent = context.swapchain_extent;
	renderPassBegin.clearValueCount = 2;
	renderPassBegin.pClearValues = clearValues;
	vkCmdBeginRenderPass(commandBuffer, &renderPassBegin, VK_SUBPASS_CONTENTS_INLINE);

	VkViewport viewport{};
	viewport.width = width;
	viewport.height = height;
	viewport.minDepth = 0.0f;
	viewport.maxDepth = 1.0f;
	vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

	VkRect2D scissor{};
	scissor.extent = context.swapchain_extent;
	vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, litPipeline);
	VkDescriptorSet sphereSets[] = {sceneSet, sphereSet};
	vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, litPipelineLayout, 0, 2, sphereSets, 0, nullptr);
	drawIndexed(commandBuffer, sphereMesh);

	VkDescriptorSet floorSets[] = {sceneSet, floorSet};
	vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, litPipelineLayout, 0, 2, floorSets, 0, nullptr);
	drawIndexed(commandBuffer, floorMesh);

	if (showLightGizmos) {
		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, unlitMeshPipeline);
		for (const PointLight& light : pointLights) {
			drawUnlitSphere(commandBuffer, viewProjection,
							vec3(light.position[0], light.position[1], light.position[2]),
							0.08f, light.color);
		}
		for (const SpotLight& light : spotLights) {
			drawUnlitSphere(commandBuffer, viewProjection,
							vec3(light.position[0], light.position[1], light.position[2]),
							0.08f, light.color);
		}

		if (!coneLines.empty()) {
			vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, unlitLinePipeline);
			PushConstants push{};
			std::memcpy(push.mvp, viewProjection.data, sizeof(push.mvp));
			push.color[0] = 1.0f;
			push.color[1] = 1.0f;
			push.color[2] = 1.0f;
			push.color[3] = 1.0f;
			vkCmdPushConstants(commandBuffer, unlitPipelineLayout,
							   VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
							   0, sizeof(push), &push);
			VkDeviceSize offset = 0;
			vkCmdBindVertexBuffers(commandBuffer, 0, 1, &lineBuffer.buffer, &offset);
			vkCmdDraw(commandBuffer, static_cast<uint32_t>(coneLines.size()), 1, 0, 0);
		}
	}

	vkCmdEndRenderPass(commandBuffer);
	vkEndCommandBuffer(commandBuffer);
}

} // namespace application
