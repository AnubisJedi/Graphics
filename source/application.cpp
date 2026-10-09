#include "application.hpp"

#include <imgui.h>

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

namespace application {

namespace {

struct Vertex
{
    float position[3];
    float normal[3];
};

struct Mat4
{
    float data[16]{};
};

struct PushConstants
{
    float mvp[16];
    float color[4];
};


// Vulkan objects

VkPipeline pipeline = VK_NULL_HANDLE;
VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;

VkShaderModule vertexShader = VK_NULL_HANDLE;
VkShaderModule fragmentShader = VK_NULL_HANDLE;

VkBuffer vertexBuffer = VK_NULL_HANDLE;
VmaAllocation vertexAllocation = VK_NULL_HANDLE;

VkBuffer indexBuffer = VK_NULL_HANDLE;
VmaAllocation indexAllocation = VK_NULL_HANDLE;

uint32_t indexCount = 0;


// Object parameters

float positionX = 0.0f;
float positionY = 0.0f;
float positionZ = 0.0f;

float angleX = 0.0f;
float angleY = 0.0f;
float angleZ = 0.0f;

bool mouseDragging = false;
double lastMouseX = 0.0;
double lastMouseY = 0.0;

float objectScale = 1.0f;

bool perspectiveProjection = true;

float objectColor[4] =
{
    0.2f,
    0.55f,
    1.0f,
    1.0f
};



// Matrix functions


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

    for (int column = 0; column < 4; ++column)
    {
        for (int row = 0; row < 4; ++row)
        {
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



// Projection matrices


Mat4 perspective(
    float fov,
    float aspect,
    float nearPlane,
    float farPlane)
{
    Mat4 result{};

    const float f =
        1.0f / std::tan(fov * 0.5f);

    result.data[0] = f / aspect;

    // Vulkan framebuffer Y direction
    result.data[5] = -f;

    result.data[10] =
        farPlane / (nearPlane - farPlane);

    result.data[11] = -1.0f;

    result.data[14] =
        (farPlane * nearPlane) /
        (nearPlane - farPlane);

    return result;
}


Mat4 orthographic(
    float width,
    float height,
    float nearPlane,
    float farPlane)
{
    Mat4 result = identity();

    result.data[0] = 2.0f / width;
    result.data[5] = -2.0f / height;

    result.data[10] =
        1.0f / (nearPlane - farPlane);

    result.data[14] =
        nearPlane / (nearPlane - farPlane);

    return result;
}



// Shader loading


std::vector<uint8_t> readShader(
    const char* filename)
{
    std::ifstream file(
        filename,
        std::ios::binary | std::ios::ate);

    if (!file)
    {
        std::cerr
            << "Cannot open shader: "
            << filename
            << '\n';

        return {};
    }

    const std::streamsize size =
        file.tellg();

    std::vector<uint8_t> buffer(
        static_cast<size_t>(size));

    file.seekg(0);

    file.read(
        reinterpret_cast<char*>(buffer.data()),
        size);

    return buffer;
}


VkShaderModule createShaderModule(
    const char* filename)
{
    const auto code =
        readShader(filename);

    if (code.empty())
        return VK_NULL_HANDLE;

    VkShaderModuleCreateInfo createInfo{};

    createInfo.sType =
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;

    createInfo.codeSize =
        code.size();

    createInfo.pCode =
        reinterpret_cast<const uint32_t*>(
            code.data());

    VkShaderModule shaderModule =
        VK_NULL_HANDLE;

    if (
        vkCreateShaderModule(
            graphics::internal::context.device,
            &createInfo,
            nullptr,
            &shaderModule)
        != VK_SUCCESS)
    {
        std::cerr
            << "Failed to create shader module\n";

        return VK_NULL_HANDLE;
    }

    return shaderModule;
}



// Sphere generation


bool createSphere()
{
    /*
        10 segments × 9 rings

        Vertices:
        (10 + 1) × (9 + 1)
        = 110 vertices

        This satisfies the requirement:
        "Sphere (~100 vertices)"
    */

    constexpr uint32_t segments = 10;
    constexpr uint32_t rings = 9;

    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;

    vertices.reserve(
        (segments + 1) * (rings + 1));

    indices.reserve(
        segments * rings * 6);


    for (uint32_t ring = 0;
         ring <= rings;
         ++ring)
    {
        const float v =
            static_cast<float>(ring) /
            static_cast<float>(rings);

        const float phi =
            v * 3.14159265359f;


        for (uint32_t segment = 0;
             segment <= segments;
             ++segment)
        {
            const float u =
                static_cast<float>(segment) /
                static_cast<float>(segments);

            const float theta =
                u * 2.0f *
                3.14159265359f;


            const float x =
                std::sin(phi) *
                std::cos(theta);

            const float y =
                std::cos(phi);

            const float z =
                std::sin(phi) *
                std::sin(theta);


            Vertex vertex{};

            vertex.position[0] = x;
            vertex.position[1] = y;
            vertex.position[2] = z;

            /*
                Для сферы нормаль совпадает
                с нормализованной позицией.
            */

            vertex.normal[0] = x;
            vertex.normal[1] = y;
            vertex.normal[2] = z;

            vertices.push_back(vertex);
        }
    }


    // Создание треугольников
    for (uint32_t ring = 0;
         ring < rings;
         ++ring)
    {
        for (uint32_t segment = 0;
             segment < segments;
             ++segment)
        {
            const uint32_t current =
                ring * (segments + 1) +
                segment;

            const uint32_t next =
                current + segments + 1;


            indices.push_back(current);
            indices.push_back(next);
            indices.push_back(current + 1);


            indices.push_back(current + 1);
            indices.push_back(next);
            indices.push_back(next + 1);
        }
    }


    
    // Vertex buffer
    

    VkBufferCreateInfo vertexBufferInfo{};

    vertexBufferInfo.sType =
        VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;

    vertexBufferInfo.size =
        vertices.size() * sizeof(Vertex);

    vertexBufferInfo.usage =
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;

    vertexBufferInfo.sharingMode =
        VK_SHARING_MODE_EXCLUSIVE;


    VmaAllocationCreateInfo allocationInfo{};

    allocationInfo.usage =
        VMA_MEMORY_USAGE_AUTO;

    allocationInfo.flags =
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
        VMA_ALLOCATION_CREATE_MAPPED_BIT;


    if (
        vmaCreateBuffer(
            graphics::internal::context.allocator,
            &vertexBufferInfo,
            &allocationInfo,
            &vertexBuffer,
            &vertexAllocation,
            nullptr)
        != VK_SUCCESS)
    {
        std::cerr
            << "Failed to create vertex buffer\n";

        return false;
    }


    VmaAllocationInfo vertexAllocationInfo{};

    vmaGetAllocationInfo(
        graphics::internal::context.allocator,
        vertexAllocation,
        &vertexAllocationInfo);


    std::memcpy(
        vertexAllocationInfo.pMappedData,
        vertices.data(),
        vertices.size() * sizeof(Vertex));


    
    // Index buffer
    

    VkBufferCreateInfo indexBufferInfo{};

    indexBufferInfo.sType =
        VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;

    indexBufferInfo.size =
        indices.size() * sizeof(uint32_t);

    indexBufferInfo.usage =
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT;

    indexBufferInfo.sharingMode =
        VK_SHARING_MODE_EXCLUSIVE;


    if (
        vmaCreateBuffer(
            graphics::internal::context.allocator,
            &indexBufferInfo,
            &allocationInfo,
            &indexBuffer,
            &indexAllocation,
            nullptr)
        != VK_SUCCESS)
    {
        std::cerr
            << "Failed to create index buffer\n";

        return false;
    }


    VmaAllocationInfo indexAllocationInfo{};

    vmaGetAllocationInfo(
        graphics::internal::context.allocator,
        indexAllocation,
        &indexAllocationInfo);


    std::memcpy(
        indexAllocationInfo.pMappedData,
        indices.data(),
        indices.size() * sizeof(uint32_t));


    indexCount =
        static_cast<uint32_t>(
            indices.size());


    std::cout
        << "Sphere created: "
        << vertices.size()
        << " vertices, "
        << indexCount
        << " indices\n";


    return true;
}



// Graphics pipeline


bool createPipeline()
{
    vertexShader =
        createShaderModule(
            "shaders/sphere.vert.spv");

    fragmentShader =
        createShaderModule(
            "shaders/sphere.frag.spv");


    if (
        vertexShader == VK_NULL_HANDLE ||
        fragmentShader == VK_NULL_HANDLE)
    {
        return false;
    }


    VkPushConstantRange pushConstantRange{};

    pushConstantRange.stageFlags =
        VK_SHADER_STAGE_VERTEX_BIT |
        VK_SHADER_STAGE_FRAGMENT_BIT;

    pushConstantRange.offset = 0;

    pushConstantRange.size =
        sizeof(PushConstants);


    VkPipelineLayoutCreateInfo layoutInfo{};

    layoutInfo.sType =
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;

    layoutInfo.pushConstantRangeCount = 1;

    layoutInfo.pPushConstantRanges =
        &pushConstantRange;


    if (
        vkCreatePipelineLayout(
            graphics::internal::context.device,
            &layoutInfo,
            nullptr,
            &pipelineLayout)
        != VK_SUCCESS)
    {
        return false;
    }


    VkPipelineShaderStageCreateInfo shaderStages[2]{};


    shaderStages[0].sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;

    shaderStages[0].stage =
        VK_SHADER_STAGE_VERTEX_BIT;

    shaderStages[0].module =
        vertexShader;

    shaderStages[0].pName =
        "main";


    shaderStages[1].sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;

    shaderStages[1].stage =
        VK_SHADER_STAGE_FRAGMENT_BIT;

    shaderStages[1].module =
        fragmentShader;

    shaderStages[1].pName =
        "main";


    VkVertexInputBindingDescription binding{};

    binding.binding = 0;

    binding.stride =
        sizeof(Vertex);

    binding.inputRate =
        VK_VERTEX_INPUT_RATE_VERTEX;


    VkVertexInputAttributeDescription attributes[2]{};


    attributes[0].location = 0;
    attributes[0].binding = 0;

    attributes[0].format =
        VK_FORMAT_R32G32B32_SFLOAT;

    attributes[0].offset =
        offsetof(Vertex, position);


    attributes[1].location = 1;
    attributes[1].binding = 0;

    attributes[1].format =
        VK_FORMAT_R32G32B32_SFLOAT;

    attributes[1].offset =
        offsetof(Vertex, normal);


    VkPipelineVertexInputStateCreateInfo vertexInput{};

    vertexInput.sType =
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    vertexInput.vertexBindingDescriptionCount = 1;

    vertexInput.pVertexBindingDescriptions =
        &binding;

    vertexInput.vertexAttributeDescriptionCount = 2;

    vertexInput.pVertexAttributeDescriptions =
        attributes;


    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};

    inputAssembly.sType =
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;

    inputAssembly.topology =
        VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;


    VkPipelineViewportStateCreateInfo viewportState{};

    viewportState.sType =
        VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;

    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;


    VkPipelineRasterizationStateCreateInfo rasterizer{};

    rasterizer.sType =
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;

    rasterizer.polygonMode =
        VK_POLYGON_MODE_FILL;

    rasterizer.cullMode =
        VK_CULL_MODE_NONE;

    rasterizer.frontFace =
        VK_FRONT_FACE_COUNTER_CLOCKWISE;

    rasterizer.lineWidth = 1.0f;


    VkPipelineMultisampleStateCreateInfo multisampling{};

    multisampling.sType =
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;

    multisampling.rasterizationSamples =
        VK_SAMPLE_COUNT_1_BIT;


    VkPipelineDepthStencilStateCreateInfo depthStencil{};

    depthStencil.sType =
        VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;

    depthStencil.depthTestEnable =
        VK_TRUE;

    depthStencil.depthWriteEnable =
        VK_TRUE;

    depthStencil.depthCompareOp =
        VK_COMPARE_OP_LESS;


    VkPipelineColorBlendAttachmentState colorBlendAttachment{};

    colorBlendAttachment.blendEnable =
        VK_FALSE;

    colorBlendAttachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT |
        VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT |
        VK_COLOR_COMPONENT_A_BIT;


    VkPipelineColorBlendStateCreateInfo colorBlending{};

    colorBlending.sType =
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;

    colorBlending.attachmentCount = 1;

    colorBlending.pAttachments =
        &colorBlendAttachment;


    VkDynamicState dynamicStates[] =
    {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR
    };


    VkPipelineDynamicStateCreateInfo dynamicState{};

    dynamicState.sType =
        VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;

    dynamicState.dynamicStateCount = 2;

    dynamicState.pDynamicStates =
        dynamicStates;


    VkGraphicsPipelineCreateInfo pipelineInfo{};

    pipelineInfo.sType =
        VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;

    pipelineInfo.stageCount = 2;

    pipelineInfo.pStages =
        shaderStages;

    pipelineInfo.pVertexInputState =
        &vertexInput;

    pipelineInfo.pInputAssemblyState =
        &inputAssembly;

    pipelineInfo.pViewportState =
        &viewportState;

    pipelineInfo.pRasterizationState =
        &rasterizer;

    pipelineInfo.pMultisampleState =
        &multisampling;

    pipelineInfo.pDepthStencilState =
        &depthStencil;

    pipelineInfo.pColorBlendState =
        &colorBlending;

    pipelineInfo.pDynamicState =
        &dynamicState;

    pipelineInfo.layout =
        pipelineLayout;

    pipelineInfo.renderPass =
        graphics::internal::context.render_pass;

    pipelineInfo.subpass = 0;


    if (
        vkCreateGraphicsPipelines(
            graphics::internal::context.device,
            VK_NULL_HANDLE,
            1,
            &pipelineInfo,
            nullptr,
            &pipeline)
        != VK_SUCCESS)
    {
        std::cerr
            << "Failed to create graphics pipeline\n";

        return false;
    }


    return true;
}

} // namespace


// ============================================================
// APPLICATION
// ============================================================

bool initialize()
{
    if (!createSphere())
    {
        return false;
    }

    if (!createPipeline())
    {
        return false;
    }

    return true;
}


void shutdown()
{
    auto& context =
        graphics::internal::context;

    vkQueueWaitIdle(
        context.graphics_queue);


    if (pipeline != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(
            context.device,
            pipeline,
            nullptr);
    }


    if (pipelineLayout != VK_NULL_HANDLE)
    {
        vkDestroyPipelineLayout(
            context.device,
            pipelineLayout,
            nullptr);
    }


    if (vertexShader != VK_NULL_HANDLE)
    {
        vkDestroyShaderModule(
            context.device,
            vertexShader,
            nullptr);
    }


    if (fragmentShader != VK_NULL_HANDLE)
    {
        vkDestroyShaderModule(
            context.device,
            fragmentShader,
            nullptr);
    }


    if (vertexBuffer != VK_NULL_HANDLE)
    {
        vmaDestroyBuffer(
            context.allocator,
            vertexBuffer,
            vertexAllocation);
    }


    if (indexBuffer != VK_NULL_HANDLE)
    {
        vmaDestroyBuffer(
            context.allocator,
            indexBuffer,
            indexAllocation);
    }
}


// ============================================================
// ImGui
// ============================================================

void update(double time)
{
    (void)time;


    ImGuiIO& io = ImGui::GetIO();

    if (!io.WantCaptureMouse)
    {
        if (ImGui::IsMouseClicked(ImGuiMouseButton_Left))
        {
            mouseDragging = true;

            lastMouseX = io.MousePos.x;
            lastMouseY = io.MousePos.y;
        }

        if (ImGui::IsMouseReleased(ImGuiMouseButton_Left))
        {
            mouseDragging = false;
        }

        if (mouseDragging)
        {
            const double mouseX = io.MousePos.x;
            const double mouseY = io.MousePos.y;

            const float deltaX =
                static_cast<float>(mouseX - lastMouseX);

            const float deltaY =
                static_cast<float>(mouseY - lastMouseY);

            angleY += deltaX * 0.01f;
            angleX += deltaY * 0.01f;

            lastMouseX = mouseX;
            lastMouseY = mouseY;
        }
    }

    ImGui::Begin(
        "Lab 1 - Variant 5");


    ImGui::Text(
        "Sphere (~100 vertices)");

    ImGui::Text(
        "Vertices: 110");

    ImGui::Separator();


    
    // Projection
    

    ImGui::Text(
        "Projection");

    ImGui::Checkbox(
        "Perspective",
        &perspectiveProjection);


    
    // Position
    

    ImGui::Separator();

    ImGui::Text(
        "Position");

    ImGui::SliderFloat(
        "X##position",
        &positionX,
        -2.0f,
        2.0f);

    ImGui::SliderFloat(
        "Y##position",
        &positionY,
        -2.0f,
        2.0f);

    ImGui::SliderFloat(
        "Z##position",
        &positionZ,
        -2.0f,
        2.0f);


    
    // Rotation
    

    ImGui::Separator();

    ImGui::Text(
        "Rotation");

    ImGui::SliderAngle(
        "X##rotation",
        &angleX,
        -180.0f,
        180.0f);

    ImGui::SliderAngle(
        "Y##rotation",
        &angleY,
        -180.0f,
        180.0f);

    ImGui::SliderAngle(
        "Z##rotation",
        &angleZ,
        -180.0f,
        180.0f);


    
    // Scale
    

    ImGui::Separator();

    ImGui::SliderFloat(
        "Scale",
        &objectScale,
        0.1f,
        3.0f);


    
    // Color
    

    ImGui::Separator();

    ImGui::ColorEdit4(
        "Color",
        objectColor);


    ImGui::End();
}


// ============================================================
// Rendering
// ============================================================

void render(
    const graphics::internal::FrameData& fd)
{
    auto& context =
        graphics::internal::context;


    VkCommandBuffer commandBuffer =
        fd.command_buffer;


    VkCommandBufferBeginInfo beginInfo{};

    beginInfo.sType =
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    beginInfo.flags =
        VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;


    vkBeginCommandBuffer(
        commandBuffer,
        &beginInfo);


    
    // MVP matrix
    

    const float aspect =
        static_cast<float>(
            context.swapchain_extent.width)
        /
        static_cast<float>(
            context.swapchain_extent.height);


    Mat4 model =
        translation(
            positionX,
            positionY,
            positionZ);


    model =
        multiply(
            model,
            rotationZ(angleZ));


    model =
        multiply(
            model,
            rotationY(angleY));


    model =
        multiply(
            model,
            rotationX(angleX));


    model =
        multiply(
            model,
            scaling(objectScale));


    /*
        Камера находится в точке (0, 0, -3).

        Поэтому объект находится перед камерой
        около координаты Z = 0.
    */

    Mat4 view =
        translation(
            0.0f,
            0.0f,
            -3.0f);


    Mat4 projection;


    if (perspectiveProjection)
    {
        projection =
            perspective(
                60.0f *
                    3.14159265359f /
                    180.0f,
                aspect,
                0.1f,
                100.0f);
    }
    else
    {
        projection =
            orthographic(
                4.0f * aspect,
                4.0f,
                0.1f,
                100.0f);
    }


    Mat4 mvp =
        multiply(
            projection,
            multiply(view, model));


    PushConstants pushConstants{};


    std::memcpy(
        pushConstants.mvp,
        mvp.data,
        sizeof(mvp.data));


    std::memcpy(
        pushConstants.color,
        objectColor,
        sizeof(objectColor));


    
    // Render pass
    

    VkClearValue clearValues[2]{};


    clearValues[0].color =
    {
        {
            0.03f,
            0.04f,
            0.06f,
            1.0f
        }
    };


    clearValues[1].depthStencil =
    {
        1.0f,
        0
    };


    VkRenderPassBeginInfo renderPassBegin{};

    renderPassBegin.sType =
        VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;

    renderPassBegin.renderPass =
        context.render_pass;

    renderPassBegin.framebuffer =
        fd.framebuffer;

    renderPassBegin.renderArea.offset =
    {
        0,
        0
    };

    renderPassBegin.renderArea.extent =
        context.swapchain_extent;

    renderPassBegin.clearValueCount =
        2;

    renderPassBegin.pClearValues =
        clearValues;


    vkCmdBeginRenderPass(
        commandBuffer,
        &renderPassBegin,
        VK_SUBPASS_CONTENTS_INLINE);


    
    // Viewport
    

    VkViewport viewport{};

    viewport.x = 0.0f;
    viewport.y = 0.0f;

    viewport.width =
        static_cast<float>(
            context.swapchain_extent.width);

    viewport.height =
        static_cast<float>(
            context.swapchain_extent.height);

    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;


    VkRect2D scissor{};

    scissor.offset =
    {
        0,
        0
    };

    scissor.extent =
        context.swapchain_extent;


    vkCmdSetViewport(
        commandBuffer,
        0,
        1,
        &viewport);


    vkCmdSetScissor(
        commandBuffer,
        0,
        1,
        &scissor);


    
    // Pipeline
    

    vkCmdBindPipeline(
        commandBuffer,
        VK_PIPELINE_BIND_POINT_GRAPHICS,
        pipeline);


    
    // Vertex buffer
    

    VkDeviceSize offset = 0;


    vkCmdBindVertexBuffers(
        commandBuffer,
        0,
        1,
        &vertexBuffer,
        &offset);


    
    // Index buffer
    

    vkCmdBindIndexBuffer(
        commandBuffer,
        indexBuffer,
        0,
        VK_INDEX_TYPE_UINT32);


    
    // Push constants
    

    vkCmdPushConstants(
        commandBuffer,
        pipelineLayout,
        VK_SHADER_STAGE_VERTEX_BIT |
        VK_SHADER_STAGE_FRAGMENT_BIT,
        0,
        sizeof(PushConstants),
        &pushConstants);


    
    // Draw
    

    vkCmdDrawIndexed(
        commandBuffer,
        indexCount,
        1,
        0,
        0,
        0);


    vkCmdEndRenderPass(
        commandBuffer);


    vkEndCommandBuffer(
        commandBuffer);
}

} // namespace application