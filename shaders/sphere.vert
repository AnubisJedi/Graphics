#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;

layout(set = 0, binding = 0) uniform SceneData {
    mat4 viewProjection;
    vec4 cameraPosition;
    vec4 ambient;
    vec4 directionalDirection;
    vec4 directionalColor;
    ivec4 lightCounts;
} scene;

layout(set = 1, binding = 0) uniform ModelData {
    mat4 model;
    mat4 normalMatrix;
    vec4 albedo;
    vec4 specular;
} modelData;

layout(location = 0) out vec3 fragWorldPos;
layout(location = 1) out vec3 fragWorldNormal;

void main()
{
    vec4 worldPosition = modelData.model * vec4(inPosition, 1.0);

    fragWorldPos = worldPosition.xyz;
    fragWorldNormal = mat3(modelData.normalMatrix) * inNormal;

    gl_Position = scene.viewProjection * worldPosition;
}
