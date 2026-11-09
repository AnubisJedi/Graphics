#version 450

layout(set = 0, binding = 0) uniform SceneData {
    mat4 viewProjection;
    vec4 cameraPosition;
    vec4 ambient;
    vec4 directionalDirection;
    vec4 directionalColor;
    ivec4 lightCounts;
} scene;

struct PointLight {
    vec4 position;
    vec4 colorIntensity;
};

layout(set = 0, binding = 1) readonly buffer PointLights {
    PointLight lights[];
} pointLights;

struct SpotLight {
    vec4 position;
    vec4 direction;
    vec4 colorIntensity;
    vec4 angles;
};

layout(set = 0, binding = 2) readonly buffer SpotLights {
    SpotLight lights[];
} spotLights;

layout(set = 1, binding = 0) uniform ModelData {
    mat4 model;
    mat4 normalMatrix;
    vec4 albedo;
    vec4 specular;
} modelData;

layout(location = 0) in vec3 fragWorldPos;
layout(location = 1) in vec3 fragWorldNormal;

layout(location = 0) out vec4 outColor;

vec3 blinnPhong(vec3 normal, vec3 viewDir, vec3 lightDir, vec3 lightRgb, float attenuation)
{
    float ndotl = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = modelData.albedo.rgb * ndotl;

    vec3 specular = vec3(0.0);
    if (ndotl > 0.0) {
        vec3 halfDir = normalize(lightDir + viewDir);
        float ndoth = max(dot(normal, halfDir), 0.0);
        specular = modelData.specular.rgb * pow(ndoth, modelData.specular.a);
    }

    return (diffuse + specular) * lightRgb * attenuation;
}

void main()
{
    vec3 normal = normalize(fragWorldNormal);
    vec3 viewDir = normalize(scene.cameraPosition.xyz - fragWorldPos);

    vec3 color = modelData.albedo.rgb * scene.ambient.rgb * scene.ambient.a;

    vec3 travelDir = scene.directionalDirection.xyz;
    if (dot(travelDir, travelDir) > 0.0) {
        vec3 lightDir = normalize(-travelDir);
        vec3 lightRgb = scene.directionalColor.rgb * scene.directionalColor.a;
        color += blinnPhong(normal, viewDir, lightDir, lightRgb, 1.0);
    }

    for (int i = 0; i < scene.lightCounts.x; ++i) {
        vec3 toLight = pointLights.lights[i].position.xyz - fragWorldPos;
        float dist2 = max(dot(toLight, toLight), 0.0001);
        vec3 lightDir = toLight * inversesqrt(dist2);
        vec4 packed = pointLights.lights[i].colorIntensity;
        color += blinnPhong(normal, viewDir, lightDir, packed.rgb * packed.a, 1.0 / dist2);
    }

    for (int i = 0; i < scene.lightCounts.y; ++i) {
        vec3 toLight = spotLights.lights[i].position.xyz - fragWorldPos;
        float dist2 = max(dot(toLight, toLight), 0.0001);
        vec3 lightDir = toLight * inversesqrt(dist2);

        vec3 spotDir = normalize(spotLights.lights[i].direction.xyz);
        float theta = dot(-lightDir, spotDir);
        float innerCos = spotLights.lights[i].angles.x;
        float outerCos = spotLights.lights[i].angles.y;
        float epsilon = innerCos - outerCos;

        float spotFactor;
        if (epsilon > 0.0001) {
            spotFactor = clamp((theta - outerCos) / epsilon, 0.0, 1.0);
        } else {
            spotFactor = theta >= innerCos ? 1.0 : 0.0;
        }

        vec4 packed = spotLights.lights[i].colorIntensity;
        color += blinnPhong(normal, viewDir, lightDir, packed.rgb * packed.a, spotFactor / dist2);
    }

    outColor = vec4(color, modelData.albedo.a);
}
