#version 450

layout(push_constant) uniform PushConstants {
    mat4 mvp;
    vec4 color;
} pc;

layout(location = 0) in vec3 fragNormal;

layout(location = 0) out vec4 outColor;

void main()
{
    vec3 normal = normalize(fragNormal);

    vec3 lightDirection =
        normalize(vec3(0.4, 0.7, 1.0));

    float diffuse =
        max(dot(normal, lightDirection), 0.0);

    float lighting =
        0.2 + 0.8 * diffuse;

    outColor =
        vec4(pc.color.rgb * lighting, pc.color.a);
}