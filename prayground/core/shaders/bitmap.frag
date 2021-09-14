#version 330 core 

in vec2 vTexCoord;
out vec4 fragColor;

uniform sampler2D tex;
uniform bool is_gray;

void main()
{
    vec4 color = is_gray ? vec4(vec3(texture(tex, vTexCoord).r), 1.0) : texture(tex, vTexCoord);    
    fragColor = color;
}