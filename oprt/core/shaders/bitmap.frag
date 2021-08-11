#version 330 core 

in vec2 vTexCoord;
out vec4 fragColor;

uniform sampler2D tex;

void main()
{
    fragColor = texture(tex, vTexCoord);
//    fragColor = vec4(vTexCoord, 0.0, 1.0);
}