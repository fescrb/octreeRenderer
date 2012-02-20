uniform sampler2D texture;

varying vec2 textureCoord;

void main() {
	vec4 color = vec4(1, 0, 0, 1);
	gl_FragColor = color;
}
