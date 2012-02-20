attribute vec4 vertex;

varying vec2 textureCoord;

void main() {
	gl_Position = vertex;
	textureCoord = vertex.xy;
}