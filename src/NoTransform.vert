attribute vec4 vertex;
attribute vec2 texCoord;

varying vec2 textureCoordinates;

void main() {
	gl_Position = vertex;
	textureCoordinates = texCoord;
}
