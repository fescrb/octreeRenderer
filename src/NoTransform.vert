attribute vec4 vertex;

varying vec2 textureCoord;

void main() {
	gl_Position = (vertex * vec4(2.0,2.0,1.0,1.0)) - vec4(1.0,1.0,0.0,0.0);
	textureCoord = vertex.xy;
}