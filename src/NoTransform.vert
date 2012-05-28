varying vec2 textureCoord;

void main() {
	gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
	textureCoord = gl_MultiTexCoord0.xy;
}