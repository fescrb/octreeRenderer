uniform sampler2D texture;

varying vec2 textureCoordinates;

void main() {
	gl_FragColor = texture2D(texture, textureCoordinates);
}
