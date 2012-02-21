uniform sampler2D myTexture;

varying vec2 textureCoord;

void main() {
	gl_FragColor = texture2D(myTexture, textureCoord);
}
