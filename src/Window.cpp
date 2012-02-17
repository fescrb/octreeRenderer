#include "Window.h"

Window::Window(int argc, char** argv, float2 dimensions) {
	glutInit(&argc, argv);
	glutInitWindowSize(dimensions[0], dimensions[1]);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);

	glutCreateWindow("Octree Renderer");
}

void Window::render() {

}

void Window::resize(GLInt width, GLInt height) {

}
