package com.huskerdev.gpkt

enum class GPType(
    val shortName: String
) {
    OpenCL("opencl"),
    CUDA("cuda"),
    Interpreter("interpreter"),
    Javac("javac"),
    WebGPU("webgpu"),
    JS("js"),
    OpenGL("opengl")
    ;

    companion object {
        val mapped = entries.associateBy { it.shortName }
    }
}