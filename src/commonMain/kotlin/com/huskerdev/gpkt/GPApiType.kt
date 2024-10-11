package com.huskerdev.gpkt

enum class GPApiType(
    val shortName: String
) {
    OpenCL("opencl"),
    CUDA("cuda"),
    Interpreter("interpreter"),
    Javac("javac"),
    WebGPU("webgpu"),
    JS("js"),
    Metal("metal")
    ;

    companion object {
        val mapped = entries.associateBy { it.shortName }

        val supported = supportedApis
    }
}