package com.huskerdev.gpkt.engines.opengl

import com.huskerdev.gpkt.*
import com.huskerdev.gpkt.ast.ScopeStatement

abstract class OpenGLDeviceBase: GPDeviceBase {
    protected val openGL = createGL()!!

    override val type = GPType.OpenGL
    override val id = 0
    override val name = openGL.name
    override val isGPU = true
    override val modules = GPModules(this)

    override fun compile(ast: ScopeStatement) =
        GLProgram(openGL, ast)
}

class OpenGLSyncDevice: OpenGLDeviceBase(), GPSyncDevice {
    override fun wrapFloats(array: FloatArray, usage: MemoryUsage) =
        GLSyncFloatMemoryPointer(openGL, usage, array.size, openGL.wrapFloat(array, usage))

    override fun allocFloats(length: Int, usage: MemoryUsage) =
        GLSyncFloatMemoryPointer(openGL, usage, length, openGL.alloc(length * Float.SIZE_BYTES, usage))

    override fun wrapDoubles(array: DoubleArray, usage: MemoryUsage) =
        GLSyncDoubleMemoryPointer(openGL, usage, array.size, openGL.wrapDouble(array, usage))

    override fun allocDoubles(length: Int, usage: MemoryUsage) =
        GLSyncDoubleMemoryPointer(openGL, usage, length, openGL.alloc(length * Double.SIZE_BYTES, usage))

    override fun wrapInts(array: IntArray, usage: MemoryUsage) =
        GLSyncIntMemoryPointer(openGL, usage, array.size, openGL.wrapInt(array, usage))

    override fun allocInts(length: Int, usage: MemoryUsage) =
        GLSyncIntMemoryPointer(openGL, usage, length, openGL.alloc(length * Int.SIZE_BYTES, usage))

    override fun wrapBytes(array: ByteArray, usage: MemoryUsage) =
        GLSyncByteMemoryPointer(openGL, usage, array.size, openGL.wrapByte(array, usage))

    override fun allocBytes(length: Int, usage: MemoryUsage) =
        GLSyncByteMemoryPointer(openGL, usage, length, openGL.alloc(length, usage))
}

class OpenGLAsyncDevice: OpenGLDeviceBase(), GPAsyncDevice {
    override fun wrapFloats(array: FloatArray, usage: MemoryUsage) =
        GLAsyncFloatMemoryPointer(openGL, usage, array.size, openGL.wrapFloat(array, usage))

    override fun allocFloats(length: Int, usage: MemoryUsage) =
        GLAsyncFloatMemoryPointer(openGL, usage, length, openGL.alloc(length * Float.SIZE_BYTES, usage))

    override fun wrapDoubles(array: DoubleArray, usage: MemoryUsage) =
        GLAsyncDoubleMemoryPointer(openGL, usage, array.size, openGL.wrapDouble(array, usage))

    override fun allocDoubles(length: Int, usage: MemoryUsage) =
        GLAsyncDoubleMemoryPointer(openGL, usage, length, openGL.alloc(length * Double.SIZE_BYTES, usage))

    override fun wrapInts(array: IntArray, usage: MemoryUsage) =
        GLAsyncIntMemoryPointer(openGL, usage, array.size, openGL.wrapInt(array, usage))

    override fun allocInts(length: Int, usage: MemoryUsage) =
        GLAsyncIntMemoryPointer(openGL, usage, length, openGL.alloc(length * Int.SIZE_BYTES, usage))

    override fun wrapBytes(array: ByteArray, usage: MemoryUsage) =
        GLAsyncByteMemoryPointer(openGL, usage, array.size, openGL.wrapByte(array, usage))

    override fun allocBytes(length: Int, usage: MemoryUsage) =
        GLAsyncByteMemoryPointer(openGL, usage, length, openGL.alloc(length, usage))
}