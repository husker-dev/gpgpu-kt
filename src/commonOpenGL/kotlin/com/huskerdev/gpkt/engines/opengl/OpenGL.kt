package com.huskerdev.gpkt.engines.opengl

import com.huskerdev.gpkt.MemoryUsage

internal expect fun createGL(): OpenGL?

interface OpenGL {

    val name: String

    fun deallocBuffer(ssbo: Int)
    fun deallocProgram(program: Int)

    fun alloc(size: Int, usage: MemoryUsage): Int

    fun createProgram(source: String): Int
    fun useProgram(program: Int)
    fun launchProgram(instances: Int)

    fun setBufferIndex(index: Int, ssbo: Int)
    fun setUniform1f(index: Int, value: Float)
    fun setUniform1i(index: Int, value: Int)
    fun setUniform1b(index: Int, value: Byte)

    fun wrapFloat(array: FloatArray, usage: MemoryUsage): Int
    fun wrapDouble(array: DoubleArray, usage: MemoryUsage): Int
    fun wrapInt(array: IntArray, usage: MemoryUsage): Int
    fun wrapByte(array: ByteArray, usage: MemoryUsage): Int

    fun readFloat(ssbo: Int, length: Int, offset: Int): FloatArray
    fun readInt(ssbo: Int, length: Int, offset: Int): IntArray
    fun readByte(ssbo: Int, length: Int, offset: Int): ByteArray

    fun writeFloat(ssbo: Int, src: FloatArray, length: Int, srcOffset: Int, dstOffset: Int)
    fun writeInt(ssbo: Int, src: IntArray, length: Int, srcOffset: Int, dstOffset: Int)
    fun writeByte(ssbo: Int, src: ByteArray, length: Int, srcOffset: Int, dstOffset: Int)
}