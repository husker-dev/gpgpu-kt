package com.huskerdev.gpkt.engines.opencl

import com.huskerdev.gpkt.*
import com.huskerdev.gpkt.ast.ScopeStatement
import java.nio.ByteBuffer
import java.nio.DoubleBuffer
import java.nio.FloatBuffer
import java.nio.IntBuffer


class OpenCLSyncDevice(
    requestedDeviceId: Int
): GPSyncDevice(GPType.OpenCL) {
    private val cl = OpenCL(requestedDeviceId)

    override val id = cl.deviceId
    override val name = cl.deviceName
    override val isGPU = true

    override fun allocFloat(array: FloatArray, usage: MemoryUsage) =
        CLSyncFloatMemoryPointer(cl, array.size, usage, cl.allocate(FloatBuffer.wrap(array), usage))

    override fun allocFloat(length: Int, usage: MemoryUsage) =
        CLSyncFloatMemoryPointer(cl, length, usage, cl.allocate(Float.SIZE_BYTES.toLong() * length, usage))

    override fun allocDouble(array: DoubleArray, usage: MemoryUsage) =
        CLSyncDoubleMemoryPointer(cl, array.size, usage, cl.allocate(DoubleBuffer.wrap(array), usage))

    override fun allocDouble(length: Int, usage: MemoryUsage) =
        CLSyncDoubleMemoryPointer(cl, length, usage, cl.allocate(Double.SIZE_BYTES.toLong() * length, usage))

    override fun allocInt(array: IntArray, usage: MemoryUsage) =
        CLSyncIntMemoryPointer(cl, array.size, usage, cl.allocate(IntBuffer.wrap(array), usage))

    override fun allocInt(length: Int, usage: MemoryUsage) =
        CLSyncIntMemoryPointer(cl, length, usage, cl.allocate(Int.SIZE_BYTES.toLong() * length, usage))

    override fun allocByte(array: ByteArray, usage: MemoryUsage) =
        CLSyncByteMemoryPointer(cl, array.size, usage, cl.allocate(ByteBuffer.wrap(array), usage))

    override fun allocByte(length: Int, usage: MemoryUsage) =
        CLSyncByteMemoryPointer(cl, length, usage, cl.allocate(length.toLong(), usage))

    override fun compile(ast: ScopeStatement) =
        OpenCLProgram(cl, ast)
}

class OpenCLAsyncDevice(
    requestedDeviceId: Int
): GPAsyncDevice(GPType.OpenCL) {
    private val cl = OpenCL(requestedDeviceId)

    override val id = cl.deviceId
    override val name = cl.deviceName
    override val isGPU = true

    override fun allocFloat(array: FloatArray, usage: MemoryUsage) =
        CLAsyncFloatMemoryPointer(cl, array.size, usage, cl.allocate(FloatBuffer.wrap(array), usage))

    override fun allocFloat(length: Int, usage: MemoryUsage) =
        CLAsyncFloatMemoryPointer(cl, length, usage, cl.allocate(Float.SIZE_BYTES.toLong() * length, usage))

    override fun allocDouble(array: DoubleArray, usage: MemoryUsage) =
        CLAsyncDoubleMemoryPointer(cl, array.size, usage, cl.allocate(DoubleBuffer.wrap(array), usage))

    override fun allocDouble(length: Int, usage: MemoryUsage) =
        CLAsyncDoubleMemoryPointer(cl, length, usage, cl.allocate(Double.SIZE_BYTES.toLong() * length, usage))

    override fun allocInt(array: IntArray, usage: MemoryUsage) =
        CLAsyncIntMemoryPointer(cl, array.size, usage, cl.allocate(IntBuffer.wrap(array), usage))

    override fun allocInt(length: Int, usage: MemoryUsage) =
        CLAsyncIntMemoryPointer(cl, length, usage, cl.allocate(Int.SIZE_BYTES.toLong() * length, usage))

    override fun allocByte(array: ByteArray, usage: MemoryUsage) =
        CLAsyncByteMemoryPointer(cl, array.size, usage, cl.allocate(ByteBuffer.wrap(array), usage))

    override fun allocByte(length: Int, usage: MemoryUsage) =
        CLAsyncByteMemoryPointer(cl, length, usage, cl.allocate(length.toLong(), usage))

    override fun compile(ast: ScopeStatement) =
        OpenCLProgram(cl, ast)


}

