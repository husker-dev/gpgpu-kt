package com.huskerdev.gpkt.engines.opencl

import com.huskerdev.gpkt.*

private typealias CLReader<T> = (mem: CLMem, length: Int, offset: Int) -> T
private typealias CLWriter<T> = (mem: CLMem, src: T, length: Int, srcOffset: Int, dstOffset: Int) -> Unit

abstract class OpenCLMemoryPointer<T>: MemoryPointer<T> {
    abstract val cl: OpenCL
    abstract val mem: CLMem

    override fun dealloc() =
        cl.deallocMemory(mem)

    abstract class Sync<T>(
        private val reader: CLReader<T>,
        private val writer: CLWriter<T>,
    ): OpenCLMemoryPointer<T>(), SyncMemoryPointer<T>{
        override fun read(length: Int, offset: Int) =
            reader(mem, length, offset)
        override fun write(src: T, length: Int, srcOffset: Int, dstOffset: Int) =
            writer(mem, src, length, srcOffset, dstOffset)
    }

    abstract class Async<T>(
        private val reader: CLReader<T>,
        private val writer: CLWriter<T>,
    ): OpenCLMemoryPointer<T>(), AsyncMemoryPointer<T>{
        override suspend fun read(length: Int, offset: Int) =
            reader(mem, length, offset)
        override suspend fun write(src: T, length: Int, srcOffset: Int, dstOffset: Int) =
            writer(mem, src, length, srcOffset, dstOffset)
    }
}

// ===================
//        Sync
// ===================

class CLSyncFloatMemoryPointer(
    override val cl: OpenCL,
    override val length: Int,
    override val usage: MemoryUsage,
    override val mem: CLMem
): OpenCLMemoryPointer.Sync<FloatArray>(
    cl::readFloats, cl::writeFloats
), SyncFloatMemoryPointer

class CLSyncIntMemoryPointer(
    override val cl: OpenCL,
    override val length: Int,
    override val usage: MemoryUsage,
    override val mem: CLMem
): OpenCLMemoryPointer.Sync<IntArray>(
    cl::readInts, cl::writeInts
), SyncIntMemoryPointer

class CLSyncByteMemoryPointer(
    override val cl: OpenCL,
    override val length: Int,
    override val usage: MemoryUsage,
    override val mem: CLMem
): OpenCLMemoryPointer.Sync<ByteArray>(
    cl::readBytes, cl::writeBytes
), SyncByteMemoryPointer

// ===================
//       Async
// ===================

class CLAsyncFloatMemoryPointer(
    override val cl: OpenCL,
    override val length: Int,
    override val usage: MemoryUsage,
    override val mem: CLMem
): OpenCLMemoryPointer.Async<FloatArray>(
    cl::readFloats, cl::writeFloats
), AsyncFloatMemoryPointer

class CLAsyncIntMemoryPointer(
    override val cl: OpenCL,
    override val length: Int,
    override val usage: MemoryUsage,
    override val mem: CLMem
): OpenCLMemoryPointer.Async<IntArray>(
    cl::readInts, cl::writeInts
), AsyncIntMemoryPointer

class CLAsyncByteMemoryPointer(
    override val cl: OpenCL,
    override val length: Int,
    override val usage: MemoryUsage,
    override val mem: CLMem
): OpenCLMemoryPointer.Async<ByteArray>(
    cl::readBytes, cl::writeBytes
), AsyncByteMemoryPointer