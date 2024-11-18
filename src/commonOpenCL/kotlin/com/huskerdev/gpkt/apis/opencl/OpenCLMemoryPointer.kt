package com.huskerdev.gpkt.apis.opencl

import com.huskerdev.gpkt.*

private typealias CLReader<T> = (commandQueue: CLCommandQueue, mem: CLMem, length: Int, offset: Int) -> T
private typealias CLWriter<T> = (commandQueue: CLCommandQueue, mem: CLMem, src: T, length: Int, srcOffset: Int, dstOffset: Int) -> Unit

abstract class OpenCLMemoryPointer<T>: MemoryPointer<T> {
    abstract val opencl: OpenCL
    abstract val commandQueue: CLCommandQueue

    abstract val mem: CLMem
    override var released = false
        get() = field || context.released

    override fun release() {
        if(released) return
        released = true
        context.releaseMemory(this)
    }

    abstract class Sync<T>(
        private val reader: CLReader<T>,
        private val writer: CLWriter<T>,
    ): OpenCLMemoryPointer<T>(), SyncMemoryPointer<T>{
        override fun read(length: Int, offset: Int): T {
            assertNotReleased()
            return reader(commandQueue, mem, length, offset)
        }
        override fun write(src: T, length: Int, srcOffset: Int, dstOffset: Int) {
            assertNotReleased()
            writer(commandQueue, mem, src, length, srcOffset, dstOffset)
        }
    }

    abstract class Async<T>(
        private val reader: CLReader<T>,
        private val writer: CLWriter<T>,
    ): OpenCLMemoryPointer<T>(), AsyncMemoryPointer<T>{
        override suspend fun read(length: Int, offset: Int): T {
            assertNotReleased()
            return reader(commandQueue, mem, length, offset)
        }
        override suspend fun write(src: T, length: Int, srcOffset: Int, dstOffset: Int) {
            assertNotReleased()
            writer(commandQueue, mem, src, length, srcOffset, dstOffset)
        }
    }
}

// ===================
//        Sync
// ===================

class CLSyncFloatMemoryPointer(
    override val context: OpenCLSyncContext,
    override val length: Int,
    override val usage: MemoryUsage,
    override val mem: CLMem,
    override val opencl: OpenCL = context.opencl,
    override val commandQueue: CLCommandQueue = context.commandQueue
): OpenCLMemoryPointer.Sync<FloatArray>(
    opencl::readFloats,
    opencl::writeFloats
), SyncFloatMemoryPointer

class CLSyncIntMemoryPointer(
    override val context: OpenCLSyncContext,
    override val length: Int,
    override val usage: MemoryUsage,
    override val mem: CLMem,
    override val opencl: OpenCL = context.opencl,
    override val commandQueue: CLCommandQueue = context.commandQueue
): OpenCLMemoryPointer.Sync<IntArray>(
    opencl::readInts,
    opencl::writeInts
), SyncIntMemoryPointer

class CLSyncByteMemoryPointer(
    override val context: OpenCLSyncContext,
    override val length: Int,
    override val usage: MemoryUsage,
    override val mem: CLMem,
    override val opencl: OpenCL = context.opencl,
    override val commandQueue: CLCommandQueue = context.commandQueue
): OpenCLMemoryPointer.Sync<ByteArray>(
    opencl::readBytes,
    opencl::writeBytes
), SyncByteMemoryPointer

// ===================
//       Async
// ===================

class CLAsyncFloatMemoryPointer(
    override val context: OpenCLAsyncContext,
    override val length: Int,
    override val usage: MemoryUsage,
    override val mem: CLMem,
    override val opencl: OpenCL = context.opencl,
    override val commandQueue: CLCommandQueue = context.commandQueue
): OpenCLMemoryPointer.Async<FloatArray>(
    opencl::readFloats,
    opencl::writeFloats
), AsyncFloatMemoryPointer

class CLAsyncIntMemoryPointer(
    override val context: OpenCLAsyncContext,
    override val length: Int,
    override val usage: MemoryUsage,
    override val mem: CLMem,
    override val opencl: OpenCL = context.opencl,
    override val commandQueue: CLCommandQueue = context.commandQueue
): OpenCLMemoryPointer.Async<IntArray>(
    opencl::readInts,
    opencl::writeInts
), AsyncIntMemoryPointer

class CLAsyncByteMemoryPointer(
    override val context: OpenCLAsyncContext,
    override val length: Int,
    override val usage: MemoryUsage,
    override val mem: CLMem,
    override val opencl: OpenCL = context.opencl,
    override val commandQueue: CLCommandQueue = context.commandQueue
): OpenCLMemoryPointer.Async<ByteArray>(
    opencl::readBytes,
    opencl::writeBytes
), AsyncByteMemoryPointer