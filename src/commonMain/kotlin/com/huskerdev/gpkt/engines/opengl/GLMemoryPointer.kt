package com.huskerdev.gpkt.engines.opengl

import com.huskerdev.gpkt.*

private typealias GLReader<T> = (ssbo: Int, length: Int, offset: Int) -> T
private typealias GLWriter<T> = (ssbo: Int, src: T, length: Int, srcOffset: Int, dstOffset: Int) -> Unit


abstract class GLMemoryPointer<T>: MemoryPointer<T> {
    abstract val openGL: OpenGL
    abstract val ssbo: Int

    override fun dealloc() =
        openGL.deallocBuffer(ssbo)

    abstract class Sync<T>(
        val reader: GLReader<T>,
        val writer: GLWriter<T>
    ): GLMemoryPointer<T>(), SyncMemoryPointer<T>{
        override fun write(src: T, length: Int, srcOffset: Int, dstOffset: Int) =
            writer(ssbo, src, length, srcOffset, dstOffset)
        override fun read(length: Int, offset: Int) =
            reader(ssbo, length, offset)
    }

    abstract class Async<T>(
        val reader: GLReader<T>,
        val writer: GLWriter<T>
    ): GLMemoryPointer<T>(), AsyncMemoryPointer<T>{
        override suspend fun write(src: T, length: Int, srcOffset: Int, dstOffset: Int) =
            writer(ssbo, src, length, srcOffset, dstOffset)
        override suspend fun read(length: Int, offset: Int) =
            reader(ssbo, length, offset)
    }
}


// ===================
//        Sync
// ===================

class GLSyncFloatMemoryPointer(
    override val openGL: OpenGL,
    override val usage: MemoryUsage,
    override val length: Int,
    override val ssbo: Int,
): GLMemoryPointer.Sync<FloatArray>(
    openGL::readFloat, openGL::writeFloat
), SyncFloatMemoryPointer

class GLSyncDoubleMemoryPointer(
    override val openGL: OpenGL,
    override val usage: MemoryUsage,
    override val length: Int,
    override val ssbo: Int,
): GLMemoryPointer.Sync<DoubleArray>(
    openGL::readDouble, openGL::writeDouble
), SyncDoubleMemoryPointer

class GLSyncIntMemoryPointer(
    override val openGL: OpenGL,
    override val usage: MemoryUsage,
    override val length: Int,
    override val ssbo: Int,
): GLMemoryPointer.Sync<IntArray>(
    openGL::readInt, openGL::writeInt
), SyncIntMemoryPointer

class GLSyncByteMemoryPointer(
    override val openGL: OpenGL,
    override val usage: MemoryUsage,
    override val length: Int,
    override val ssbo: Int,
): GLMemoryPointer.Sync<ByteArray>(
    openGL::readByte, openGL::writeByte
), SyncByteMemoryPointer


// ===================
//       Async
// ===================

class GLAsyncFloatMemoryPointer(
    override val openGL: OpenGL,
    override val usage: MemoryUsage,
    override val length: Int,
    override val ssbo: Int,
): GLMemoryPointer.Async<FloatArray>(
    openGL::readFloat, openGL::writeFloat
), AsyncFloatMemoryPointer

class GLAsyncDoubleMemoryPointer(
    override val openGL: OpenGL,
    override val usage: MemoryUsage,
    override val length: Int,
    override val ssbo: Int,
): GLMemoryPointer.Async<DoubleArray>(
    openGL::readDouble, openGL::writeDouble
), AsyncDoubleMemoryPointer

class GLAsyncIntMemoryPointer(
    override val openGL: OpenGL,
    override val usage: MemoryUsage,
    override val length: Int,
    override val ssbo: Int,
): GLMemoryPointer.Async<IntArray>(
    openGL::readInt, openGL::writeInt
), AsyncIntMemoryPointer

class GLAsyncByteMemoryPointer(
    override val openGL: OpenGL,
    override val usage: MemoryUsage,
    override val length: Int,
    override val ssbo: Int,
): GLMemoryPointer.Async<ByteArray>(
    openGL::readByte, openGL::writeByte
), AsyncByteMemoryPointer