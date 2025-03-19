package com.huskerdev.gpkt.apis.cuda

import kotlin.math.min

private const val CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2

expect class CUdevice
expect class CUcontext
expect class CUdeviceptr
expect class CUmodule
expect class nvrtcProgram
expect class CUfunction

internal expect fun isCUDASupported(): Boolean

internal expect fun cuInit(flags: Int): Int
internal expect fun cuGetErrorName(code: Int): String
internal expect fun cuGetErrorString(code: Int): String
internal expect fun cuDeviceGetCount(): Int
internal expect fun cuDeviceGet(index: Int): CUdevice
internal expect fun cuDeviceGetName(device: CUdevice): String
internal expect fun cuDeviceGetMemory(device: CUdevice): ULong
internal expect fun cuCtxCreate(flags: Int, device: CUdevice): CUcontext
internal expect fun cuCtxSetCurrent(context: CUcontext?): Int
internal expect fun cuCtxDestroy(context: CUcontext): Int
internal expect fun cuMemFree(ptr: CUdeviceptr): Int
internal expect fun cuMemAlloc(size: Long): CUdeviceptr
internal expect fun cuMemcpyDtoHFloats(dst: FloatArray, src: CUdeviceptr, byteCount: Long, srcOffset: Long): Int
internal expect fun cuMemcpyDtoHInts(dst: IntArray, src: CUdeviceptr, byteCount: Long, srcOffset: Long): Int
internal expect fun cuMemcpyDtoHBytes(dst: ByteArray, src: CUdeviceptr, byteCount: Long, srcOffset: Long): Int
internal expect fun cuMemcpyHtoDFloats(dst: CUdeviceptr, src: FloatArray, byteCount: Long, srcOffset: Long, dstOffset: Long): Int
internal expect fun cuMemcpyHtoDInts(dst: CUdeviceptr, src: IntArray, byteCount: Long, srcOffset: Long, dstOffset: Long): Int
internal expect fun cuMemcpyHtoDBytes(dst: CUdeviceptr, src: ByteArray, byteCount: Long, srcOffset: Long, dstOffset: Long): Int
internal expect fun nvrtcCreateProgram(src: String): nvrtcProgram
internal expect fun nvrtcCompileProgram(program: nvrtcProgram): Int
internal expect fun nvrtcGetProgramLog(program: nvrtcProgram): String
internal expect fun nvrtcGetPTX(program: nvrtcProgram): String
internal expect fun nvrtcDestroyProgram(program: nvrtcProgram): Int
internal expect fun cuModuleLoadData(ptx: String): CUmodule
internal expect fun cuModuleUnload(module: CUmodule): Int
internal expect fun cuModuleGetFunction(module: CUmodule, name: String): CUfunction
internal expect fun cuDeviceGetAttribute(attrib: Int, device: CUdevice): Int
internal expect fun cuLaunchKernel(
    function: CUfunction,
    gridDimX: Int, gridDimY: Int, gridDimZ: Int,
    blockDimX: Int, blockDimY: Int, blockDimZ: Int,
    sharedMemBytes: Int, vararg kernelParams: Any): Int

internal fun createString(bytes: ByteArray): String {
    val sb = StringBuilder()
    for (i in bytes.indices) {
        val c = Char(bytes[i].toUShort())
        if (c.code == 0)
            break
        sb.append(c)
    }
    return sb.toString()
}

private fun Int.checkError() {
    if(this != 0)
        throw Exception("${cuGetErrorName(this)}: ${cuGetErrorString(this)}")
}

object Cuda {
    val supported = isCUDASupported()

    init {
        try {
            cuInit(0).checkError()
        }catch (_: Throwable) { }
    }

    fun getDevices() =
        Array(cuDeviceGetCount(), ::cuDeviceGet)

    fun getDeviceName(device: CUdevice) =
        cuDeviceGetName(device)

    fun getDeviceMemory(device: CUdevice) =
        cuDeviceGetMemory(device)

    fun createContext(device: CUdevice) =
        cuCtxCreate(0, device)

    fun dispose(context: CUcontext) {
        cuCtxSetCurrent(context)
        cuCtxDestroy(context)
    }

    fun dealloc(context: CUcontext, ptr: CUdeviceptr) {
        cuCtxSetCurrent(context)
        cuMemFree(ptr).checkError()
    }

    fun alloc(context: CUcontext, size: Int): CUdeviceptr {
        cuCtxSetCurrent(context)
        return cuMemAlloc(size.toLong())
    }

    fun wrapFloats(context: CUcontext, array: FloatArray): CUdeviceptr {
        cuCtxSetCurrent(context)
        val ptr = cuMemAlloc(array.size.toLong() * Float.SIZE_BYTES)
        writeFloats(context, ptr, array, array.size, 0, 0)
        return ptr
    }

    fun wrapInts(context: CUcontext, array: IntArray): CUdeviceptr {
        cuCtxSetCurrent(context)
        val ptr = cuMemAlloc(array.size.toLong() * Int.SIZE_BYTES)
        writeInts(context, ptr, array, array.size, 0, 0)
        return ptr
    }

    fun wrapBytes(context: CUcontext, array: ByteArray): CUdeviceptr {
        cuCtxSetCurrent(context)
        val ptr = cuMemAlloc(array.size.toLong())
        writeBytes(context, ptr, array, array.size, 0, 0)
        return ptr
    }

    fun readFloats(context: CUcontext, src: CUdeviceptr, length: Int, offset: Int) = FloatArray(length).apply {
        cuCtxSetCurrent(context)
        cuMemcpyDtoHFloats(this, src, length.toLong() * Float.SIZE_BYTES, offset.toLong() * Float.SIZE_BYTES)
            .checkError()
    }

    fun readInts(context: CUcontext, src: CUdeviceptr, length: Int, offset: Int) = IntArray(length).apply {
        cuCtxSetCurrent(context)
        cuMemcpyDtoHInts(this, src, length.toLong() * Int.SIZE_BYTES, offset.toLong() * Int.SIZE_BYTES)
            .checkError()
    }

    fun readBytes(context: CUcontext, src: CUdeviceptr, length: Int, offset: Int) = ByteArray(length).apply {
        cuCtxSetCurrent(context)
        cuMemcpyDtoHBytes(this, src, length.toLong(), offset.toLong())
            .checkError()
    }

    fun writeFloats(context: CUcontext, dst: CUdeviceptr, src: FloatArray, length: Int, srcOffset: Int, dstOffset: Int) {
        cuCtxSetCurrent(context)
        cuMemcpyHtoDFloats(dst, src, length.toLong() * Float.SIZE_BYTES, srcOffset.toLong() * Float.SIZE_BYTES, dstOffset.toLong() * Float.SIZE_BYTES)
            .checkError()
    }

    fun writeInts(context: CUcontext, dst: CUdeviceptr, src: IntArray, length: Int, srcOffset: Int, dstOffset: Int) {
        cuCtxSetCurrent(context)
        cuMemcpyHtoDInts(dst, src, length.toLong() * Int.SIZE_BYTES, srcOffset.toLong() * Int.SIZE_BYTES, dstOffset.toLong() * Int.SIZE_BYTES)
            .checkError()
    }

    fun writeBytes(context: CUcontext, dst: CUdeviceptr, src: ByteArray, length: Int, srcOffset: Int, dstOffset: Int) {
        cuCtxSetCurrent(context)
        cuMemcpyHtoDBytes(dst, src, length.toLong(), srcOffset.toLong(), dstOffset.toLong())
            .checkError()
    }

    fun compileToModule(context: CUcontext, src: String): CUmodule{
        cuCtxSetCurrent(context)
        val program = nvrtcCreateProgram(src)

        if(nvrtcCompileProgram(program) != 0)
            throw Exception("Failed to compile CUDA program:\n${nvrtcGetProgramLog(program)}")

        val ptx = nvrtcGetPTX(program)
        nvrtcDestroyProgram(program).checkError()

        return cuModuleLoadData(ptx)
    }

    fun unloadModule(module: CUmodule) =
        cuModuleUnload(module).checkError()

    fun getFunctionPointer(context: CUcontext, module: CUmodule, name: String): CUfunction{
        cuCtxSetCurrent(context)
        return cuModuleGetFunction(module, name)
    }

    fun launch(device: CUdevice, context: CUcontext, function: CUfunction, count: Int, vararg arguments: Any){
        cuCtxSetCurrent(context)

        val maxBlockDimX = cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, device)

        val blockSizeX = min(maxBlockDimX, count)
        val gridSizeX = (count + blockSizeX - 1) / blockSizeX

        cuLaunchKernel(function,
            gridSizeX, 1, 1,
            blockSizeX, 1, 1,
            0, *arguments
        ).checkError()
    }
}