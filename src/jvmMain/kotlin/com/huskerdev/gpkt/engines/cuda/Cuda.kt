package com.huskerdev.gpkt.engines.cuda

import jcuda.CudaException
import jcuda.Pointer
import jcuda.driver.*
import jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X
import jcuda.driver.JCudaDriver.*
import jcuda.nvrtc.JNvrtc
import jcuda.nvrtc.JNvrtc.*
import jcuda.nvrtc.nvrtcProgram
import kotlin.math.max
import kotlin.math.min


class Cuda(
    requestedDeviceId: Int
) {
    companion object {
        val supported = try {
            JNvrtc.setExceptionsEnabled(true)

            cuInit(0)
            val buffer = IntArray(1)
            cuDeviceGetCount(buffer)
            if(buffer[0] == 0){
                println("[INFO] CUDA is supported, but can't find supported devices.")
                false
            }else true
        }catch (e: UnsatisfiedLinkError){
            println("[INFO] Failed to load CUDA. Check toolkit installation.")
            false
        }
    }

    val deviceId: Int
    private val device = CUdevice()
    private val context = CUcontext()

    val deviceName: String
    private val maxBlockDimX: Int

    private fun createString(bytes: ByteArray): String {
        val sb = StringBuilder()
        for (i in bytes.indices) {
            val c = Char(bytes[i].toUShort())
            if (c.code == 0)
                break
            sb.append(c)
        }
        return sb.toString()
    }

    init {
        val buffer = IntArray(1)
        cuDeviceGetCount(buffer)

        deviceId = max(0, min(requestedDeviceId, buffer[0] - 1))
        cuDeviceGet(device, deviceId)
        cuCtxCreate(context, 0, device)

        val nameBuffer = ByteArray(1024)
        cuDeviceGetName(nameBuffer, nameBuffer.size, device)
        deviceName = createString(nameBuffer)

        cuDeviceGetAttribute(buffer, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, device)
        maxBlockDimX = buffer[0]
    }

    fun dealloc(ptr: CUdeviceptr) {
        cuMemFree(ptr)
    }

    fun alloc(size: Int) = CUdeviceptr().apply {
        cuMemAlloc(this, size.toLong())
    }

    fun wrapFloats(array: FloatArray) = CUdeviceptr().apply {
        cuMemAlloc(this, array.size.toLong() * Float.SIZE_BYTES)
        writeFloats(this, array, array.size, 0, 0)
    }

    fun wrapInts(array: IntArray) = CUdeviceptr().apply {
        cuMemAlloc(this, array.size.toLong() * Int.SIZE_BYTES)
        writeInts(this, array, array.size, 0, 0)
    }

    fun wrapBytes(array: ByteArray) = CUdeviceptr().apply {
        cuMemAlloc(this, array.size.toLong())
        writeBytes(this, array, array.size, 0, 0)
    }

    fun readFloats(src: CUdeviceptr, length: Int, offset: Int) = FloatArray(length).apply {
        cuMemcpyDtoH(
            Pointer.to(this),
            src.withByteOffset(offset.toLong()),
            length.toLong() * Float.SIZE_BYTES)
    }

    fun readInts(src: CUdeviceptr, length: Int, offset: Int) = IntArray(length).apply {
        cuMemcpyDtoH(
            Pointer.to(this),
            src.withByteOffset(offset.toLong()),
            length.toLong() * Int.SIZE_BYTES)
    }

    fun readBytes(src: CUdeviceptr, length: Int, offset: Int) = ByteArray(length).apply {
        cuMemcpyDtoH(
            Pointer.to(this),
            src.withByteOffset(offset.toLong()),
            length.toLong())
    }

    fun writeFloats(dst: CUdeviceptr, src: FloatArray, length: Int, srcOffset: Int, dstOffset: Int) {
        cuMemcpyHtoD(
            dst.withByteOffset(dstOffset.toLong()),
            Pointer.to(src).withByteOffset(srcOffset.toLong()),
            length.toLong() * Float.SIZE_BYTES)
    }

    fun writeInts(dst: CUdeviceptr, src: IntArray, length: Int, srcOffset: Int, dstOffset: Int) {
        cuMemcpyHtoD(
            dst.withByteOffset(dstOffset.toLong()),
            Pointer.to(src).withByteOffset(srcOffset.toLong()),
            length.toLong() * Int.SIZE_BYTES)
    }

    fun writeBytes(dst: CUdeviceptr, src: ByteArray, length: Int, srcOffset: Int, dstOffset: Int) {
        cuMemcpyHtoD(
            dst.withByteOffset(dstOffset.toLong()),
            Pointer.to(src).withByteOffset(srcOffset.toLong()),
            length.toLong())
    }

    fun compileToModule(src: String): CUmodule{
        val program = nvrtcProgram()
        try {
            nvrtcCreateProgram(program, src, null, 0, null, null)
            nvrtcCompileProgram(program, 0, null)
        }catch (e: CudaException){
            val logBuffer = arrayOfNulls<String>(1)
            nvrtcGetProgramLog(program, logBuffer)
            throw Exception("Failed to compile CUDA program: \n${logBuffer[0]}")
        }

        val ptx = arrayOfNulls<String>(1)
        nvrtcGetPTX(program, ptx)
        nvrtcDestroyProgram(program)

        val module = CUmodule()
        cuModuleLoadData(module, ptx[0])
        return module
    }

    fun getFunctionPointer(module: CUmodule, name: String): CUfunction{
        val function = CUfunction()
        cuModuleGetFunction(function, module, name)
        return function
    }

    fun launch(function: CUfunction, count: Int, vararg pointers: Pointer){
        val blockSizeX = min(maxBlockDimX, count)
        val gridSizeX = (count + blockSizeX - 1) / blockSizeX

        cuLaunchKernel(function,
            gridSizeX, 1, 1,
            blockSizeX, 1, 1,
            0, null,
            Pointer.to(*pointers), null
        )
    }
}