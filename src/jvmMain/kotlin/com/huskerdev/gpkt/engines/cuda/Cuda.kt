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

    fun alloc(pointer: Pointer, size: Long) = CUdeviceptr().apply {
        cuMemAlloc(this, size)
        cuMemcpyHtoD(this, pointer, size)
    }

    fun alloc(size: Long) = CUdeviceptr().apply {
        cuMemAlloc(this, size)
    }

    fun dealloc(ptr: CUdeviceptr) {
        cuMemFree(ptr)
    }

    fun read(src: CUdeviceptr, dst: Pointer, size: Long, dstOffset: Long, srcOffset: Long) {
        val shiftedDst = if(dstOffset == 0L) dst else dst.withByteOffset(dstOffset)
        val shiftedSrc = if(srcOffset == 0L) src else src.withByteOffset(srcOffset)
        cuMemcpyDtoH(shiftedDst, shiftedSrc, size)
    }

    fun write(dst: CUdeviceptr, src: Pointer, size: Long, srcOffset: Long, dstOffset: Long) {
        val shiftedDst = if(dstOffset == 0L) dst else dst.withByteOffset(dstOffset)
        val shiftedSrc = if(srcOffset == 0L) src else src.withByteOffset(srcOffset)
        cuMemcpyHtoD(shiftedDst, shiftedSrc, size)
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