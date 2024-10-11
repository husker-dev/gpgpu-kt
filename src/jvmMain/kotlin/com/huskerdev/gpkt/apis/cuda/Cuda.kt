package com.huskerdev.gpkt.apis.cuda

import jcuda.CudaException
import jcuda.Pointer
import jcuda.driver.*
import jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X
import jcuda.driver.JCudaDriver.*
import jcuda.nvrtc.JNvrtc.*
import jcuda.nvrtc.nvrtcProgram
import kotlin.math.min


class Cuda {
    companion object {
        val supported = try {
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

    fun getDevices(): Array<CUdevice> {
        val buffer = IntArray(1)
        cuDeviceGetCount(buffer)

        return Array(buffer[0]) { i ->
            val device = CUdevice()
            cuDeviceGet(device, i)
            device
        }
    }

    fun getDeviceName(device: CUdevice): String{
        val nameBuffer = ByteArray(1024)
        cuDeviceGetName(nameBuffer, nameBuffer.size, device)
        return createString(nameBuffer)
    }

    fun createContext(device: CUdevice): CUcontext{
        val context = CUcontext()
        cuCtxCreate(context, 0, device)
        return context
    }
/*
    val deviceId: Int
    private val device = CUdevice()
    private val context = CUcontext()

    val deviceName: String
    private val maxBlockDimX: Int

 */

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
/*
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

 */

    fun dispose(context: CUcontext) {
        cuCtxSetCurrent(context)
        cuCtxDestroy(context)
    }

    fun dealloc(context: CUcontext, ptr: CUdeviceptr) {
        cuCtxSetCurrent(context)
        cuMemFree(ptr)
    }

    fun alloc(context: CUcontext, size: Int) = CUdeviceptr().apply {
        cuCtxSetCurrent(context)
        cuMemAlloc(this, size.toLong())
    }

    fun wrapFloats(context: CUcontext, array: FloatArray) = CUdeviceptr().apply {
        cuCtxSetCurrent(context)
        cuMemAlloc(this, array.size.toLong() * Float.SIZE_BYTES)
        writeFloats(context, this, array, array.size, 0, 0)
    }

    fun wrapInts(context: CUcontext, array: IntArray) = CUdeviceptr().apply {
        cuCtxSetCurrent(context)
        cuMemAlloc(this, array.size.toLong() * Int.SIZE_BYTES)
        writeInts(context, this, array, array.size, 0, 0)
    }

    fun wrapBytes(context: CUcontext, array: ByteArray) = CUdeviceptr().apply {
        cuCtxSetCurrent(context)
        cuMemAlloc(this, array.size.toLong())
        writeBytes(context, this, array, array.size, 0, 0)
    }

    fun readFloats(context: CUcontext, src: CUdeviceptr, length: Int, offset: Int) = FloatArray(length).apply {
        cuCtxSetCurrent(context)
        cuMemcpyDtoH(
            Pointer.to(this),
            src.withByteOffset(offset.toLong()),
            length.toLong() * Float.SIZE_BYTES)
    }

    fun readInts(context: CUcontext, src: CUdeviceptr, length: Int, offset: Int) = IntArray(length).apply {
        cuCtxSetCurrent(context)
        cuMemcpyDtoH(
            Pointer.to(this),
            src.withByteOffset(offset.toLong()),
            length.toLong() * Int.SIZE_BYTES)
    }

    fun readBytes(context: CUcontext, src: CUdeviceptr, length: Int, offset: Int) = ByteArray(length).apply {
        cuCtxSetCurrent(context)
        cuMemcpyDtoH(
            Pointer.to(this),
            src.withByteOffset(offset.toLong()),
            length.toLong())
    }

    fun writeFloats(context: CUcontext, dst: CUdeviceptr, src: FloatArray, length: Int, srcOffset: Int, dstOffset: Int) {
        cuCtxSetCurrent(context)
        cuMemcpyHtoD(
            dst.withByteOffset(dstOffset.toLong()),
            Pointer.to(src).withByteOffset(srcOffset.toLong()),
            length.toLong() * Float.SIZE_BYTES)
    }

    fun writeInts(context: CUcontext, dst: CUdeviceptr, src: IntArray, length: Int, srcOffset: Int, dstOffset: Int) {
        cuCtxSetCurrent(context)
        cuMemcpyHtoD(
            dst.withByteOffset(dstOffset.toLong()),
            Pointer.to(src).withByteOffset(srcOffset.toLong()),
            length.toLong() * Int.SIZE_BYTES)
    }

    fun writeBytes(context: CUcontext, dst: CUdeviceptr, src: ByteArray, length: Int, srcOffset: Int, dstOffset: Int) {
        cuCtxSetCurrent(context)
        cuMemcpyHtoD(
            dst.withByteOffset(dstOffset.toLong()),
            Pointer.to(src).withByteOffset(srcOffset.toLong()),
            length.toLong())
    }

    fun compileToModule(context: CUcontext, src: String): CUmodule{
        cuCtxSetCurrent(context)
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

    fun getFunctionPointer(context: CUcontext, module: CUmodule, name: String): CUfunction{
        cuCtxSetCurrent(context)
        val function = CUfunction()
        cuModuleGetFunction(function, module, name)
        return function
    }

    fun launch(device: CUdevice, context: CUcontext, function: CUfunction, count: Int, vararg pointers: Pointer){
        cuCtxSetCurrent(context)

        val buffer = IntArray(1)
        cuDeviceGetAttribute(buffer, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, device)
        val maxBlockDimX = buffer[0]

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