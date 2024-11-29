package com.huskerdev.gpkt.apis.cuda

import jcuda.Pointer
import jcuda.driver.JCudaDriver
import jcuda.driver.JCudaDriver.*
import jcuda.nvrtc.JNvrtc
import jcuda.nvrtc.JNvrtc.*


actual class CUdevice(val ptr: jcuda.driver.CUdevice)
actual class CUcontext(val ptr: jcuda.driver.CUcontext)
actual class CUdeviceptr(val ptr: jcuda.driver.CUdeviceptr)
actual class CUmodule(val ptr: jcuda.driver.CUmodule)
actual class nvrtcProgram(val ptr: jcuda.nvrtc.nvrtcProgram)
actual class CUfunction(val ptr: jcuda.driver.CUfunction)

internal actual fun isCUDASupported() = try {
    JNvrtc.setExceptionsEnabled(true)
    cuInit(0)
    val buffer = IntArray(1)
    cuDeviceGetCount(buffer)
    if(buffer[0] == 0){
        println("[INFO] CUDA is supported, but can not find supported devices.")
        false
    }else true
}catch (_: UnsatisfiedLinkError){
    println("[INFO] Failed to load CUDA. Check toolkit installation.")
    false
}catch (_: Throwable){
    false
}

internal actual fun cuInit(flags: Int) =
    JCudaDriver.cuInit(0)

internal actual fun cuDeviceGetCount(): Int {
    val buffer = IntArray(1)
    cuDeviceGetCount(buffer)
    return buffer[0]
}

internal actual fun cuDeviceGet(index: Int): CUdevice {
    val device = jcuda.driver.CUdevice()
    cuDeviceGet(device, index)
    return CUdevice(device)
}

internal actual fun cuDeviceGetName(device: CUdevice): String {
    val nameBuffer = ByteArray(1024)
    cuDeviceGetName(nameBuffer, nameBuffer.size, device.ptr)
    return createString(nameBuffer)
}

internal actual fun cuCtxCreate(flags: Int, device: CUdevice): CUcontext {
    val context = jcuda.driver.CUcontext()
    cuCtxCreate(context, 0, device.ptr)
    return CUcontext(context)
}

internal actual fun cuCtxSetCurrent(context: CUcontext?) =
    cuCtxSetCurrent(context?.ptr)

internal actual fun cuCtxDestroy(context: CUcontext) =
    cuCtxDestroy(context.ptr)

internal actual fun cuMemFree(ptr: CUdeviceptr) =
    cuMemFree(ptr.ptr)

internal actual fun cuMemAlloc(size: Long): CUdeviceptr {
    val ptr = jcuda.driver.CUdeviceptr()
    cuMemAlloc(ptr, size)
    return CUdeviceptr(ptr)
}

internal actual fun cuMemcpyDtoHFloats(
    dst: FloatArray,
    src: CUdeviceptr,
    byteCount: Long,
    srcOffset: Long
) = cuMemcpyDtoH(Pointer.to(dst), src.ptr.withByteOffset(srcOffset), byteCount)

internal actual fun cuMemcpyDtoHInts(
    dst: IntArray,
    src: CUdeviceptr,
    byteCount: Long,
    srcOffset: Long
) = cuMemcpyDtoH(Pointer.to(dst), src.ptr.withByteOffset(srcOffset), byteCount)

internal actual fun cuMemcpyDtoHBytes(
    dst: ByteArray,
    src: CUdeviceptr,
    byteCount: Long,
    srcOffset: Long
) = cuMemcpyDtoH(Pointer.to(dst), src.ptr.withByteOffset(srcOffset), byteCount)

internal actual fun cuMemcpyHtoDFloats(
    dst: CUdeviceptr,
    src: FloatArray,
    byteCount: Long,
    srcOffset: Long,
    dstOffset: Long
) = cuMemcpyHtoD(dst.ptr.withByteOffset(dstOffset), Pointer.to(src).withByteOffset(srcOffset), byteCount)


internal actual fun cuMemcpyHtoDInts(
    dst: CUdeviceptr,
    src: IntArray,
    byteCount: Long,
    srcOffset: Long,
    dstOffset: Long
) = cuMemcpyHtoD(dst.ptr.withByteOffset(dstOffset), Pointer.to(src).withByteOffset(srcOffset), byteCount)

internal actual fun cuMemcpyHtoDBytes(
    dst: CUdeviceptr,
    src: ByteArray,
    byteCount: Long,
    srcOffset: Long,
    dstOffset: Long
) = cuMemcpyHtoD(dst.ptr.withByteOffset(dstOffset), Pointer.to(src).withByteOffset(srcOffset), byteCount)

internal actual fun nvrtcCreateProgram(
    src: String
): nvrtcProgram {
    val program = jcuda.nvrtc.nvrtcProgram()
    nvrtcCreateProgram(program, src, null, 0, null, null)
    return nvrtcProgram(program)
}

internal actual fun nvrtcCompileProgram(
    program: nvrtcProgram
) = nvrtcCompileProgram(program.ptr, 0, null)

internal actual fun nvrtcGetProgramLog(program: nvrtcProgram): String {
    val logBuffer = arrayOfNulls<String>(1)
    nvrtcGetProgramLog(program.ptr, logBuffer)
    return logBuffer[0]!!
}

internal actual fun nvrtcGetPTX(program: nvrtcProgram): String {
    val ptx = arrayOfNulls<String>(1)
    nvrtcGetPTX(program.ptr, ptx)
    return ptx[0]!!
}

internal actual fun nvrtcDestroyProgram(program: nvrtcProgram) =
    nvrtcDestroyProgram(program.ptr)

internal actual fun cuModuleLoadData(ptx: String): CUmodule {
    val module = jcuda.driver.CUmodule()
    cuModuleLoadData(module, ptx)
    return CUmodule(module)
}

internal actual fun cuModuleUnload(module: CUmodule) =
    cuModuleUnload(module.ptr)

internal actual fun cuModuleGetFunction(
    module: CUmodule,
    name: String
): CUfunction {
    val function = jcuda.driver.CUfunction()
    cuModuleGetFunction(function, module.ptr, name)
    return CUfunction(function)
}

internal actual fun cuDeviceGetAttribute(attrib: Int, device: CUdevice): Int {
    val buffer = IntArray(1)
    cuDeviceGetAttribute(buffer, attrib, device.ptr)
    return buffer[0]
}

internal actual fun cuLaunchKernel(
    function: CUfunction,
    gridDimX: Int,
    gridDimY: Int,
    gridDimZ: Int,
    blockDimX: Int,
    blockDimY: Int,
    blockDimZ: Int,
    sharedMemBytes: Int,
    vararg kernelParams: Any
): Int {
    val args = kernelParams.map {
        when(it){
            is Float -> Pointer.to(floatArrayOf(it))
            is Int -> Pointer.to(intArrayOf(it))
            is Byte -> Pointer.to(byteArrayOf(it))
            is Boolean -> Pointer.to(byteArrayOf(if(it) 1 else 0))
            is CUdeviceptr -> Pointer.to(it.ptr)
            else -> throw UnsupportedOperationException(it.toString())
        }
    }.toTypedArray()
    return cuLaunchKernel(
        function.ptr,
        gridDimX, gridDimY, gridDimZ,
        blockDimX, blockDimY, blockDimZ,
        sharedMemBytes, null,
        Pointer.to(*args), null
    )
}

internal actual fun cuGetErrorName(code: Int): String {
    val res = Array(1) { "" }
    cuGetErrorName(code, res)
    return res[0]
}

internal actual fun cuGetErrorString(code: Int): String {
    val res = Array(1) { "" }
    cuGetErrorString(code, res)
    return res[0]
}