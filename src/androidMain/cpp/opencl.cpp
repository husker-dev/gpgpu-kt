#include <jni.h>

#include "opencl_loader.h"
#include <CL/cl.hpp>



extern "C" JNIEXPORT jboolean JNICALL Java_com_huskerdev_gpkt_apis_opencl_OpenCLBindings_nTryInitCL(JNIEnv* env, jclass) {
    cl_int err = OpenCLHelper::Loader::Init();
    return (jboolean)!err;
}

extern "C" JNIEXPORT jlongArray JNICALL Java_com_huskerdev_gpkt_apis_opencl_OpenCLBindings_nGetPlatforms(JNIEnv* env, jclass) {
    cl_uint size;
    clGetPlatformIDs(0, nullptr, &size);

    auto* platforms = new cl_platform_id[(int)size];
    clGetPlatformIDs(size, platforms, nullptr);

    jlongArray res = env->NewLongArray((int)size);
    env->SetLongArrayRegion(res, 0, (int)size, (const jlong*) platforms);
    delete[] platforms;
    return res;
}

extern "C" JNIEXPORT jlongArray JNICALL Java_com_huskerdev_gpkt_apis_opencl_OpenCLBindings_nGetDevices(JNIEnv* env, jclass, jlong platform, jlong type) {
    cl_uint size;
    clGetDeviceIDs((cl_platform_id)platform, type, 0, nullptr, &size);

    auto* devices = new cl_device_id[(int)size];
    clGetDeviceIDs((cl_platform_id)platform, type, size, devices, nullptr);

    jlongArray res = env->NewLongArray((int)size);
    env->SetLongArrayRegion(res, 0, (int)size, (const jlong*) devices);
    delete[] devices;
    return res;
}

extern "C" JNIEXPORT jbyteArray JNICALL Java_com_huskerdev_gpkt_apis_opencl_OpenCLBindings_nGetDeviceInfo(JNIEnv* env, jclass, jlong device, jint param) {
    size_t size;
    clGetDeviceInfo((cl_device_id)device, (cl_device_info)param, 0, nullptr, &size);

    char* buffer = new char[(int)size];
    clGetDeviceInfo((cl_device_id)device, (cl_device_info)param, size, buffer, nullptr);

    jbyteArray res = env->NewByteArray((int)size);
    env->SetByteArrayRegion(res, 0, (int)size, (const jbyte*) buffer);
    delete[] buffer;
    return res;
}

extern "C" JNIEXPORT jlong JNICALL Java_com_huskerdev_gpkt_apis_opencl_OpenCLBindings_nCreateContext(JNIEnv* env, jclass, jlongArray _properties, jlong _device) {
    jboolean isCopy = false;
    auto device = (cl_device_id)_device;

    auto properties = env->GetLongArrayElements(_properties, &isCopy);
    auto* cl_properties = new cl_context_properties[env->GetArrayLength(_properties) + 1];
    for(int i = 0; i < env->GetArrayLength(_properties); i++)
        cl_properties[i] = (cl_context_properties)properties[i];
    cl_properties[2] = 0;

    auto result = (jlong) clCreateContext(cl_properties, 1, &device, nullptr, nullptr, nullptr);
    delete[] cl_properties;
    return result;
}

extern "C" JNIEXPORT void JNICALL Java_com_huskerdev_gpkt_apis_opencl_OpenCLBindings_nReleaseContext(JNIEnv* env, jclass, jlong _context) {
    clReleaseContext((cl_context)_context);
}

extern "C" JNIEXPORT jlong JNICALL Java_com_huskerdev_gpkt_apis_opencl_OpenCLBindings_nCreateCommandQueue(JNIEnv* env, jclass, jlong context, jlong device) {
    return (jlong)clCreateCommandQueue((cl_context)context, (cl_device_id)device, 0, nullptr);
}

extern "C" JNIEXPORT void JNICALL Java_com_huskerdev_gpkt_apis_opencl_OpenCLBindings_nReleaseMem(JNIEnv* env, jclass, jlong mem) {
    clReleaseMemObject((cl_mem)mem);
}

extern "C" JNIEXPORT void JNICALL Java_com_huskerdev_gpkt_apis_opencl_OpenCLBindings_nReleaseProgram(JNIEnv* env, jclass, jlong program) {
    clReleaseProgram((cl_program)program);
}

extern "C" JNIEXPORT void JNICALL Java_com_huskerdev_gpkt_apis_opencl_OpenCLBindings_nReleaseKernel(JNIEnv* env, jclass, jlong kernel) {
    clReleaseKernel((cl_kernel)kernel);
}

extern "C" JNIEXPORT jlong JNICALL Java_com_huskerdev_gpkt_apis_opencl_OpenCLBindings_nCreateBuffer(JNIEnv* env, jclass, jlong context, jlong flags, jlong size) {
    return (jlong) clCreateBuffer((cl_context)context, flags, size, nullptr, nullptr);
}

extern "C" JNIEXPORT jlong JNICALL Java_com_huskerdev_gpkt_apis_opencl_OpenCLBindings_nCreateFloatBuffer(JNIEnv* env, jclass, jlong context, jlong flags, jfloatArray array) {
    jboolean isCopy = false;
    auto arr = env->GetPrimitiveArrayCritical(array, &isCopy);
    auto result = (jlong) clCreateBuffer((cl_context)context, flags, 4 * env->GetArrayLength(array), arr, nullptr);
    env->ReleasePrimitiveArrayCritical(array, arr, JNI_ABORT);
    return result;
}

extern "C" JNIEXPORT jlong JNICALL Java_com_huskerdev_gpkt_apis_opencl_OpenCLBindings_nCreateIntBuffer(JNIEnv* env, jclass, jlong context, jlong flags, jintArray array) {
    jboolean isCopy = false;
    auto arr = env->GetPrimitiveArrayCritical(array, &isCopy);
    auto result = (jlong) clCreateBuffer((cl_context)context, flags, 4 * env->GetArrayLength(array), arr, nullptr);
    env->ReleasePrimitiveArrayCritical(array, arr, JNI_ABORT);
    return result;
}

extern "C" JNIEXPORT jlong JNICALL Java_com_huskerdev_gpkt_apis_opencl_OpenCLBindings_nCreateByteBuffer(JNIEnv* env, jclass, jlong context, jlong flags, jbyteArray array) {
    jboolean isCopy = false;
    auto arr = env->GetPrimitiveArrayCritical(array, &isCopy);
    auto result = (jlong) clCreateBuffer((cl_context)context, flags, env->GetArrayLength(array), arr, nullptr);
    env->ReleasePrimitiveArrayCritical(array, arr, JNI_ABORT);
    return result;
}

extern "C" JNIEXPORT void JNICALL Java_com_huskerdev_gpkt_apis_opencl_OpenCLBindings_nReadFloatBuffer(JNIEnv* env, jclass, jlong commandQueue, jlong mem, jboolean blockingRead, jlong offset, jlong size, jfloatArray dst) {
    auto arr = new char[size];
    clEnqueueReadBuffer(
        (cl_command_queue)commandQueue, (cl_mem)mem, blockingRead,
        offset, size, arr, 0, nullptr, nullptr
    );
    env->SetFloatArrayRegion(dst, 0, env->GetArrayLength(dst), (jfloat*)arr);
    delete[] arr;
}

extern "C" JNIEXPORT void JNICALL Java_com_huskerdev_gpkt_apis_opencl_OpenCLBindings_nReadIntBuffer(JNIEnv* env, jclass, jlong commandQueue, jlong mem, jboolean blockingRead, jlong offset, jlong size, jintArray dst) {
    auto arr = new char[size];
    clEnqueueReadBuffer(
            (cl_command_queue)commandQueue, (cl_mem)mem, blockingRead,
            offset, size, arr, 0, nullptr, nullptr
    );
    env->SetIntArrayRegion(dst, 0, env->GetArrayLength(dst), (jint*)arr);
    delete[] arr;
}

extern "C" JNIEXPORT void JNICALL Java_com_huskerdev_gpkt_apis_opencl_OpenCLBindings_nReadByteBuffer(JNIEnv* env, jclass, jlong commandQueue, jlong mem, jboolean blockingRead, jlong offset, jlong size, jbyteArray dst) {
    auto arr = new char[size];
    clEnqueueReadBuffer(
            (cl_command_queue)commandQueue, (cl_mem)mem, blockingRead,
            offset, size, arr, 0, nullptr, nullptr
    );
    env->SetByteArrayRegion(dst, 0, env->GetArrayLength(dst), (jbyte*)arr);
    delete[] arr;
}

extern "C" JNIEXPORT void JNICALL Java_com_huskerdev_gpkt_apis_opencl_OpenCLBindings_nWriteFloatBuffer(JNIEnv* env, jclass, jlong commandQueue, jlong mem, jboolean blockingRead, jlong offset, jlong size, jfloatArray src, jlong srcOffset) {
    jboolean isCopy = false;
    auto arr = (char*)env->GetPrimitiveArrayCritical(src, &isCopy);
    clEnqueueWriteBuffer(
            (cl_command_queue)commandQueue, (cl_mem)mem, blockingRead,
            offset, size, arr + srcOffset, 0, nullptr, nullptr
    );
    env->ReleasePrimitiveArrayCritical(src, arr, JNI_ABORT);
    delete[] arr;
}

extern "C" JNIEXPORT void JNICALL Java_com_huskerdev_gpkt_apis_opencl_OpenCLBindings_nWriteIntBuffer(JNIEnv* env, jclass, jlong commandQueue, jlong mem, jboolean blockingRead, jlong offset, jlong size, jintArray src, jlong srcOffset) {
    jboolean isCopy = false;
    auto arr = (char*)env->GetPrimitiveArrayCritical(src, &isCopy);
    clEnqueueWriteBuffer(
            (cl_command_queue)commandQueue, (cl_mem)mem, blockingRead,
            offset, size, arr + srcOffset, 0, nullptr, nullptr
    );
    env->ReleasePrimitiveArrayCritical(src, arr, JNI_ABORT);
    delete[] arr;
}

extern "C" JNIEXPORT void JNICALL Java_com_huskerdev_gpkt_apis_opencl_OpenCLBindings_nWriteByteBuffer(JNIEnv* env, jclass, jlong commandQueue, jlong mem, jboolean blockingRead, jlong offset, jlong size, jbyteArray src, jlong srcOffset) {
    jboolean isCopy = false;
    auto arr = (char*)env->GetPrimitiveArrayCritical(src, &isCopy);
    clEnqueueWriteBuffer(
            (cl_command_queue)commandQueue, (cl_mem)mem, blockingRead,
            offset, size, arr + srcOffset, 0, nullptr, nullptr
    );
    env->ReleasePrimitiveArrayCritical(src, arr, JNI_ABORT);
    delete[] arr;
}

extern "C" JNIEXPORT jlong JNICALL Java_com_huskerdev_gpkt_apis_opencl_OpenCLBindings_nCreateProgram(JNIEnv* env, jclass, jlong context, jstring source, jintArray error) {
    jboolean isCopy = false;
    const char* text = env->GetStringUTFChars(source, &isCopy);

    cl_int err;
    auto result = (jlong) clCreateProgramWithSource((cl_context)context, 1, &text, nullptr, &err);
    env->SetIntArrayRegion(error, 0, 1, &err);
    env->ReleaseStringUTFChars(source, text);
    return result;
}

extern "C" JNIEXPORT jint JNICALL Java_com_huskerdev_gpkt_apis_opencl_OpenCLBindings_nBuildProgram(JNIEnv* env, jclass, jlong program) {
    return clBuildProgram((cl_program)program, 0, nullptr, nullptr, nullptr, nullptr);
}

extern "C" JNIEXPORT jstring JNICALL Java_com_huskerdev_gpkt_apis_opencl_OpenCLBindings_nGetBuildInfo(JNIEnv* env, jclass, jlong program, jlong device) {
    size_t log_size;
    clGetProgramBuildInfo((cl_program)program, (cl_device_id)device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);

    auto log = (char*) malloc(log_size);
    clGetProgramBuildInfo((cl_program)program, (cl_device_id)device, CL_PROGRAM_BUILD_LOG, log_size, log, nullptr);
    return env->NewStringUTF(log);
}

extern "C" JNIEXPORT jlong JNICALL Java_com_huskerdev_gpkt_apis_opencl_OpenCLBindings_nCreateKernel(JNIEnv* env, jclass, jlong program, jstring _main) {
    jboolean isCopy = false;
    const char* main = env->GetStringUTFChars(_main, &isCopy);

    auto result = (jlong) clCreateKernel((cl_program)program, main, nullptr);
    env->ReleaseStringUTFChars(_main, main);
    return result;
}

extern "C" JNIEXPORT jlongArray JNICALL Java_com_huskerdev_gpkt_apis_opencl_OpenCLBindings_nGetKernelWorkGroupInfo(JNIEnv* env, jclass, jlong kernel, jlong device, jint param) {
    size_t size;
    clGetKernelWorkGroupInfo((cl_kernel)kernel, (cl_device_id)device, param, 0, nullptr, &size);

    auto* buffer = new jlong[(int)size];
    clGetKernelWorkGroupInfo((cl_kernel)kernel, (cl_device_id)device, param, size, buffer, nullptr);

    jlongArray res = env->NewLongArray((int)size);
    env->SetLongArrayRegion(res, 0, (int)size, buffer);
    delete[] buffer;
    return res;
}

extern "C" JNIEXPORT void JNICALL Java_com_huskerdev_gpkt_apis_opencl_OpenCLBindings_nEnqueueNDRangeKernel(JNIEnv* env, jclass, jlong commandQueue, jlong kernel, jint workDim, jlongArray _globalWorkSize, jlongArray _localWorkSize) {
    jboolean isCopy = false;
    auto globalWorkSize = (const size_t*) env->GetPrimitiveArrayCritical(_globalWorkSize, &isCopy);
    auto localWorkSize = (const size_t*) env->GetPrimitiveArrayCritical(_localWorkSize, &isCopy);

    clEnqueueNDRangeKernel(
        (cl_command_queue)commandQueue, (cl_kernel)kernel, workDim,
        nullptr, globalWorkSize, localWorkSize,
        0, nullptr, nullptr
    );
    env->ReleasePrimitiveArrayCritical(_globalWorkSize, (void*)globalWorkSize, JNI_ABORT);
    env->ReleasePrimitiveArrayCritical(_localWorkSize, (void*)localWorkSize, JNI_ABORT);
}

extern "C" JNIEXPORT void JNICALL Java_com_huskerdev_gpkt_apis_opencl_OpenCLBindings_nSetKernelArg(JNIEnv* env, jclass, jlong kernel, jint index, jlong mem) {
    clSetKernelArg((cl_kernel)kernel, index, sizeof(cl_mem), (cl_mem)&mem);
}

extern "C" JNIEXPORT void JNICALL Java_com_huskerdev_gpkt_apis_opencl_OpenCLBindings_nSetKernelArg1f(JNIEnv* env, jclass, jlong kernel, jint index, jfloat value) {
    clSetKernelArg((cl_kernel)kernel, index, 4, &value);
}

extern "C" JNIEXPORT void JNICALL Java_com_huskerdev_gpkt_apis_opencl_OpenCLBindings_nSetKernelArg1i(JNIEnv* env, jclass, jlong kernel, jint index, jint value) {
    clSetKernelArg((cl_kernel)kernel, index, 4, &value);
}

extern "C" JNIEXPORT void JNICALL Java_com_huskerdev_gpkt_apis_opencl_OpenCLBindings_nSetKernelArg1b(JNIEnv* env, jclass, jlong kernel, jint index, jbyte value) {
    clSetKernelArg((cl_kernel)kernel, index, 1, &value);
}
