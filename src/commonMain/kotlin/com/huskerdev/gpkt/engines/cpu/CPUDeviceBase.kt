package com.huskerdev.gpkt.engines.cpu

import com.huskerdev.gpkt.GPAsyncDevice
import com.huskerdev.gpkt.GPSyncDevice
import com.huskerdev.gpkt.GPType
import com.huskerdev.gpkt.MemoryUsage

abstract class CPUSyncDeviceBase(type: GPType): GPSyncDevice(type) {
    override val id = 0
    override val name = "CPU"
    override val isGPU = false

    override fun allocFloat(array: FloatArray, usage: MemoryUsage) =
        CPUSyncFloatMemoryPointer(array.copyOf(), usage)

    override fun allocFloat(length: Int, usage: MemoryUsage) =
        CPUSyncFloatMemoryPointer(FloatArray(length), usage)

    override fun allocDouble(array: DoubleArray, usage: MemoryUsage) =
        CPUSyncDoubleMemoryPointer(array.copyOf(), usage)

    override fun allocDouble(length: Int, usage: MemoryUsage) =
        CPUSyncDoubleMemoryPointer(DoubleArray(length), usage)

    override fun allocInt(array: IntArray, usage: MemoryUsage) =
        CPUSyncIntMemoryPointer(array.copyOf(), usage)

    override fun allocInt(length: Int, usage: MemoryUsage) =
        CPUSyncIntMemoryPointer(IntArray(length), usage)

    override fun allocByte(array: ByteArray, usage: MemoryUsage) =
        CPUSyncByteMemoryPointer(array.copyOf(), usage)

    override fun allocByte(length: Int, usage: MemoryUsage) =
        CPUSyncByteMemoryPointer(ByteArray(length), usage)
}


abstract class CPUAsyncDeviceBase(type: GPType): GPAsyncDevice(type) {
    override val id = 0
    override val name = "CPU"
    override val isGPU = false

    override fun allocFloat(array: FloatArray, usage: MemoryUsage) =
        CPUAsyncFloatMemoryPointer(array.copyOf(), usage)

    override fun allocFloat(length: Int, usage: MemoryUsage) =
        CPUAsyncFloatMemoryPointer(FloatArray(length), usage)

    override fun allocDouble(array: DoubleArray, usage: MemoryUsage) =
        CPUAsyncDoubleMemoryPointer(array.copyOf(), usage)

    override fun allocDouble(length: Int, usage: MemoryUsage) =
        CPUAsyncDoubleMemoryPointer(DoubleArray(length), usage)

    override fun allocInt(array: IntArray, usage: MemoryUsage) =
        CPUAsyncIntMemoryPointer(array.copyOf(), usage)

    override fun allocInt(length: Int, usage: MemoryUsage) =
        CPUAsyncIntMemoryPointer(IntArray(length), usage)

    override fun allocByte(array: ByteArray, usage: MemoryUsage) =
        CPUAsyncByteMemoryPointer(array.copyOf(), usage)

    override fun allocByte(length: Int, usage: MemoryUsage) =
        CPUAsyncByteMemoryPointer(ByteArray(length), usage)
}