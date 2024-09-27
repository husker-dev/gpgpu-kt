package com.huskerdev.gpkt.engines.cpu

import com.huskerdev.gpkt.*


class CPUFloatMemoryPointer(
    var array: FloatArray?
): FloatMemoryPointer {
    override fun read() = array!!
    override val length = array!!.size
    override fun dealloc() {
        array = null
    }
}

class CPUDoubleMemoryPointer(
    var array: DoubleArray?
): DoubleMemoryPointer {
    override fun read() = array!!
    override val length = array!!.size
    override fun dealloc() {
        array = null
    }
}

class CPULongMemoryPointer(
    var array: LongArray?
): LongMemoryPointer {
    override fun read() = array!!
    override val length = array!!.size
    override fun dealloc() {
        array = null
    }
}

class CPUIntMemoryPointer(
    var array: IntArray?
): IntMemoryPointer {
    override fun read() = array!!
    override val length = array!!.size
    override fun dealloc() {
        array = null
    }
}

class CPUByteMemoryPointer(
    var array: ByteArray?
): ByteMemoryPointer {
    override fun read() = array!!
    override val length = array!!.size
    override fun dealloc() {
        array = null
    }
}