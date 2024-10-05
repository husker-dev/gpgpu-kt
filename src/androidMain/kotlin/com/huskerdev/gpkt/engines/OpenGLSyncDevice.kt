package com.huskerdev.gpkt.engines

import com.huskerdev.gpkt.AsyncByteMemoryPointer
import com.huskerdev.gpkt.AsyncDoubleMemoryPointer
import com.huskerdev.gpkt.AsyncFloatMemoryPointer
import com.huskerdev.gpkt.AsyncIntMemoryPointer
import com.huskerdev.gpkt.AsyncLongMemoryPointer
import com.huskerdev.gpkt.GPAsyncDevice
import com.huskerdev.gpkt.GPSyncDevice
import com.huskerdev.gpkt.GPType
import com.huskerdev.gpkt.MemoryUsage
import com.huskerdev.gpkt.Program
import com.huskerdev.gpkt.SyncByteMemoryPointer
import com.huskerdev.gpkt.SyncDoubleMemoryPointer
import com.huskerdev.gpkt.SyncFloatMemoryPointer
import com.huskerdev.gpkt.SyncIntMemoryPointer
import com.huskerdev.gpkt.SyncLongMemoryPointer
import com.huskerdev.gpkt.ast.ScopeStatement

class OpenGLSyncDevice: GPSyncDevice(GPType.OpenGL) {
    override fun compile(ast: ScopeStatement): Program {
        TODO("Not yet implemented")
    }

    override fun allocFloat(array: FloatArray, usage: MemoryUsage): SyncFloatMemoryPointer {
        TODO("Not yet implemented")
    }

    override fun allocFloat(length: Int, usage: MemoryUsage): SyncFloatMemoryPointer {
        TODO("Not yet implemented")
    }

    override fun allocDouble(array: DoubleArray, usage: MemoryUsage): SyncDoubleMemoryPointer {
        TODO("Not yet implemented")
    }

    override fun allocDouble(length: Int, usage: MemoryUsage): SyncDoubleMemoryPointer {
        TODO("Not yet implemented")
    }

    override fun allocLong(array: LongArray, usage: MemoryUsage): SyncLongMemoryPointer {
        TODO("Not yet implemented")
    }

    override fun allocLong(length: Int, usage: MemoryUsage): SyncLongMemoryPointer {
        TODO("Not yet implemented")
    }

    override fun allocInt(array: IntArray, usage: MemoryUsage): SyncIntMemoryPointer {
        TODO("Not yet implemented")
    }

    override fun allocInt(length: Int, usage: MemoryUsage): SyncIntMemoryPointer {
        TODO("Not yet implemented")
    }

    override fun allocByte(array: ByteArray, usage: MemoryUsage): SyncByteMemoryPointer {
        TODO("Not yet implemented")
    }

    override fun allocByte(length: Int, usage: MemoryUsage): SyncByteMemoryPointer {
        TODO("Not yet implemented")
    }

    override val id: Int
        get() = TODO("Not yet implemented")
    override val name: String
        get() = TODO("Not yet implemented")
    override val isGPU: Boolean
        get() = TODO("Not yet implemented")
}

class OpenGLAsyncDevice: GPAsyncDevice(GPType.OpenGL) {
    override fun compile(ast: ScopeStatement): Program {
        TODO("Not yet implemented")
    }

    override fun allocFloat(array: FloatArray, usage: MemoryUsage): AsyncFloatMemoryPointer {
        TODO("Not yet implemented")
    }

    override fun allocFloat(length: Int, usage: MemoryUsage): AsyncFloatMemoryPointer {
        TODO("Not yet implemented")
    }

    override fun allocDouble(array: DoubleArray, usage: MemoryUsage): AsyncDoubleMemoryPointer {
        TODO("Not yet implemented")
    }

    override fun allocDouble(length: Int, usage: MemoryUsage): AsyncDoubleMemoryPointer {
        TODO("Not yet implemented")
    }

    override fun allocLong(array: LongArray, usage: MemoryUsage): AsyncLongMemoryPointer {
        TODO("Not yet implemented")
    }

    override fun allocLong(length: Int, usage: MemoryUsage): AsyncLongMemoryPointer {
        TODO("Not yet implemented")
    }

    override fun allocInt(array: IntArray, usage: MemoryUsage): AsyncIntMemoryPointer {
        TODO("Not yet implemented")
    }

    override fun allocInt(length: Int, usage: MemoryUsage): AsyncIntMemoryPointer {
        TODO("Not yet implemented")
    }

    override fun allocByte(array: ByteArray, usage: MemoryUsage): AsyncByteMemoryPointer {
        TODO("Not yet implemented")
    }

    override fun allocByte(length: Int, usage: MemoryUsage): AsyncByteMemoryPointer {
        TODO("Not yet implemented")
    }


    override val id: Int
        get() = TODO("Not yet implemented")
    override val name: String
        get() = TODO("Not yet implemented")
    override val isGPU: Boolean
        get() = TODO("Not yet implemented")
}