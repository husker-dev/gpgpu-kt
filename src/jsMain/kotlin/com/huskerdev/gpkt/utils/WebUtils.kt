package com.huskerdev.gpkt.utils

import org.khronos.webgl.*
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlin.coroutines.suspendCoroutine
import kotlin.js.Promise

inline fun jsObject(init: dynamic.() -> Unit): dynamic {
    val o = js("{}")
    init(o)
    return o
}

fun FloatArray.toArrayBuffer(): ArrayBuffer {
    val buffer = ArrayBuffer(size * Float.SIZE_BYTES)
    val tmp = Float32Array(buffer)
    tmp.set(toTypedArray())
    return tmp.buffer
}

fun DoubleArray.toArrayBuffer(): ArrayBuffer {
    val buffer = ArrayBuffer(size * Double.SIZE_BYTES)
    val tmp = Float64Array(buffer)
    tmp.set(toTypedArray())
    return tmp.buffer
}

fun IntArray.toArrayBuffer(): ArrayBuffer {
    val buffer = ArrayBuffer(size * Int.SIZE_BYTES)
    val tmp = Int32Array(buffer)
    tmp.set(toTypedArray())
    return tmp.buffer
}

fun LongArray.toArrayBuffer(): ArrayBuffer {
    val buffer = ArrayBuffer(size * Long.SIZE_BYTES)
    val tmp = Int32Array(buffer)
    tmp.set(flatMap {
        listOf(
            (it shr 32).toInt(),
            (it and 0x00000000FFFFFFFF).toInt()
        )
    }.toTypedArray())
    return tmp.buffer
}

fun ByteArray.toArrayBuffer(): ArrayBuffer {
    val buffer = ArrayBuffer(size)
    val tmp = Int8Array(buffer)
    tmp.set(toTypedArray())
    return tmp.buffer
}

suspend fun <T> Promise<T>.await(): T = suspendCoroutine { cont ->
    then({ cont.resume(it) }, { cont.resumeWithException(it) })
}