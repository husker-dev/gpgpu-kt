package com.huskerdev.gpkt.utils

import org.lwjgl.system.MemoryStack
import org.lwjgl.system.MemoryStack.stackPush
import java.nio.ByteBuffer


inline fun <T> useStack(block: MemoryStack.() -> T) =
    stackPush().use(block)

fun ByteBuffer.readArray(): ByteArray {
    val array = ByteArray(capacity())
    for(i in array.indices)
        array[i] = get(i)
    return array
}