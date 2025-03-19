package com.huskerdev.gpkt.utils

actual val threads = 1

actual fun runThread(f: () -> Unit): AbstractThread = object: AbstractThread{
    init {
        f()
    }
    override fun waitEnd() = Unit
}

actual val ram: ULong = 0u