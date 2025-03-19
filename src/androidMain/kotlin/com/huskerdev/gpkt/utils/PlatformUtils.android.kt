package com.huskerdev.gpkt.utils

import kotlin.concurrent.thread

actual val threads = Runtime.getRuntime().availableProcessors()

actual fun runThread(f: () -> Unit) = object: AbstractThread {
    val thread = thread(block = f, isDaemon = true)

    override fun waitEnd() = thread.join()
}

actual val ram: ULong = 0u