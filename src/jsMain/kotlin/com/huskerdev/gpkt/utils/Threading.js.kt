package com.huskerdev.gpkt.utils

actual val threads = 1

actual fun runThread(f: () -> Unit): AbstractThread =
    throw UnsupportedOperationException("Can't run threads in Web")