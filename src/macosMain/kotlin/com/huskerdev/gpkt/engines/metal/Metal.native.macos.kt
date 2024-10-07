package com.huskerdev.gpkt.engines.metal

import platform.Metal.MTLCopyAllDevices
import platform.Metal.MTLDeviceProtocol

actual fun getDevices(): Array<MTLDeviceProtocol> =
    MTLCopyAllDevices().map { it as MTLDeviceProtocol }.toTypedArray()