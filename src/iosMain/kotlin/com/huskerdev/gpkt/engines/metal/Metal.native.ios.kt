package com.huskerdev.gpkt.engines.metal

import platform.Metal.MTLCreateSystemDefaultDevice
import platform.Metal.MTLDeviceProtocol

actual fun getDevices(): Array<MTLDeviceProtocol> =
    arrayOf(MTLCreateSystemDefaultDevice()!!)