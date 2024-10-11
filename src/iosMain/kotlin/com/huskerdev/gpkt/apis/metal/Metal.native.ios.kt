package com.huskerdev.gpkt.apis.metal

import platform.Metal.MTLCreateSystemDefaultDevice
import platform.Metal.MTLDeviceProtocol

actual fun getDevices(): Array<MTLDeviceProtocol> =
    arrayOf(MTLCreateSystemDefaultDevice()!!)