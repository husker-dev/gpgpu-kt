package com.huskerdev.gpkt


internal expect val supportedApis: Array<GPApiType>
internal expect val defaultSyncApisOrder: Array<GPApiType>
internal expect val defaultAsyncApisOrder: Array<GPApiType>
internal expect val defaultDeviceId: Int

internal expect fun createSyncApiInstance(type: GPApiType): GPSyncApi?
internal expect suspend fun createAsyncApiInstance(type: GPApiType): GPAsyncApi?


interface GPApi {
    val type: GPApiType
}

@Suppress("unused")
interface GPSyncApi: GPApi {
    companion object {
        private val supportedSyncApisMap = defaultSyncApisOrder.mapNotNull(::createSyncApiInstance).associateBy { it.type }

        fun getAll() =
            supportedSyncApisMap.values

        fun getDefault() =
            defaultSyncApisOrder.firstNotNullOf { supportedSyncApisMap[it] }

        fun getByType(type: GPApiType) =
            supportedSyncApisMap[type]
    }

    val devices: Array<GPSyncDevice>

    val defaultDevice: GPSyncDevice
        get() = devices[defaultDeviceId.coerceIn(0, devices.lastIndex)]
}

@Suppress("unused")
interface GPAsyncApi: GPApi {
    companion object {
        private lateinit var supportedAsyncApisMap: Map<GPApiType, GPAsyncApi>

        suspend fun getAll(): List<GPAsyncApi> {
            checkAsyncInit()
            return defaultAsyncApisOrder.mapNotNull { createAsyncApiInstance(it) }
        }

        suspend fun getDefault(): GPAsyncApi {
            checkAsyncInit()
            return defaultAsyncApisOrder.firstNotNullOf { createAsyncApiInstance(it) }
        }

        suspend fun getByType(type: GPApiType): GPAsyncApi? {
            checkAsyncInit()
            return supportedAsyncApisMap[type]
        }

        private suspend fun checkAsyncInit(){
            if(!::supportedAsyncApisMap.isInitialized)
                supportedAsyncApisMap = defaultAsyncApisOrder.mapNotNull { createAsyncApiInstance(it) }.associateBy { it.type }
        }
    }

    val devices: Array<GPAsyncDevice>

    val defaultDevice: GPAsyncDevice
        get() = devices[defaultDeviceId.coerceIn(0, devices.lastIndex)]
}