package com.huskerdev.gpkt

interface GPDevice {
    val api: GPApi
    val name: String
    val isAccelerated: Boolean
}

interface GPSyncDevice: GPDevice {
    fun createContext(): GPSyncContext
}

interface GPAsyncDevice: GPDevice {
    suspend fun createContext(): GPAsyncContext
}