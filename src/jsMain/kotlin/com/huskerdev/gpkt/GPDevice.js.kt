package com.huskerdev.gpkt

import com.huskerdev.gpkt.engines.js.JSAsyncDevice
import com.huskerdev.gpkt.engines.js.JSSyncDevice
import com.huskerdev.gpkt.engines.webgpu.WebGPU
import com.huskerdev.gpkt.engines.webgpu.WebGPUAsyncDevice
import kotlin.coroutines.Continuation
import kotlin.coroutines.CoroutineContext
import kotlin.coroutines.EmptyCoroutineContext
import kotlin.coroutines.startCoroutine
import kotlin.js.Promise

internal actual val defaultExpectedTypes = arrayOf(GPType.WebGPU, GPType.JS)
internal actual val defaultExpectedDeviceId = 0

internal actual fun createSupportedSyncInstance(
    requestedDeviceId: Int,
    requestedType: Array<out GPType>
): GPSyncDevice? = requestedType.firstNotNullOfOrNull {
    when {
        it == GPType.JS -> JSSyncDevice()
        else -> null
    }
}

internal actual suspend fun createSupportedAsyncInstance(
    requestedDeviceId: Int,
    requestedType: Array<out GPType>
): GPAsyncDevice? = requestedType.firstNotNullOfOrNull {
    when {
        it == GPType.WebGPU && WebGPU.supported -> WebGPUAsyncDevice.create()
        it == GPType.JS -> JSAsyncDevice()
        else -> null
    }
}


@OptIn(ExperimentalJsExport::class, ExperimentalJsStatic::class)
@JsExport
@JsName("test")
@JsStatic
fun test() = Promise<Boolean> { _, _ ->
    launch {
        try {
            val device = GPAsyncDevice.create()!!
            println("Device created: ${device.name}")

            val result = device.allocFloat(FloatArray(20))
            val data = device.allocFloat(FloatArray(20){ it.toFloat() })

            val program = device.compile("""
                extern readonly float[] data;
                extern float[] result;
                
                float a(float b){
                    return 13f;
                }
                
                void main(const int i){
                    result[i] = data[i] + a(23) + i;
                }
            """.trimIndent())

            program.execute(20,
                "data" to data,
                "result" to result
            )

            println(result.read().toList())
        }catch (e: Throwable){
            e.printStackTrace()
            println(e.message)
        }
    }
}

fun <T> launch(block: suspend () -> T) = block.startCoroutine(object : Continuation<T> {
    override val context: CoroutineContext
        get() = EmptyCoroutineContext
    override fun resumeWith(result: Result<T>) = Unit
})