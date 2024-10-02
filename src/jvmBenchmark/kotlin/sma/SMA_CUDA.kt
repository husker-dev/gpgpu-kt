package sma

import com.huskerdev.gpkt.GPSyncDevice
import com.huskerdev.gpkt.GPType
import org.openjdk.jmh.annotations.*


@State(Scope.Benchmark)
open class SMA_CUDA {
    private lateinit var gp: GP

    @Setup
    open fun prepare() {
        gp = GP(GPSyncDevice.create(requestedType = arrayOf(GPType.CUDA))!!)
    }

    @Benchmark
    open fun exec() =
        gp.execute()

    @TearDown
    open fun cleanup() =
        gp.cleanup()
}