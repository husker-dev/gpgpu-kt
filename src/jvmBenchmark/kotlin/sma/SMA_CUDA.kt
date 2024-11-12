package sma

import com.huskerdev.gpkt.GPApiType
import com.huskerdev.gpkt.GPSyncApi
import org.openjdk.jmh.annotations.*


@State(Scope.Benchmark)
@Suppress("unused")
open class SMA_CUDA {
    private lateinit var gp: GP

    @Setup
    open fun prepare() {
        gp = GP(GPSyncApi.getByType(GPApiType.CUDA)!!.defaultDevice)
    }

    //@Benchmark
    open fun exec() =
        gp.execute()

    @TearDown
    open fun cleanup() =
        gp.cleanup()
}