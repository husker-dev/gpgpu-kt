package sma

import com.huskerdev.gpkt.GPApiType
import com.huskerdev.gpkt.GPSyncApi
import org.openjdk.jmh.annotations.*


@State(Scope.Benchmark)
@Suppress("unused")
open class SMA_INTERPRETER {
    private lateinit var gp: GP

    @Setup
    open fun prepare() {
        gp = GP(GPSyncApi.getByType(GPApiType.Interpreter)!!.defaultDevice)
    }

    //@Benchmark
    open fun exec() =
        gp.execute()

    @TearDown
    open fun cleanup() =
        gp.cleanup()
}