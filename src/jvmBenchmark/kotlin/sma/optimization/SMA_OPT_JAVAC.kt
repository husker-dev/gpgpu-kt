package sma.optimization

import com.huskerdev.gpkt.GPApiType
import com.huskerdev.gpkt.GPSyncApi
import org.openjdk.jmh.annotations.*


@State(Scope.Benchmark)
@Suppress("unused")
open class SMA_OPT_JAVAC {
    private lateinit var gp: GP

    @Setup
    open fun prepare() {
        gp = GP(GPSyncApi.getByType(GPApiType.Javac).defaultDevice)
    }

    @Benchmark
    open fun exec() =
        gp.execute()

    @TearDown
    open fun cleanup() =
        gp.cleanup()
}