package sma.optimization

import com.huskerdev.gpkt.GPSyncDevice
import com.huskerdev.gpkt.GPType
import org.openjdk.jmh.annotations.*


@State(Scope.Benchmark)
open class SMA_OPT_CL {
    private lateinit var gp: GP

    @Setup
    open fun prepare() {
        gp = GP(GPSyncDevice.create(requestedType = arrayOf(GPType.OpenCL))!!)
    }

    @Benchmark
    open fun exec() =
        gp.execute()

    @TearDown
    open fun cleanup() =
        gp.cleanup()
}