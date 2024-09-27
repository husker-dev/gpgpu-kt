package sma.optimization

import com.huskerdev.gpkt.GPDevice
import com.huskerdev.gpkt.GPType
import org.openjdk.jmh.annotations.*


@State(Scope.Benchmark)
open class SMA_OPT_JAVAC {
    private lateinit var gp: GP

    @Setup
    open fun prepare() {
        gp = GP(GPDevice.create(requestedType = arrayOf(GPType.Javac))!!)
    }

    @Benchmark
    open fun exec() =
        gp.execute()

    @TearDown
    open fun cleanup() =
        gp.cleanup()
}