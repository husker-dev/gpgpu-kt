package sma

import org.openjdk.jmh.annotations.*


@State(Scope.Benchmark)
open class SMA_CPU {
    private val size = 1000
    private lateinit var data: FloatArray
    private lateinit var result: FloatArray

    @Setup
    open fun prepare() {
        data = FloatArray(size) { it.toFloat() }
        result = FloatArray(size * (maxPeriod - minPeriod))
    }

    //@Benchmark
    open fun exec(): FloatArray {
        for(i in 0 until size){
            for(period in minPeriod until maxPeriod)
                result[(period - minPeriod) * size + i] = ema(data, i, period)
        }
        return result
    }

    private fun ema(data: FloatArray, from: Int, period: Int): Float {
        var sum = 0f
        for(r in 0 until period)
            sum += if(from - r < 0) 0f else data[from - r]
        return sum / period
    }
}