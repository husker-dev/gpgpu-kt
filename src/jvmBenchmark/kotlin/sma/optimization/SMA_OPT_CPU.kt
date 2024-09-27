package sma.optimization

import org.openjdk.jmh.annotations.*


@State(Scope.Benchmark)
open class SMA_OPT_CPU {
    private lateinit var data: FloatArray
    private lateinit var sma: FloatArray
    private lateinit var result: FloatArray

    @Setup
    open fun prepare() {
        data = FloatArray(candles) { it.toFloat() }
        sma = FloatArray(candles * (maxPeriod - minPeriod) * (maxShift - minShift))
        result = FloatArray(sma.size)
    }

    @Benchmark
    open fun exec(): FloatArray {
        // Calculate SMA
        for(i in sma.indices){
            val deltaPeriod = maxPeriod - minPeriod
            val deltaShift = maxShift - minShift

            val currentIndex = i % candles
            val currentBlock = i / candles

            val currentPeriod = minPeriod + currentBlock % deltaPeriod
            val currentShift = minShift + (currentBlock / deltaPeriod) % deltaShift

            sma[i] = sma(data, currentIndex + currentShift, currentPeriod, candles)
        }

        // Calculate signals
        for(i in sma.indices){
            val currentIndex = i % candles

            result[i] = maSignals(sma, data, i, currentIndex)
        }
        return result
    }

    private fun sma(d: FloatArray, from: Int, period: Int, size: Int): Float{
        var sum = 0f
        for(i in 0 until period) {
            val ni = from - i
            if (ni in 0..<size)
                sum += d[from - i]
        }
        return sum / period
    }

    private fun maSignals(ma: FloatArray, close: FloatArray, maI: Int, closeI: Int): Float{
        if(closeI <= 2 || maI <= 2)
            return 0f
        return if(close[closeI-2] <= ma[maI-2] && close[closeI-1] > ma[maI-1])
            1f
        else 0f
    }
}