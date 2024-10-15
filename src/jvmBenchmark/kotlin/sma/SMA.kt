package sma

import com.huskerdev.gpkt.GPSyncDevice

val candles = 1000

val minPeriod = 2
val maxPeriod = 500


class GP(
    device: GPSyncDevice
) {
    private val context = device.createContext()

    private var data = context.wrapFloats(FloatArray(candles) { it.toFloat() })
    private var result = context.allocFloats(candles * (maxPeriod - minPeriod))

    private var program = context.compile("""
        // gpkt
        extern float[] data;
        extern float[] result;
        
        int minPeriod = ${minPeriod};
        int maxPeriod = ${maxPeriod};
        int count = ${candles};
        
        float sma(float[] d, int from, int period){
            float sum = 0;
            for(int i = 0; i < period; i++)
                if(from - i >= 0) sum += d[from - i];
            return sum / (float)period;
        }
        
        void main(int i){
            int currentPeriod = i / (maxPeriod - minPeriod) + minPeriod;
            int currentIndex = i % (maxPeriod - minPeriod);
        
            result[i] = sma(data, currentIndex, currentPeriod);
        }
    """.trimIndent())

    fun execute(): FloatArray{
        program.execute(
            result.length,
            "data" to data,
            "result" to result
        )
        return result.read()
    }

    fun cleanup() {
        context.dispose()
        data.dealloc()
        result.dealloc()
        program.dealloc()
    }
}