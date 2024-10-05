package sma.optimization

import com.huskerdev.gpkt.GPSyncDevice
import com.huskerdev.gpkt.GPType

const val candles = 1000

const val minPeriod = 2
const val maxPeriod = 500

const val minShift = 0
const val maxShift = 200


class GP(
    device: GPSyncDevice
) {
    private var data = device.wrapFloats(FloatArray(candles) { Math.random().toFloat() * 10000 })
    private var sma = device.allocFloats(candles * (maxPeriod - minPeriod) * (maxShift - minShift))
    private var result = device.allocFloats(sma.length)

    private var progSMA = device.compile("""
        extern float[] candlesData;
        extern float[] smaData;
        
        int minPeriod = $minPeriod;
        int maxPeriod = $maxPeriod;
        int minShift = $minShift;
        int maxShift = $maxShift;
        int candles = $candles;
        
        float sma(float[] d, int from, int period, int size){
            float sum = 0;
            for(int i = 0; i < period; i++){
                int ni = from - i;
                if(ni >= 0 && ni < size) 
                    sum += d[from - i];
            }
            return sum / (float)period;
        }
        
        void main(int i){
            int deltaPeriod = maxPeriod - minPeriod;
            int deltaShift = maxShift - minShift;
            
            int currentIndex = i % candles;
            int currentBlock = i / candles;
            
            int currentPeriod = minPeriod + currentBlock % deltaPeriod;
            int currentShift = minShift + (currentBlock / deltaPeriod) % deltaShift;
            
            smaData[i] = sma(candlesData, currentIndex + currentShift, currentPeriod, candles);
        }
    """.trimIndent())

    private var progSignals = device.compile("""
        extern float[] smaData;
        extern float[] closeData;
        extern float[] result;
        
        int candles = $candles;
        
        float maSignals(float[] ma, float[] close, int maI, int closeI){
            if(closeI <= 2 || maI <= 2)
                return 0;
            if(close[closeI-2] <= ma[maI-2] && close[closeI-1] > ma[maI-1])
                return 1;
            else return 0;
        }
        
        void main(int i){
            int currentIndex = i % candles;
        
            result[i] = maSignals(smaData, closeData, i, currentIndex);
        }
    """.trimIndent())


    fun execute(): FloatArray{
        progSMA.execute(
            sma.length,
            "candlesData" to data,
            "smaData" to sma
        )
        progSignals.execute(
            sma.length,
            "closeData" to data,
            "smaData" to sma,
            "result" to result
        )
        result.read()
        return FloatArray(0)
    }

    fun cleanup() {
        data.dealloc()
        sma.dealloc()
        result.dealloc()
        progSMA.dealloc()
        progSignals.dealloc()
    }
}

fun main(){
    GP(GPSyncDevice.create(requestedType = arrayOf(GPType.CUDA))!!).apply {
        execute()
        cleanup()
    }
}