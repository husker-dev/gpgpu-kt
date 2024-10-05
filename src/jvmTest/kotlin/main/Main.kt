package main

import com.huskerdev.gpkt.GPSyncDevice
import com.huskerdev.gpkt.GPType
import com.huskerdev.gpkt.ast.GPCompilationException
import kotlin.system.exitProcess


// 1 2 3 4 5 6 7 8
fun exampleArray() = FloatArray(100) { it.toFloat() }

fun main() {
    val device = GPSyncDevice.create(requestedType = arrayOf(GPType.OpenCL))!!
    println("======== Device info ========")
    println("Type: ${device.type}")
    println("Name: ${device.name}")
    println("Id:   ${device.id}")
    println("=============================")

    device.modules.add("sma", """
        float sma(float[] d, int from, int period){
            float sum = 0;
            for(int i = 0; i < period; i++)
                if(from - i >= 0) sum += d[from - i];
            return sum / (float)period;
        }
    """.trimIndent())

    val program = try {
        device.compile("""
            import sma;
            
            extern float[] data;
            extern float[] result;
            
            extern int minPeriod;
            extern int maxPeriod;
            extern int count;
            
            void main(const int i){
                int localPeriod = i / (maxPeriod - minPeriod) + minPeriod;
                int localCandle = i % (maxPeriod - minPeriod);
            
                result[i] = sma(data, localCandle, localPeriod);
            }
        """.trimIndent())
    }catch (e: GPCompilationException){
        System.err.println(e.message)
        exitProcess(0)
    }

    val arr1 = device.wrapFloats(exampleArray())
    val result = device.allocFloats(arr1.length)
    program.execute(
        instances = result.length,
        "data" to arr1,
        "result" to result,
        "minPeriod" to 1,
        "maxPeriod" to 11,
        "count" to 10
    )

    val r = result.read()
    println(r.toList().take(20))
    println(r.toList().takeLast(20))
}