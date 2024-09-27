package main

import com.huskerdev.gpkt.GPDevice
import com.huskerdev.gpkt.GPType
import com.huskerdev.gpkt.ast.GPCompilationException
import kotlin.system.exitProcess


// 1 2 3 4 5 6 7 8
fun exampleArray() = FloatArray(100) { it.toFloat() }

fun main() {
    val device = GPDevice.create(requestedType = arrayOf(GPType.OpenCL))!!
    println("======== Device info ========")
    println("Type: ${device.type}")
    println("Name: ${device.name}")
    println("Id:   ${device.id}")
    println("=============================")

    val program = try {
        device.compile("""
            external float[] data;
            external float[] result;
            
            int minPeriod = ${1};
            int maxPeriod = ${11};
            int count = ${10};
            
            float sma(float[] d, int from, int period){
                float sum = 0;
                for(int i = 0; i < period; i++){
                    if(from - i >= 0) sum += d[from - i];
                }
                return sum / (float)period;
            }
            
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

    val arr1 = device.allocFloat(exampleArray())
    val result = device.allocFloat(arr1.length)
    program.execute(
        instances = result.length,
        "data" to arr1,
        "result" to result
    )

    val r = result.read()
    println(r.toList().take(20))
    println(r.toList().takeLast(20))
}