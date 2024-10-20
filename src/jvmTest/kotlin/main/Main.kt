package main

import com.huskerdev.gpkt.GPApiType
import com.huskerdev.gpkt.GPSyncApi
import com.huskerdev.gpkt.ast.GPCompilationException
import kotlin.system.exitProcess


// 1 2 3 4 5 6 7 8
fun exampleArray() = FloatArray(100) { it.toFloat() }

fun main() {
    val device = GPSyncApi.getByType(GPApiType.Interpreter)!!.defaultDevice
    val context = device.createContext()
    println("======== Device info ========")
    println("Type: ${device.api.type}")
    println("Name: ${device.name}")
    println("=============================")

    context.modules.add("test", """
        int test(int index){
            if(index == 0) return 100;
            if(index == 1) return 101;
            if(index == 2) return 102;
            if(index == 3) return 103;
            return 99;
        }
    """.trimIndent())

    context.modules.add("sma", """
        const int a = 1;
        
        float sma(float[] d, int from, int period){
            float sum = 0;
            for(int i = 0; i < period; i++)
                if(from - i >= 0) sum += d[from - i];
            return sum / (float)period;
        }
    """.trimIndent())

    val program = try {
        context.compile("""
            import sma, test;
            
            extern float[] data;
            extern float[] result;
            
            extern int minPeriod;
            extern int maxPeriod;
            extern int count;
            
            int b = 12;
            
            
            void main(const int i){
                int localPeriod = i / (maxPeriod - minPeriod) + minPeriod;
                int localCandle = i % (maxPeriod - minPeriod);
                
                b = sin(23)*10;
            
                //result[i] = sma(data, localCandle, localPeriod) + a;
                result[i] = test(0) + sma(data, localCandle, localPeriod) + a + b;
            }
        """.trimIndent())
    }catch (e: GPCompilationException){
        System.err.println(e.message)
        exitProcess(0)
    }

    val arr1 = context.wrapFloats(exampleArray())
    val result = context.allocFloats(arr1.length)
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