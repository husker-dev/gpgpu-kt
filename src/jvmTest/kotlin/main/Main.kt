package main

import com.huskerdev.gpkt.GPApiType
import com.huskerdev.gpkt.GPSyncApi
import com.huskerdev.gpkt.ast.GPCompilationException
import kotlin.random.Random
import kotlin.system.exitProcess


// 1 2 3 4 5 6 7 8
fun exampleArray(): FloatArray {
    val random = Random(100)
    return FloatArray(100) { (random.nextFloat() * 2 - 1) * 5 }
}

fun print(type: GPApiType): ByteArray {
    val device = GPSyncApi.getByType(type)!!.defaultDevice
    val context = device.createContext()
    println("======== Device info ========")
    println("Type: ${device.api.type}")
    println("Name: ${device.name}")
    println("=============================")

    context.modules.add("test", """
        // gpkt
        const int a = 2, a1 = 3;
        int toImpl();
        
        int test(float[] data, int index){
            if(data[index] < 10)
                return 1;
            if(data[index] > 10)
                return 2;
            return 99;
        }
    """.trimIndent())

    context.modules.add("sma", """
        // gpkt
        import test;
        float sma(float[] d, int from, int period){
            float sum = 0;
            for(int i = 0; i < period; i++)
                if(from - i >= 0) sum += d[from - i];
            return sum / (float)period;
        }
        
        byte zeroCrossSignals(float[] d, int i){
            if(0 >= d[i - 1] && 0 < d[i])
                return 2;
            if(0 <= d[i - 1] && 0 > d[i])
                return 1;
            return 0;
        }
    """.trimIndent())

    val program = try {
        context.compile("""
            // gpkt
            import sma;
            
            extern float[] data;
            extern byte[] result;
            
            
            
            void main(const int i){
                float[3] asgdf = { 1f, 2f, 3f };
            
                result[i] = zeroCrossSignals(data, i);
            }
        """.trimIndent())
    }catch (e: GPCompilationException){
        e.printStackTrace()
        System.err.println(e.message)
        exitProcess(0)
    }

    val arr1 = context.wrapFloats(exampleArray())
    val result = context.allocBytes(arr1.length)
    program.execute(
        instances = result.length,
        "data" to arr1,
        "result" to result
    )

    val r = result.read()
    println("${r.toList().take(20)}...")
    println("...${r.toList().takeLast(20)}")
    return r
}

fun main() {
    println("Equal: ${print(GPApiType.OpenCL).contentEquals(print(GPApiType.Interpreter))}")

}