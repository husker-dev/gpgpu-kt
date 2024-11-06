package main

import com.huskerdev.gpkt.GPApiType
import com.huskerdev.gpkt.GPSyncApi
import com.huskerdev.gpkt.ast.GPCompilationException
import kotlin.system.exitProcess


// 1 2 3 4 5 6 7 8
fun exampleArray() = FloatArray(100) { it.toFloat() }

fun main() {
    val device = GPSyncApi.getByType(GPApiType.CUDA)!!.defaultDevice
    val context = device.createContext()
    println("======== Device info ========")
    println("Type: ${device.api.type}")
    println("Name: ${device.name}")
    println("=============================")

    context.modules.add("test", """
        // gpkt
        float a = 0.01f;
        int toImpl();
        
        int test(int index){
            if(index == 0) return 100;
            if(index == 1) return 101;
            if(index == 2) return 102;
            if(index == 3) return 103;
            return 99;
        }
    """.trimIndent())

    context.modules.add("sma", """
        // gpkt
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
            // gpkt
            import sma, test;
            
            extern float[] data;
            extern float[] result;
            
            int b = 12;
            
            float[3] getArray(){
                return { 6f, 2f, 4f };
            }
            
            int toImpl(){
                return 2;
            }
            
            
            class Floats(
                float[] arr,
                int index
            ): Float {
                float getFloat(){
                    return arr[index];
                }
                void setFloat(float a){
                    arr[index] = a;
                }
            }
            
            void main(const int i){
                float[3] asgdf = { 1f, 2f, 3f };
            
                var myVariable = new Floats(data, i);
                
                result[i] = myVariable;
            }
        """.trimIndent())
    }catch (e: GPCompilationException){
        e.printStackTrace()
        System.err.println(e.message)
        exitProcess(0)
    }

    val arr1 = context.wrapFloats(exampleArray())
    val result = context.allocFloats(arr1.length)
    program.execute(
        instances = result.length,
        "data" to arr1,
        "result" to result,
    )

    val r = result.read()
    println("${r.toList().take(20)}...")
    println("...${r.toList().takeLast(20)}")
}