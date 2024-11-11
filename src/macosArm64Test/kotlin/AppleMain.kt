import com.huskerdev.gpkt.GPSyncApi
import com.huskerdev.gpkt.ast.GPCompilationException
import kotlin.system.exitProcess
import kotlin.test.Test


fun exampleArray() = FloatArray(100) { it.toFloat() }

@Test
fun main(){
    val api = GPSyncApi.getDefault()
    val device = api.defaultDevice
    val context = device.createContext()
    println("======== Device info ========")
    println("Type: ${api.type}")
    println("Name: ${device.name}")
    println("=============================")

    context.modules["sma"] = {"""
        float sma(float[] d, int from, int period){
            float sum = 0;
            for(int i = 0; i < period; i++)
                if(from - i >= 0) sum += d[from - i];
            return sum / (float)period;
        }
    """.trimIndent()}

    val program = try {
        context.compile("""
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
        e.printStackTrace()
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