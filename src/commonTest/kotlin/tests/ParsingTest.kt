package tests

import com.huskerdev.gpkt.GPAst
import com.huskerdev.gpkt.ast.objects.GPField
import com.huskerdev.gpkt.ast.objects.GPFunction
import com.huskerdev.gpkt.utils.CProgramPrinter
import kotlin.test.Test


class ParsingTest {
    @Test
    fun parsing(){
        val ast = GPAst.parse("""
            int a;
            int b;
            int d = 10, e;
            int f, g = 10;
            int h = 10, i = 10;
            
            float _f1 = 1f;
            float _f2 = 1.0f;
            float _f3 = 1.f;
            float _f4 = .2f;
            float _f5 = 1;
            float _f6 = (byte)1;
            
            int _i1 = 0b1;
            int _i2 = 0xffFFffff;
            
            extern float[] inputs1, inputs2;
            extern float[] outputs1, outputs2;
            
            void emptyFunc(){
            }
            
            float floatFunc(int arg1, float arg2){
                return 1.2f;
            }
            
            int intFunc(int arg1, float arg2){
                return 0;
            }
            
            boolean boolFunc(int arg1, float arg2){
                return true;
            }
            
            void main(int index){
                /* ================= *\
                      Expressions
                \* ================= */
                emptyFunc();
                123 + 321;
                123 + intFunc(123, 321f);
                int c = 10;
                c = 123 + intFunc(123, 321f);
                
            
                /* ================= *\
                         Parse
                \* ================= */
                int iVar1 = (int)123f;
                int iVar2 = (int)(123f + floatFunc(1, 1f));
                int iVar3 = (int)(float)1;
                
                float fVar1 = (float)123;
                float fVar2 = (float)(123 + intFunc(1, 1f));
                float fVar3 = (float)(int)1;
            
                /* ================= *\
                       Functions
                \* ================= */
            
                floatFunc(c, 2f);
                intFunc(d, 1.23f);
                
                /* ================= *\
                      If statement
                \* ================= */
                
                if(boolFunc(1, 1.2f)){
                    intFunc(d, 1.23f);
                }
                
                if(boolFunc(1, 1.2f) == true){
                    intFunc(d, 1.23f);
                }
                
                if(123 == 321){
                    intFunc(d, 1.23f);
                }
                
                if(boolFunc(1, 1.2f))
                    intFunc(d, 1.23f);
                    
                if(boolFunc(1, 1.2f)){
                    intFunc(d, 1.23f);
                }else {
                    floatFunc(c, 2.2f);
                }
                
                if(boolFunc(1, 1.2f)){
                    intFunc(d, 1.23f);
                }else
                    floatFunc(c, 2.2f);
                
                if(boolFunc(1, 1.2f))
                    intFunc(d, 1.23f);
                else {
                    floatFunc(c, 2.2f);
                }
                
                if(boolFunc(1, 1.2f))
                    intFunc(d, 1.23f);
                else
                    floatFunc(c, 2.2f);
                    
                /* ================= *\
                       For loops
                \* ================= */
                
                for(int it = 0; it < 100; it++){
                    intFunc(d, 1.23f);
                }
                
                for(int it = 0; it < 100; it++){
                    intFunc(d, 1.23f);
                    break;
                }
                
                for(int it = 0; it < 100; it++){
                    intFunc(d, 1.23f);
                    continue;
                }
                
                for(int it = 0; it < 100; it++)
                    intFunc(d, 1.23f);
                
                for(int it = 0; it < 100;)
                    intFunc(d, 1.23f);
                    
                for(int it = 0;;)
                    intFunc(d, 1.23f);
                    
                for(;;)
                    intFunc(d, 1.23f);
                    
                for(;c < 100;)
                    intFunc(d, 1.23f);
                    
                for(;c < 100;c++)
                    intFunc(d, 1.23f);
                    
                /* ================= *\
                      While loops
                \* ================= */
                
                while(boolFunc(1, 1.2f)) {
                    intFunc(d, 1.23f);
                }
                
                while(boolFunc(1, 1.2f)) {
                    intFunc(d, 1.23f);
                    break;
                }
                
                while(boolFunc(1, 1.2f)) {
                    intFunc(d, 1.23f);
                    continue;
                }
                
                while(true) {
                    intFunc(d, 1.23f);
                }
                
                while(true)
                    intFunc(d, 1.23f);
            }
        """.trimIndent())

        object: CProgramPrinter(
            ast,
                ast.fields.filter {
                it.value.isExtern
            }.map { it.value }.toList(),
            ast.fields.filter {
                it.value.isLocal
            }.map { it.value }.toList()){

            override fun stringifyMainFunctionDefinition(header: MutableMap<String, String>, buffer: StringBuilder, function: GPFunction) = Unit
            override fun stringifyMainFunctionBody(header: MutableMap<String, String>, buffer: StringBuilder, function: GPFunction) = Unit
            override fun stringifyModifiersInStruct(field: GPField) = ""
            override fun stringifyModifiersInGlobal(obj: Any) = ""
            override fun stringifyModifiersInLocal(field: GPField) = ""
            override fun stringifyModifiersInArg(field: GPField) = ""
            override fun stringifyModifiersInLocalsStruct() = ""
        }.apply {
            println(stringify())
        }
    }

}

