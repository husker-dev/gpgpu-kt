import io.gitee.pkmer.enums.PublishingType
import org.gradle.nativeplatform.platform.internal.DefaultNativePlatform
import org.jetbrains.kotlin.gradle.dsl.JvmTarget
import org.jetbrains.kotlin.gradle.dsl.KotlinJsCompile
import org.jetbrains.kotlin.gradle.plugin.mpp.KotlinNativeTarget


plugins {
    kotlin("multiplatform") version "2.1.0"
    id("com.android.library") version "8.5.2"
    id("org.jetbrains.kotlinx.benchmark") version "0.4.11"
    id("org.jetbrains.kotlinx.atomicfu") version "0.27.0"

    id("maven-publish")
    id("signing")
    id("io.gitee.pkmer.pkmerboot-central-publisher") version "1.1.1"
}

group = "com.huskerdev"
version = "1.0.7"

repositories {
    google()
    mavenCentral()
}

kotlin {
    compilerOptions {
        freeCompilerArgs.add("-Xexpect-actual-classes")
    }

    jvm {
        compilerOptions {
            jvmTarget = JvmTarget.JVM_11
        }
        compilations.create("benchmark") {
            associateWith(this@jvm.compilations.getByName("main"))
        }
    }
    js {
        browser()
        binaries.executable()
    }
    macosArm64()
    macosX64()
    iosX64()
    iosArm64()

    linuxArm64 {
        linkCUDA()
    }
    linuxX64 {
        linkCUDA()
    }
    mingwX64 {
        linkCUDA()
    }

    androidTarget {
        publishLibraryVariants("release")
        compilerOptions {
            jvmTarget = JvmTarget.JVM_1_8
        }
    }

    sourceSets {
        val macosMain by creating {
            macosX64Main.get().dependsOn(this)
            macosArm64Main.get().dependsOn(this)
        }

        val iosMain by creating {
            iosX64Main.get().dependsOn(this)
            iosArm64Main.get().dependsOn(this)
        }

        val windowsMain by creating {
            mingwX64Main.get().dependsOn(this)
        }

        val linuxMain by creating {
            linuxX64Main.get().dependsOn(this)
            linuxArm64Main.get().dependsOn(this)
        }

        val commonOpenCL by creating {
            dependsOn(commonMain.get())

            jvmMain.get().dependsOn(this)
            androidMain.get().dependsOn(this)
        }

        val commonCUDA by creating {
            dependsOn(commonMain.get())

            jvmMain.get().dependsOn(this)
        }

        val commonCUDANative by creating {
            dependsOn(commonCUDA)

            windowsMain.dependsOn(this)
            linuxMain.dependsOn(this)
        }

        val commonMetal by creating {
            dependsOn(commonMain.get())

            jvmMain.get().dependsOn(this)
        }

        val commonMetalNative by creating {
            dependsOn(commonMetal)

            macosMain.dependsOn(this)
            iosMain.dependsOn(this)
        }

        commonTest.dependencies {
            implementation(kotlin("test"))
        }

        jvmMain {
            dependencies {
                // JOCL (OpenCL)
                api("org.jocl:jocl:2.0.4")

                // JCuda (Cuda)
                api("org.jcuda:jcuda:12.0.0") {
                    isTransitive = false
                }
                api("org.jcuda:jcuda-natives:12.0.0:windows-x86_64")
                api("org.jcuda:jcuda-natives:12.0.0:linux-x86_64")

                // Metal
                api("ca.weblite:java-objc-bridge:1.2")
                api("net.java.dev.jna:jna:5.15.0")
            }
        }

        val jvmBenchmark by getting {
            dependencies {
                implementation("org.jetbrains.kotlinx:kotlinx-benchmark-runtime:0.4.11")
            }
        }
    }
}

android {
    namespace = "com.huskerdev.gpkt"
    compileSdk = 34
    ndkVersion = "19.2.5345600"

    defaultConfig {
        minSdk = 19
    }

    externalNativeBuild {
        cmake {
            path("src/androidMain/cpp/CMakeLists.txt")
        }
    }
}

benchmark {
    targets {
        register("jvmBenchmark")
    }
}

tasks.withType(KotlinJsCompile::class.java).configureEach {
    compilerOptions {
        target = "es2015"
    }
}

publishing {
    publications {
        withType<MavenPublication> {
            artifact(tasks.register("${name}JavadocJar", Jar::class) {
                archiveClassifier = "javadoc"
                archiveAppendix = this@withType.name
            })
            pom {
                name = "gpgpu-kt"
                description = "Cross-platform general-purpose computing Kotlin Multiplatform library"
                url = "https://github.com/husker-dev/gpgpu-kt"

                licenses {
                    license {
                        name = "The Apache License, Version 2.0"
                        url = "http://www.apache.org/licenses/LICENSE-2.0.txt"
                    }
                }
                developers {
                    developer {
                        id = "husker-dev"
                        name = "Nikita Shtengauer"
                        email = "redfancoestar@gmail.com"
                    }
                }
                scm {
                    connection = "https://github.com/husker-dev/gpgpu-kt.git"
                    developerConnection = "https://github.com/husker-dev/gpgpu-kt.git"
                    url = "https://github.com/husker-dev/gpgpu-kt"
                }
            }
        }
    }
    repositories {
        maven {
            name = "Local"
            url = uri(layout.buildDirectory.dir("repos/bundles"))
        }
    }
}

signing {
    setRequired {
        gradle.taskGraph.allTasks.any { it is PublishToMavenRepository }
    }
    if(hasProperty("ossrhUsername"))
        sign(publishing.publications)
}

pkmerBoot {
    sonatypeMavenCentral{
        stagingRepository = layout.buildDirectory.dir("repos/bundles")

        username = properties["ossrhUsername"].toString()
        password = properties["ossrhPassword"].toString()

        publishingType = PublishingType.AUTOMATIC
    }
}


fun KotlinNativeTarget.linkCUDA(){
    if(DefaultNativePlatform.getCurrentOperatingSystem().isMacOsX)
        return

    compilations.getByName("main"){
        cinterops {
            val dir = System.getenv()["CUDA_PATH"] ?: "/usr/local/cuda"
            val libFolder = when{
                DefaultNativePlatform.getCurrentOperatingSystem().isWindows -> "lib/x64"
                else -> "lib64"
            }

            val cuda by creating {
                includeDirs("$dir/include")
                extraOpts(
                    "-libraryPath", "$dir/$libFolder",
                    "-staticLibrary", when{
                        DefaultNativePlatform.getCurrentOperatingSystem().isWindows -> "cuda.lib"
                        else -> "stubs/libcuda.so"
                    }
                )
            }
            val nvrtc by creating {
                includeDirs("$dir/include")
                extraOpts(
                    "-libraryPath", "$dir/$libFolder",
                    "-staticLibrary", when{
                        DefaultNativePlatform.getCurrentOperatingSystem().isWindows -> "nvrtc.lib"
                        else -> "libnvrtc.so"
                    }
                )
            }
        }
    }
}