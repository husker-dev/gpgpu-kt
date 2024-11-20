@file:OptIn(ExperimentalKotlinGradlePluginApi::class)
import org.gradle.nativeplatform.platform.internal.DefaultNativePlatform
import org.jetbrains.kotlin.gradle.ExperimentalKotlinGradlePluginApi
import org.jetbrains.kotlin.gradle.dsl.JvmTarget
import org.jetbrains.kotlin.gradle.dsl.KotlinJsCompile
import org.jetbrains.kotlin.gradle.plugin.mpp.KotlinNativeTarget


plugins {
    kotlin("multiplatform") version "2.0.21"
    id("com.android.library") version "8.5.2"
    id("org.jetbrains.kotlinx.benchmark") version "0.4.11"

    id("maven-publish")
    id("signing")
    id("io.codearte.nexus-staging") version "0.30.0"
}

group = "com.huskerdev"
version = "1.0"

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
            jvmTarget.set(JvmTarget.JVM_11)
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
            jvmTarget.set(JvmTarget.JVM_1_8)
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

            //jvmMain.get().dependsOn(this)
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
            url = project.uri("https://s01.oss.sonatype.org/service/local/staging/deploy/maven2/")
            credentials {
                username = project.properties["ossrhUsername"].toString()
                password = project.properties["ossrhPassword"].toString()
            }
        }
    }
}
project.signing {
    if(project.hasProperty("ossrhUsername"))
        sign(publishing.publications)
}
nexusStaging {
    packageGroup = group.toString()
    serverUrl = "https://s01.oss.sonatype.org/service/local/"
    username = project.properties["ossrhUsername"].toString()
    password = project.properties["ossrhPassword"].toString()
}


fun KotlinNativeTarget.linkCUDA(){
    if(DefaultNativePlatform.getCurrentOperatingSystem().isMacOsX)
        return

    compilations.getByName("main"){
        cinterops {
            val dir = System.getenv()["CUDA_PATH"]
            val staticLibExt = when{
                DefaultNativePlatform.getCurrentOperatingSystem().isWindows -> "lib"
                else -> "a"
            }

            val cuda by creating {
                includeDirs("$dir/include")
                extraOpts(
                    "-libraryPath", "${dir}/lib/x64",
                    "-staticLibrary", "cuda.$staticLibExt"
                )
            }
            val nvrtc by creating {
                includeDirs("$dir/include")
                extraOpts(
                    "-libraryPath", "${dir}/lib/x64",
                    "-staticLibrary", "nvrtc.$staticLibExt"
                )
            }
        }
    }
}