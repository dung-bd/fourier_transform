import com.jlibrosa.audio.JLibrosa
import nu.pattern.OpenCV
import org.apache.commons.math3.complex.Complex
import org.apache.commons.math3.transform.DftNormalization
import org.apache.commons.math3.transform.FastFourierTransformer
import org.apache.commons.math3.transform.TransformType
import org.jtransforms.fft.DoubleFFT_1D
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.imgcodecs.Imgcodecs
import java.io.ByteArrayInputStream
import java.io.File
import java.lang.Math.*
import java.nio.ByteBuffer
import java.nio.ByteOrder
import javax.sound.sampled.AudioFileFormat
import javax.sound.sampled.AudioFormat
import javax.sound.sampled.AudioSystem

//fun main(args: Array<String>) {
//    OpenCV.loadLocally()
//    val filename = "C:\\Users\\Admin\\Downloads\\Telegram Desktop\\Test\\Test\\src\\main\\kotlin\\1712290038679.wav"
//    val fileWithoutdot = filename.substringBeforeLast('.')
//    val data = wav2stft(filename, 44100, 1024, 256, -1)
//    println(data?.size)
//    val value = data?.logScale()
//    val vall = findMinMax(value!!)
//    println("Minimum value: ${vall.first}, Maximum value: ${vall.second}")
//    val minScale = (Math.floor(vall.first)).toInt()
//    val maxScale = (Math.floor(vall.second)).toInt()
//    println("Using [${minScale},${maxScale}] -> [0, 255] for png conversion")
//    val pnginfo = mutableMapOf<String, String>()
//    pnginfo["datasetMin"] = vall.first.toString()
//    pnginfo["datasetMax"] = vall.second.toString()
//    pnginfo["scaleMin"] = minScale.toString()
//    pnginfo["scaleMax"] = maxScale.toString()
//    logSpect2PNG(value!!,"${fileWithoutdot}.png" ,pnginfo)
//
//    println("0: ${fileWithoutdot}")
//    println("COMPLETE")
//}
//
//fun wav2stft(fname: String, srate: Int, fftSize: Int, fftHop: Int, dur: Int): Array<Array<Complex>>? {
//    return try {
//        val jLibrosa = JLibrosa()
//        val audiodata = jLibrosa.loadAndRead(fname, srate, dur?.toInt() ?: -1)
//
//        if (srate == -1) {
//            println("Using native samplerate of ${jLibrosa.sampleRate}")
//        }
//
//        val S = jLibrosa.generateSTFTFeaturesWithPadOption(audiodata, jLibrosa.sampleRate, 0, fftSize, 0, fftHop, false)
//        S.map { it.map { Complex(it.real.absoluteValue, it.imaginary.absoluteValue) }.toTypedArray() }.toTypedArray()
//    } catch (e: IOException) {
//        println("Can not read $fname")
//        println(e.message)
//        null
//    } catch (e: WavFileException) {
//        println("Can not read $fname")
//        println(e.message)
//        null
//    } catch (e: FileFormatNotSupportedException) {
//        println("Can not read $fname")
//        println(e.message)
//        null
//    }
//
//}
//
//fun Array<Array<Complex>>.logScale(): Array<DoubleArray> {
//    return map { row ->
//        row.map { complex ->
//            ln1p(complex.abs())
//        }.toDoubleArray()
//    }.toTypedArray()
//}
//
//private fun ln1p(x: Double): Double {
//    return if (x > 0) ln(x + 1.0) else 0.0
//}
//
//fun findMinMax(img: Array<DoubleArray>): Pair<Double, Double> {
//    var minVal = Double.POSITIVE_INFINITY
//    var maxVal = Double.NEGATIVE_INFINITY
//
//    for (row in img) {
//        for (pixel in row) {
//            minVal = FastMath.min(minVal, pixel)
//            maxVal = FastMath.max(maxVal, pixel)
//        }
//    }
//
//    return Pair(minVal, maxVal)
//}
//
//fun logSpect2PNG(outimg: Array<DoubleArray>, fname: String, pnginfo: MutableMap<String, String>?) {
//    val pnginfo = pnginfo ?: mutableMapOf()
//    var minVal = outimg[0][0]
//    var maxVal = outimg[0][0]
//    for (row in outimg) {
//        for (value in row) {
//            if (value < minVal) minVal = value
//            if (value > maxVal) maxVal = value
//        }
//    }
//    pnginfo["fileMin"] = minVal.toString()
//    pnginfo["fileMax"] = maxVal.toString()
//
//    val scaleMin = pnginfo["scaleMin"]!!.toDouble()
//    val shift = (pnginfo["scaleMax"]!!.toDouble() - scaleMin)
//
//    val mat = Mat(outimg.size, outimg[0].size, CvType.CV_64FC1)
//    for (i in outimg.indices) {
//        for (j in outimg[i].indices) {
//            mat.put(i, j, (outimg[i][j] - scaleMin) / shift * 255.0)
//        }
//    }
//
//    val savimg = Mat()
//    Core.flip(mat, savimg, 0)
//
//    val pngimg = Mat()
//    savimg.convertTo(pngimg, CvType.CV_8UC1)
//    Imgcodecs.imwrite(fname, pngimg)
//}

fun main(args: Array<String>) {
    OpenCV.loadLocally()
    val jLibrosa = JLibrosa()
    val fname =
        "C:\\Users\\Admin\\Downloads\\Telegram Desktop\\Test\\Test\\src\\main\\kotlin\\ChungTaCuaHienTaijava.png"
    val scalemin = 0.0
    val scalemax = 5.0
    val (D, _) = PNG2LogSpect(fname, scalemin, scalemax)
    println(D.height())
    val fftSize = 2 * (D.size().height - 1)
    println(fftSize)
    val magD = invLog(D)
    var yOut = spsi(magD, fftSize.toInt(), 256)
    for (i in 0..1024){
        println(yOut[i])
    }
    var x = jLibrosa.generateSTFTFeaturesWithPadOption(yOut, jLibrosa.sampleRate, 0, fftSize.toInt(), 0, 256, true);
    println("x: ${x[0][0]}")
    var p = angle(x)
    print("${magD.size()}")
    println(p.size)
    for (i in 0 until  50) {
        var S = calculateS(magD, p)
        yOut = jLibrosa.generateInvSTFTFeaturesWithPadOption(S, 44100, 0, fftSize.toInt(), 0, 256, -1, true)
        p = angle(jLibrosa.generateSTFTFeaturesWithPadOption(yOut, 44100, 0,  true))
        println("end $$i")
    }
    for (i in 0..10){
        println("yout: ${yOut[i]}")
    }
    val scaleFactor = calculateScaleFactor(yOut)
    println("scaling peak sample $scaleFactor to 1")
    yOut.forEachIndexed { i, value ->
        yOut[i] = value / scaleFactor
    }
    val outputFilename = "C:\\Users\\Admin\\Downloads\\Telegram Desktop\\Test\\Test\\src\\main\\kotlin\\ChungTaCuaHienTaijava.wav" // Specify output audio filename
    val new  = yOut.map { it -> it.toDouble() }.toDoubleArray()
    saveAudio(new, outputFilename, 44100) // Adjust sample rate accordingly
    println("COMPLETE")

}

fun saveAudio(data: DoubleArray, filename: String, sr: Int) {
    val scale = Short.MAX_VALUE.toDouble()
    val samples = mutableListOf<Short>()
    for (value in data) {
        samples.add((value * scale).toInt().toShort())
    }

    val format = AudioFormat(sr.toFloat(), 16, 1, true, false)
    val audioData = ByteBuffer.allocate(samples.size * 2).order(ByteOrder.LITTLE_ENDIAN)
    for (sample in samples) {
        audioData.putShort(sample)
    }
    val byteArray = audioData.array()

    val byteArrayInputStream = ByteArrayInputStream(byteArray)
    val audioInputStream = javax.sound.sampled.AudioInputStream(byteArrayInputStream, format, byteArray.size.toLong())
    val wavFile = File(filename)
    AudioSystem.write(audioInputStream, AudioFileFormat.Type.WAVE, wavFile)
}

fun calculateScaleFactor(yOut: FloatArray): Float {
    var maxAbsValue = Float.MIN_VALUE

    for (i in yOut.indices) {
        val absValue = kotlin.math.abs(yOut[i])
        if (absValue > maxAbsValue) {
            maxAbsValue = absValue
        }
    }

    return maxAbsValue
}

fun PNG2LogSpect(fileName: String, scaleMin: Double, scaleMax: Double): Pair<Mat, Map<String, Any>> {
    // Đọc ảnh từ file
    val img = Imgcodecs.imread(fileName, Imgcodecs.IMREAD_GRAYSCALE)
// Tính toán giá trị min và max theo yêu cầu
    val minx = 0.0
    val maxx = 5.0
    println(img[0, 0].get(0))

    val outImg = Mat()
    img.convertTo(outImg, CvType.CV_32F)
    println(outImg.size())
    println(outImg)
    Core.divide(outImg, Scalar(255.0), outImg)
    Core.multiply(outImg, Scalar(maxx - minx), outImg)
    Core.add(outImg, Scalar(minx), outImg)

    // Scale ảnh theo min và max đã cho
    println(outImg[0, 0].get(0))
    // Lật ảnh theo trục dọc
    Core.flip(outImg, outImg, 0)

    println(outImg[0, 0].get(0))

    // Tạo map lwinfo và thêm các thông tin cần thiết
    val lwinfo = mutableMapOf<String, Any>()
    lwinfo["scaleMin"] = 0
    lwinfo["scaleMax"] = 5

    // Trả về cặp giá trị (ảnh đã lật, thông tin lwinfo)
    System.gc();
    System.runFinalization();
    return Pair(outImg, lwinfo)

}

fun invLog(img: Mat): Mat {
    val expImg = Mat()
    Core.exp(img, expImg)
    val result = Mat()
    Core.subtract(expImg, Scalar.all(1.0), result)
    System.gc();
    System.runFinalization();
    return result
}

fun ifft1(input: Array<Complex>): DoubleArray {
    val transformer = FastFourierTransformer(DftNormalization.STANDARD)

    val complexOutput = transformer.transform(input, TransformType.INVERSE)
    return complexOutput.map { it.real }.toDoubleArray()
}

private fun DoubleArray.toComplexArray(): Array<Complex> {
    val complexArray = Array(size / 2) { Complex(this[it * 2], this[it * 2 + 1]) }
    return complexArray
}

private fun Array<Complex>.toDoubleArray(): DoubleArray {
    val doubleArray = DoubleArray(size * 2)
    for (i in indices) {
        doubleArray[i * 2] = this[i].real
        doubleArray[i * 2 + 1] = this[i].imaginary
    }
    return doubleArray
}

fun hamming(n: Int, symmetric: Boolean = false): DoubleArray {
    val window = DoubleArray(n)
    val factor = if (symmetric) PI / (n - 1) else 2 * PI / n
    for (i in 0 until n) {
        window[i] = 0.54 - 0.46 * Math.cos(factor * i)
    }
    return window
}

fun spsi(msgram: Mat, fftSize: Int, hopLength: Int): FloatArray {
    val numBins = msgram.rows()
    val numFrames = msgram.cols()
    val yOut = FloatArray(numFrames * hopLength + fftSize - hopLength)
    val mPhase = DoubleArray(numBins)
    val mWin = hamming(fftSize, true)

    for (i in 0 until numFrames) {
        println(i)
        val mMag = DoubleArray(numBins)
        for (j in 0 until numBins) {
            mMag[j] = msgram.get(j, i)[0]
        }
        if (i == 51939){
            for (u in 0..512){
                println("mmag${mMag[u]}")
            }
        }

        for (j in 1 until numBins - 1) {
            if (mMag[j] > mMag[j - 1] && mMag[j] > mMag[j + 1]) {
                val alpha = mMag[j - 1]
                val beta = mMag[j]
                val gamma = mMag[j + 1]
                val denom = alpha - 2 * beta + gamma
                val p = if (denom != 0.0) 0.5 * (alpha - gamma) / denom else 0.0
                val phaseRate = 2 * PI * (j + p) / fftSize
                mPhase[j] += hopLength * phaseRate

                val peakPhase = mPhase[j]


                if (p > 0) {
                    mPhase[j + 1] = peakPhase + PI

                    var bin = j - 1
                    while (bin > 0 && mMag[bin] < mMag[bin + 1]) {
                        mPhase[bin] = peakPhase + PI
                        bin--
                    }

                    bin = j + 2
                    while (bin < numBins && mMag[bin] < mMag[bin - 1]) {
                        mPhase[bin] = peakPhase
                        bin++
                    }
                } else {
                    mPhase[j - 1] = peakPhase + PI

                    var bin = j + 1
                    while (bin < numBins && mMag[bin] < mMag[bin - 1]) {
                        mPhase[bin] = peakPhase + PI
                        bin++
                    }

                    bin = j - 2
                    while (bin > 0 && mMag[bin] < mMag[bin + 1]) {
                        mPhase[bin] = peakPhase
                        bin--
                    }
                }
            }
        }

        val magphase = Array(numBins) { Complex(0.0, 0.0) }
        for (j in 0 until numBins) {
            val re = Complex(mMag[j], 0.0)
            val phase = Complex(Math.cos(mPhase[j]), Math.sin(mPhase[j]))
            magphase[j] = re.multiply(phase)
        }
        magphase[0] = Complex(0.0, 0.0)
        magphase[numBins - 1] = Complex(0.0, 0.0)
        //magphase hiện tại đã đúng

//        println("magphase: ${magphase[109]}")

        val magphaseSlice = magphase.sliceArray(1 until numBins - 1)
        val conjugatedSlice = magphaseSlice.map { it.conjugate() }
        val flippedConjugatedSlice = conjugatedSlice.reversed().toTypedArray()
        val mRecon = magphase + flippedConjugatedSlice

//Sai trong biến đổi ifft

        var reconSignal = ifft1(mRecon)

        for (j in 0 until fftSize) {
            reconSignal[j] *= mWin[j]
        }

        for (i in 0 until numFrames) {
            val startIndex = i * hopLength
            val endIndex = startIndex + fftSize
            for (j in startIndex until endIndex) {
                yOut[j] += reconSignal[j - startIndex].toFloat()
            }
        }
    }
//    for (i in 0..10){
//        println("y_out")
//        println(yOut[i])
//    }
    return yOut
}

fun angle(complexArray: Array<Array<Complex>>): FloatArray {
    val angles = FloatArray(complexArray.sumOf { it.size })
    var index = 0
    for (row in complexArray) {
        for (complex in row) {
            angles[index++] = complex.argument.toFloat()
        }
    }
    return angles
}

fun calculateS(magD: Mat, p: FloatArray): Array<Array<Complex>> {
    val numBins = magD.rows()
    val numFrames = magD.cols()
    val S = Array(numBins) { Array(numFrames) { Complex(0.0, 0.0) } }
    for (i in 0 until numBins) {
        for (j in 0 until numFrames) {
            val magnitude = magD.get(i, j)[0]
            val phase = p[i]
            val complexMagnitude = Complex(magnitude, 0.0)
            val complexPhase = Complex(0.0, phase.toDouble())
            val expPhase = complexPhase.exp()
            val complexValue = complexMagnitude.multiply(expPhase)
            S[i][j] = complexValue
        }
    }

    return S
}




