//import org.opencv.core.Core
//import org.opencv.core.CvType
//import org.opencv.core.Mat
//import org.opencv.imgcodecs.Imgcodecs
//import org.json.simple.JSONObject
//import kotlin.math.exp
//import kotlin.math.floor
//
//fun PngtoAudio(args: Array<String>) {
//    System.loadLibrary(Core.NATIVE_LIBRARY_NAME)
//
//    val filename = "path/to/input/image.png"
//    val scalemin = 0 // Set this value according to your requirements
//    val scalemax = 5 // Set this value according to your requirements
//
//    val (outimg, lwinfo) = PNG2LogSpect(filename, scalemin, scalemax)
//    val data = inv_log(outimg)
//    val magD = inv_log(data)
//    val fftsize = 2 * (magD.size - 1) // Infer fftsize from the number of fft bins
//
//    val y_out = spsi(magD, fftsize, 256) // Adjust parameters accordingly
//    val scalefactor = y_out.map { it.map { it.absoluteValue }.maxOrNull() }.maxOrNull() ?: 1.0
//
//    println("Scaling peak sample, $scalefactor to 1")
//    y_out.forEachIndexed { i, row ->
//        y_out[i] = row.map { it / scalefactor }.toDoubleArray()
//    }
//
//    val outputFilename = "path/to/output/audio.wav" // Specify output audio filename
//    saveAudio(y_out, outputFilename, 44100) // Adjust sample rate accordingly
//    println("COMPLETE")
//}
//
//fun PNG2LogSpect(fname: String, scalemin: Int, scalemax: Int): Pair<Array<DoubleArray>, JSONObject> {
//    val img = Imgcodecs.imread(fname, Imgcodecs.IMREAD_GRAYSCALE)
//    val lwinfo = mutableMapOf<String, String>()
//
//    val minx = lwinfo["$scalemin"]?.toFloat()
//    val maxx = lwinfo["$scalemax"]?.toFloat()
//
////    val minx = lwinfo.getDouble("scaleMin").toFloat()
////    val maxx = lwinfo.getDouble("scaleMax").toFloat()
//
//    val outimg = Mat()
//    img.convertTo(outimg, CvType.CV_32F, 1.0 / 255.0, 0.0)
//    Core.flip(outimg, outimg, 0)
//
//    val lwinfoResult = JSONObject().apply {
//        put("fileMin", outimg.min())
//        put("fileMax", outimg.max())
//    }
//
//    return Pair(outimg.toArray(), lwinfoResult)
//}
//
//fun saveAudio(data: Array<DoubleArray>, filename: String, sr: Int) {
//    val scale = Short.MAX_VALUE.toDouble()
//    val samples = data.flatten().map { (it * scale).toShort() }.toShortArray()
//    val outputAudio = org.jaudiolibs.audioops.IOAudioWriter.createWriter(filename, sr, 1, true)
//    outputAudio.use { writer ->
//        writer.write(samples)
//    }
//}
//
//fun inv_log(img: Array<DoubleArray>): Array<DoubleArray> {
//    return img.map { row ->
//        row.map { exp(it) - 1.0 }.toDoubleArray()
//    }.toTypedArray()
//}
//
//fun Mat.min(): Float {
//    val buffer = FloatArray(width() * height())
//    get(0, 0, buffer)
//    return buffer.minOrNull() ?: 0f
//}
//
//fun Mat.max(): Float {
//    val buffer = FloatArray(width() * height())
//    get(0, 0, buffer)
//    return buffer.maxOrNull() ?: 0f
//}
//
//fun Mat.toArray(): Array<DoubleArray> {
//    val array = Array(height()) { DoubleArray(width()) }
//    for (i in 0 until height()) {
//        for (j in 0 until width()) {
//            array[i][j] = get(i, j)[0].toDouble()
//        }
//    }
//    return array
//}
