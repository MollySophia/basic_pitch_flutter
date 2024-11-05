library basic_pitch_package;
import 'package:onnxruntime/onnxruntime.dart';
import 'package:flutter/services.dart';

class Basic_Pitch {
  void init() async {
    OrtEnv.instance.init();
    // final sessionOptions = OrtSessionOptions();
    // const modelPath = 'assets/models/nmp.onnx';
    // final rawAssetFile = await rootBundle.load(modelPath);
    // final bytes = rawAssetFile.buffer.asUint8List();
    // final session = OrtSession.fromBuffer(bytes, sessionOptions!);
  }
}
