library basic_pitch_package;
import 'dart:ffi';
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:onnxruntime/onnxruntime.dart';

class BasicPitch {
  late OrtSession _session;

  void init() async {
    OrtEnv.instance.init();
    final sessionOptions = OrtSessionOptions();
    const modelPath = 'assets/models/nmp.onnx';
    final rawAssetFile = await rootBundle.load(modelPath);
    final bytes = rawAssetFile.buffer.asUint8List();
    _session = OrtSession.fromBuffer(bytes, sessionOptions);
  }

  Future<List<List<List<List<double>>>>> predict(Float32List data, bool asyncRun) async {
    if (data.length != 43844) {
      throw ArgumentError('Input data length must be 43844');
    }
    final shape = [1, 43844, 1];
    final inputOrt = OrtValueTensor.createTensorWithDataList(data, shape);
    final inputs = {'serving_default_input_2:0': inputOrt};
    final runOptions = OrtRunOptions();
    final List<OrtValue?>? outputs;
    if (asyncRun) {
      outputs = await _session.runAsync(runOptions, inputs);
    } else {
      outputs = _session.run(runOptions, inputs);
    }
    final List<List<List<List<double>>>> result = [];
    for (final output in outputs!) {
      final outputData = output?.value as List<List<List<double>>>;
      result.add(outputData);
    }
    inputOrt.release();
    runOptions.release();
    return result;
  }

  void release() {
    _session.release();
  }
}
