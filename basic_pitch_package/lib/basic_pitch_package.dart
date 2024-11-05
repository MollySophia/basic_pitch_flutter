library basic_pitch_package;
import 'dart:typed_data';
import 'dart:io';
import 'dart:convert';
import 'package:flutter/services.dart';
import 'package:onnxruntime/onnxruntime.dart';

class BasicPitch {
  late OrtSession _session;
  static const windowedAudioDataLength = 43844;
  static const expectedAudioSampleRate = 22050;

  void init() async {
    OrtEnv.instance.init();
    final sessionOptions = OrtSessionOptions();
    const modelPath = 'assets/models/nmp.onnx';
    final rawAssetFile = await rootBundle.load(modelPath);
    final bytes = rawAssetFile.buffer.asUint8List();
    _session = OrtSession.fromBuffer(bytes, sessionOptions);
  }

  Future<List<List<List<List<double>>>>> inference(Float32List data, bool asyncRun) async {
    if (data.length != windowedAudioDataLength) {
      throw ArgumentError('Input data length must be $windowedAudioDataLength');
    }
    final shape = [1, windowedAudioDataLength, 1];
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

  Future<Map<String, Object>> loadAudioMono(String filePath) async {
    final file = File(filePath);
    final bytes = await file.readAsBytes();
    final byteData = ByteData.sublistView(bytes);

    if (utf8.decode(bytes.sublist(0, 4)) != 'RIFF' || utf8.decode(bytes.sublist(8, 12)) != 'WAVE') {
      throw const FormatException('Unsupported file format');
    }

    int offset = 12;
    ByteData fmt = ByteData(0);
    while (offset < bytes.length) {
      final chunkHeader = bytes.sublist(offset, offset + 8);
      final chunkSize = byteData.getUint32(offset + 4, Endian.little);
      offset += 8;

      if (utf8.decode(chunkHeader.sublist(0, 4)) == 'fmt ') {
        fmt = ByteData.sublistView(bytes, offset, offset + chunkSize);
        offset += chunkSize;
        break;
      } else if (utf8.decode(chunkHeader.sublist(0, 4)) == 'JUNK') {
        offset += chunkSize;
      } else {
        throw FormatException('Unsupported chunk type: ${utf8.decode(chunkHeader.sublist(0, 4))}');
      }
    }

    final audioFormat = fmt.getUint16(0, Endian.little);
    final numChannels = fmt.getUint16(2, Endian.little);
    final sampleRate = fmt.getUint32(4, Endian.little);
    final byteRate = fmt.getUint32(8, Endian.little);
    final blockAlign = fmt.getUint16(12, Endian.little);
    final bitsPerSample = fmt.getUint16(14, Endian.little);

    print('audio_format: $audioFormat');
    print('num_channels: $numChannels');
    print('sample_rate: $sampleRate');
    print('byte_rate: $byteRate');
    print('block_align: $blockAlign');
    print('bits_per_sample: $bitsPerSample');

    final dataHeader = bytes.sublist(offset, offset + 8);
    if (utf8.decode(dataHeader.sublist(0, 4)) != 'data') {
      throw const FormatException('Unsupported data header');
    }
    final dataSize = byteData.getUint32(offset + 4, Endian.little);
    offset += 8;

    final rawData = bytes.sublist(offset, offset + dataSize);
    List<double> data;
    if (bitsPerSample == 16) {
      final int16Data = Int16List.sublistView(Uint8List.fromList(rawData));
      data = int16Data.map((e) => e / 32768.0).toList();
    } else if (bitsPerSample == 32) {
      final int32Data = Int32List.sublistView(Uint8List.fromList(rawData));
      data = int32Data.map((e) => e / 2147483648.0).toList();
    } else {
      throw FormatException('Unsupported bit depth: $bitsPerSample');
    }

    if (numChannels == 2) {
      final List<double> monoData = [];
      for (int i = 0; i < data.length; i += 2) {
        monoData.add((data[i] + data[i + 1]) / 2);
      }
      data = monoData;
    }

    return {'data': data, 'sampleRate': sampleRate};
  }

  Future<List<List<double>>> loadAudioWithPreProcess(String filePath) async {
    final audio_out = await loadAudioMono(filePath);
    final data = audio_out['data'] as List<double>;
    final sampleRate = audio_out['sampleRate'] as int;
    if (sampleRate != expectedAudioSampleRate) {
      // TODO: add resample support
      throw ArgumentError('Unsupported sample rate');
    }

    const windowSize = 43844;
    const overlapLength = 7680;
    final hopLength = windowSize - overlapLength;
    final paddedData = List<double>.filled(overlapLength ~/ 2, 0.0) + data;

    final totalSamples = paddedData.length;
    final numWindows = (totalSamples / hopLength).ceil();
    final windows = <List<double>>[];

    for (int i = 0; i < numWindows; i++) {
      final start = i * hopLength;
      final end = start + windowSize;

      List<double> window;
      if (end > totalSamples) {
        window = paddedData.sublist(start) + List<double>.filled(end - totalSamples, 0.0);
      } else {
        window = paddedData.sublist(start, end);
      }

      windows.add(window);
    }

    return windows;
  }
}
