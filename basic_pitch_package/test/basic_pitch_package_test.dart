import 'dart:ffi';

import 'package:flutter_test/flutter_test.dart';

import 'package:basic_pitch_package/basic_pitch_package.dart';
import 'dart:typed_data';
import 'dart:io';
import 'dart:core';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  test('Test', () async {
    final basicPitchInstance = BasicPitch();
    basicPitchInstance.init();

    final windowedAudioData = await basicPitchInstance.loadAudioWithPreProcess('assets/testdata/test_audio.wav');
    for (int i = 0; i < windowedAudioData.length; i++) {
      if (windowedAudioData[i].length != BasicPitch.windowedAudioDataLength) {
        throw ArgumentError('Windowed audio data length must be 43844');
      }

      final expected = File('assets/testdata/test_input_$i.bin').readAsBytesSync();
      final expectedData = Float32List.sublistView(Uint8List.fromList(expected));
      final data = Float32List.fromList(windowedAudioData[i]);
      for (int j = 0; j < BasicPitch.windowedAudioDataLength; j++) {
        if ((expectedData[j] - data[j]).abs() > 1e-5) {
          throw ArgumentError('Data mismatch at index $j');
        }
      }
      print("Windowed audio data $i matches expected data");
      final inferenceResult = await basicPitchInstance.inference(data, false);
      // test inference result here
    }
    basicPitchInstance.release();
  });
}
