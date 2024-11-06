import 'package:flutter_test/flutter_test.dart';

import 'package:basic_pitch_package/basic_pitch_package.dart';
import 'dart:io';
import 'dart:core';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  test('Test', () async {
    final basicPitchInstance = BasicPitch();
    basicPitchInstance.init();

    final audioData = await File('assets/testdata/test_audio.wav').readAsBytes();

    final noteEvents = await basicPitchInstance.predictBytes(audioData);
    for (var note in noteEvents) {
      print("start: ${note['start']}, end: ${note['end']}, pitch: ${note['pitch']}");
    }
    basicPitchInstance.release();
  });
}
