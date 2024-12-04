import 'package:flutter_test/flutter_test.dart';

import 'package:basic_pitch_flutter/basic_pitch_flutter.dart';
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

  test('Perf', () async {
    final basicPitchInstance = BasicPitch();
    basicPitchInstance.init();

    final audioData = await File('assets/testdata/test_audio.wav').readAsBytes();

    int totalMilliseconds = 0;
    int iterations = 100;

    for (int i = 0; i < iterations; i++) {
      final stopwatch = Stopwatch()..start();
      await basicPitchInstance.predictBytes(audioData);
      stopwatch.stop();
      totalMilliseconds += stopwatch.elapsedMilliseconds;
    }

    final averageMilliseconds = totalMilliseconds / iterations;
    print('Average performance test executed in ${averageMilliseconds.toStringAsFixed(2)} ms');

    basicPitchInstance.release();
  });
}
