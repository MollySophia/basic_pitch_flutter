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
      final expectedOutputContour = File('assets/testdata/test_output_contour_$i.bin').readAsBytesSync();
      final expectedOutputContourData = Float32List.sublistView(Uint8List.fromList(expectedOutputContour));
      final expectedOutputNote = File('assets/testdata/test_output_note_$i.bin').readAsBytesSync();
      final expectedOutputNoteData = Float32List.sublistView(Uint8List.fromList(expectedOutputNote));
      final expectedOutputOnset = File('assets/testdata/test_output_onset_$i.bin').readAsBytesSync();
      final expectedOutputOnsetData = Float32List.sublistView(Uint8List.fromList(expectedOutputOnset));
      final noteOutput = inferenceResult['note'] as List<List<List<double>>>;
      final onsetOutput = inferenceResult['onset'] as List<List<List<double>>>;
      final contourOutput = inferenceResult['contour'] as List<List<List<double>>>;
      for (int j = 0; j < contourOutput[0].length; j++) {
        for (int k = 0; k < contourOutput[0][j].length; k++) {
          if ((expectedOutputContourData[j * contourOutput[0][j].length + k] - contourOutput[0][j][k]).abs() > 1e-5) {
            throw ArgumentError('Output contour mismatch at index $j, $k');
          }
        }
      }
      print("Output contour $i matches expected data");
      for (int j = 0; j < noteOutput[0].length; j++) {
        for (int k = 0; k < noteOutput[0][j].length; k++) {
          if ((expectedOutputNoteData[j * noteOutput[0][j].length + k] - noteOutput[0][j][k]).abs() > 1e-5) {
            throw ArgumentError('Output note mismatch at index $j, $k');
          }
        }
      }
      print("Output note $i matches expected data");
      for (int j = 0; j < onsetOutput[0].length; j++) {
        for (int k = 0; k < onsetOutput[0][j].length; k++) {
          if ((expectedOutputOnsetData[j * onsetOutput[0][j].length + k] - onsetOutput[0][j][k]).abs() > 1e-5) {
            throw ArgumentError('Output onset mismatch at index $j, $k');
          }
        }
      }
      print("Output onset $i matches expected data\n");
    }
    basicPitchInstance.release();
  });
}
