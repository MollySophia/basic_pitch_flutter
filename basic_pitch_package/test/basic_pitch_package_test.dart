import 'dart:ffi';

import 'package:flutter_test/flutter_test.dart';

import 'package:basic_pitch_package/basic_pitch_package.dart';
import 'dart:typed_data';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  test('Test', () async {
    final basicPitchInstance = BasicPitch();
    basicPitchInstance.init();
    Float32List data = Float32List(43844);
    final result = await basicPitchInstance.predict(data, false);
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 10; j++) {
        print("${result[i][0][0][j]}");
      }
      print("==============");
    }
    basicPitchInstance.release();
  });
}
