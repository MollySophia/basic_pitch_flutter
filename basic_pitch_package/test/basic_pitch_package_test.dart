import 'dart:ffi';

import 'package:flutter_test/flutter_test.dart';

import 'package:basic_pitch_package/basic_pitch_package.dart';
import 'dart:typed_data';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  test('Test', () {
    final basic_pitch = Basic_Pitch();
    basic_pitch.init();
    Float32List data = Float32List(43844);
    basic_pitch.predict(data, false);
  });
}
