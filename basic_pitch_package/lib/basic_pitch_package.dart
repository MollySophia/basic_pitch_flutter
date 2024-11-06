library basic_pitch_package;
import 'dart:math';
import 'dart:typed_data';
import 'dart:io';
import 'dart:convert';
import 'package:flutter/services.dart';
import 'package:onnxruntime/onnxruntime.dart';

class BasicPitch {
  late OrtSession _session;
  static const windowedAudioDataLength = 43844;
  static const expectedAudioSampleRate = 22050;
  static const overlappingFrames = 30;
  static const fftHopSize = 256;
  static const midiOffset = 21;
  static const annotationsBaseFreq = 27.5;  // lowest key on a piano
  static const annotationsNSemitones = 88;  // number of piano keys
  static const contoursBinsPerSemitone = 3;

  List<double> _lastAudioDataOverlap = [];

  void init() async {
    OrtEnv.instance.init();
    final sessionOptions = OrtSessionOptions();
    const modelPath = 'assets/models/nmp.onnx';
    final rawAssetFile = await rootBundle.load(modelPath);
    final bytes = rawAssetFile.buffer.asUint8List();
    _session = OrtSession.fromBuffer(bytes, sessionOptions);
  }

  Future<Map<String, Object>> inference(Float32List data, bool asyncRun) async {
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
    final List<List<List<double>>> resultContour = outputs?[2]?.value as List<List<List<double>>>;
    final List<List<List<double>>> resultNote = outputs?[1]?.value as List<List<List<double>>>;
    final List<List<List<double>>> resultOnset = outputs?[0]?.value as List<List<List<double>>>;
    inputOrt.release();
    runOptions.release();
    return {'contour': resultContour, 'note': resultNote, 'onset': resultOnset};
  }

  void release() {
    _session.release();
  }

  Future<Map<String, Object>> loadAudioMonoBytes(Uint8List bytes) async {
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

    // print('audio_format: $audioFormat');
    // print('num_channels: $numChannels');
    // print('sample_rate: $sampleRate');
    // print('byte_rate: $byteRate');
    // print('block_align: $blockAlign');
    // print('bits_per_sample: $bitsPerSample');

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

  Future<Map<String, Object>> loadAudioMono(String filePath) async {
    final file = File(filePath);
    final bytes = await file.readAsBytes();
    return loadAudioMonoBytes(bytes);
  }

  Map<String, Object> preprocess(List<double> data, int sampleRate) {
    if (sampleRate != expectedAudioSampleRate) {
      // TODO: add resample support
      throw ArgumentError('Unsupported sample rate');
    }
    final originalAudioLength = data.length;

    const windowSize = windowedAudioDataLength;
    const overlapLength = overlappingFrames * fftHopSize;
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

    return {'data': windows, 'originalAudioLength': originalAudioLength};
  }

  Future<Map<String, Object>> loadAudioWithPreProcess(String filePath) async {
    final output = await loadAudioMono(filePath);
    final data = output['data'] as List<double>;
    final sampleRate = output['sampleRate'] as int;
    return preprocess(data, sampleRate);
  }

  Future<Map<String, Object>> loadAudioBytesWithPreProcess(Uint8List bytes) async {
    final output = await loadAudioMonoBytes(bytes);
    final data = output['data'] as List<double>;
    final sampleRate = output['sampleRate'] as int;
    return preprocess(data, sampleRate);
  }

  List<Map<String, Object?>> postprocess(List<List<List<double>>> contour,
                           List<List<List<double>>> note,
                           List<List<List<double>>> onset,
                           int originalAudioLength,
                           double onsetThreshold,
                           double frameThreshold,
                           double minimalNoteLength,
                           double minimalFrequency,
                           double maximalFrequency,
                           bool multiplePitchBends,
                           bool melodiaTrick) {
    const nOverlappingFramesSide = overlappingFrames ~/ 2;

    if (nOverlappingFramesSide > 0) {
      contour = contour.map((window) => window.sublist(nOverlappingFramesSide, window.length - nOverlappingFramesSide)).toList();
      note = note.map((window) => window.sublist(nOverlappingFramesSide, window.length - nOverlappingFramesSide)).toList();
      onset = onset.map((window) => window.sublist(nOverlappingFramesSide, window.length - nOverlappingFramesSide)).toList();
    }

    final nOutputFramesOriginal = (originalAudioLength * ((expectedAudioSampleRate / fftHopSize).floor() / expectedAudioSampleRate)).floor();
    final unwrappedContour = contour.expand((window) => window).toList();
    final unwrappedNote = note.expand((window) => window).toList();
    final unwrappedOnset = onset.expand((window) => window).toList();

    final truncatedContour = unwrappedContour.sublist(0, nOutputFramesOriginal);
    final truncatedNote = unwrappedNote.sublist(0, nOutputFramesOriginal);
    final truncatedOnset = unwrappedOnset.sublist(0, nOutputFramesOriginal);
    final minimalNoteLengthFrames = (minimalNoteLength / 1000 * (expectedAudioSampleRate / fftHopSize)).round();

    final noteEvents = decodePolyphonicNotes(truncatedNote, truncatedOnset, onsetThreshold, frameThreshold, minimalNoteLengthFrames, true, maximalFrequency, minimalFrequency, melodiaTrick, 11);
    final noteEventsWithPitchBends = getPitchBends(truncatedContour, noteEvents);
    
    final audioDurationInSeconds = modelFramesToTime(truncatedContour.length);
    final noteEventsWithSeconds = noteEventsWithPitchBends.map((note) {
      final start = audioDurationInSeconds[note['start'] as int];
      final end = audioDurationInSeconds[note['end'] as int];
      final pitch = note['pitch'];
      final amplitude = note['amplitude'];
      final bends = note['bends'];
      return {'start': start, 'end': end, 'pitch': pitch, 'amplitude': amplitude, 'bends': bends};
    }).toList();
    return noteEventsWithSeconds;
  }

  List<List<double>> getInferedOnsets(List<List<double>> onsets, List<List<double>> frames, {int nDiff = 2}) {
    final nFrames = frames.length;
    final nBins = frames[0].length;
    final diffs = List.generate(nDiff, (_) => List.generate(nFrames, (_) => List.filled(nBins, 0.0)));

    for (int n = 1; n <= nDiff; n++) {
      for (int i = n; i < nFrames; i++) {
        for (int j = 0; j < nBins; j++) {
          diffs[n - 1][i][j] = frames[i][j] - frames[i - n][j];
        }
      }
    }

    final frameDiff = List.generate(nFrames, (_) => List.filled(nBins, 0.0));
    for (int i = 0; i < nFrames; i++) {
      for (int j = 0; j < nBins; j++) {
        frameDiff[i][j] = diffs.map((diff) => diff[i][j]).reduce((a, b) => a < b ? a : b);
        if (frameDiff[i][j] < 0) frameDiff[i][j] = 0;
      }
    }

    for (int i = 0; i < nDiff; i++) {
      for (int j = 0; j < nBins; j++) {
        frameDiff[i][j] = 0;
      }
    }

    final maxOnsets = onsets.expand((e) => e).reduce((a, b) => a > b ? a : b);
    final maxFrameDiff = frameDiff.expand((e) => e).reduce((a, b) => a > b ? a : b);

    for (int i = 0; i < nFrames; i++) {
      for (int j = 0; j < nBins; j++) {
        frameDiff[i][j] = maxOnsets * frameDiff[i][j] / maxFrameDiff;
      }
    }

    final maxOnsetsDiff = List.generate(nFrames, (_) => List.filled(nBins, 0.0));
    for (int i = 0; i < nFrames; i++) {
      for (int j = 0; j < nBins; j++) {
        maxOnsetsDiff[i][j] = onsets[i][j] > frameDiff[i][j] ? onsets[i][j] : frameDiff[i][j];
      }
    }

    return maxOnsetsDiff;
  }

  List<Map<String, Object>> decodePolyphonicNotes(
        List<List<double>> frames,
        List<List<double>> onset,
        double onsetThreshold,
        double frameThreshold,
        int minimalNoteLengthFrames,
        bool inferOnsets,
        double maximalFrequency,
        double minimalFrequency,
        bool melodiaTrick,
        int energyTol
  ) {
    final nFrames = frames.length;

    // set frequency outside of [minimalFrequency, maximalFrequency] to 0
    // hz_to_midi: 12 * (log2(freq) - log2(440)) + 69
    final maxFreqIdx = (12 * (log(maximalFrequency / 440) / ln2)).round() + 69 - midiOffset;
    for (var frame in frames) {
      for (var i = maxFreqIdx; i < frame.length; i++) {
        frame[i] = 0;
      }
    }
    for (var onsetFrame in onset) {
      for (var i = maxFreqIdx; i < onsetFrame.length; i++) {
        onsetFrame[i] = 0;
      }
    }

    final minFreqIdx = (12 * (log(minimalFrequency / 440) / ln2)).round() + 69 - midiOffset;
    for (var frame in frames) {
      for (var i = 0; i < minFreqIdx; i++) {
        frame[i] = 0;
      }
    }
    for (var onsetFrame in onset) {
      for (var i = 0; i < minFreqIdx; i++) {
        onsetFrame[i] = 0;
      }
    }

    if (inferOnsets) {
      onset = getInferedOnsets(onset, frames);
    }

    final onsetPeakX = [];
    final onsetPeakY = [];
    for (int j = nFrames - 1; j >= 0; j--) {
      for (int i = 0; i < onset[0].length; i++) {
        if (j != 0 && j != nFrames - 1) {
          if (onset[j][i] > onset[j - 1][i] && onset[j][i] > onset[j + 1][i]) {
            if (onset[j][i] >= onsetThreshold) {
              onsetPeakX.add(j);
              onsetPeakY.add(i);
            }
          }
        }
      }
    }

    final remainingEnergy = List.generate(frames.length, (i) => List<double>.from(frames[i]));

    final noteEvents = <Map<String, Object>>[];
    for (int idx = 0; idx < onsetPeakX.length; idx++) {
      final noteStartIdx = onsetPeakX[idx];
      final freqIdx = onsetPeakY[idx];

      if (noteStartIdx >= nFrames - 1) continue;

      int i = noteStartIdx + 1;
      int k = 0;
      while (i < nFrames - 1 && k < energyTol) {
        if (remainingEnergy[i][freqIdx] < frameThreshold) {
          k++;
        } else {
          k = 0;
        }
        i++;
      }

      i -= k;

      if (i - noteStartIdx <= minimalNoteLengthFrames) continue;

      for (int j = noteStartIdx; j < i; j++) {
        remainingEnergy[j][freqIdx] = 0;
        if (freqIdx < maxFreqIdx) remainingEnergy[j][freqIdx + 1] = 0;
        if (freqIdx > 0) remainingEnergy[j][freqIdx - 1] = 0;
      }

      final amplitude = frames.sublist(noteStartIdx, i).map((frame) => frame[freqIdx]).reduce((a, b) => a + b) / (i - noteStartIdx);
      noteEvents.add({
        'start': noteStartIdx,
        'end': i,
        'pitch': freqIdx + midiOffset,
        'amplitude': amplitude,
      });
    }

    if (melodiaTrick) {
      final energyShape = remainingEnergy.length;

      while (remainingEnergy.expand((e) => e).reduce((a, b) => a > b ? a : b) > frameThreshold) {
        final maxIndex = remainingEnergy.expand((e) => e).toList().indexOf(remainingEnergy.expand((e) => e).reduce((a, b) => a > b ? a : b));
        final iMid = maxIndex ~/ remainingEnergy[0].length;
        final freqIdx = maxIndex % remainingEnergy[0].length;
        remainingEnergy[iMid][freqIdx] = 0;

        // forward pass
        int i = iMid + 1;
        int k = 0;
        while (i < nFrames - 1 && k < energyTol) {
          if (remainingEnergy[i][freqIdx] < frameThreshold) {
            k++;
          } else {
            k = 0;
          }

          remainingEnergy[i][freqIdx] = 0;
          if (freqIdx < maxFreqIdx) {
            remainingEnergy[i][freqIdx + 1] = 0;
          }
          if (freqIdx > 0) {
            remainingEnergy[i][freqIdx - 1] = 0;
          }

          i++;
        }

        final iEnd = i - 1 - k;

        // backward pass
        i = iMid - 1;
        k = 0;
        while (i > 0 && k < energyTol) {
          if (remainingEnergy[i][freqIdx] < frameThreshold) {
            k++;
          } else {
            k = 0;
          }

          remainingEnergy[i][freqIdx] = 0;
          if (freqIdx < maxFreqIdx) {
            remainingEnergy[i][freqIdx + 1] = 0;
          }
          if (freqIdx > 0) {
            remainingEnergy[i][freqIdx - 1] = 0;
          }

          i--;
        }

        final iStart = i + 1 + k;
        assert(iStart >= 0);
        assert(iEnd < nFrames);

        if (iEnd - iStart <= minimalNoteLengthFrames) {
          continue;
        }

        final amplitude = frames.sublist(iStart, iEnd).map((frame) => frame[freqIdx]).reduce((a, b) => a + b) / (iEnd - iStart);
        noteEvents.add({
          'start': iStart,
          'end': iEnd,
          'pitch': freqIdx + midiOffset,
          'amplitude': amplitude,
        });
      }
    }

    return noteEvents;
  }

  List<Map<String, Object>> getPitchBends(
      List<List<double>> contours,
      List<Map<String, Object>> noteEvents,
      {int nBinsTolerance = 25}) {
    final windowLength = nBinsTolerance * 2 + 1;
    final freqGaussian = List.generate(windowLength, (i) => exp(-0.5 * pow((i - nBinsTolerance) / 5, 2)));

    final noteEventsWithPitchBends = <Map<String, Object>>[];
    for (var noteEvent in noteEvents) {
      final startIdx = noteEvent['start'] as int;
      final endIdx = noteEvent['end'] as int;
      final pitchMidi = noteEvent['pitch'] as int;
      final amplitude = noteEvent['amplitude'] as double;

      // midi_pitch_to_contour_bin: 
      // pitch_hz = midi_to_hz(pitchMidi) = 2 ^ ((pitchMidi - 69) / 12) * 440
      // freqIdx = 12 * contoursBinsPerSemitone * log2(pitch_hz / annotationsBaseFreq)
      // = 36 * log2(2 ^ ((pitchMidi - 69) / 12) * 440 / annotationsBaseFreq)
      final freqIdx = (12 * contoursBinsPerSemitone * log(pow(2, (pitchMidi - 69) / 12) * 440 / annotationsBaseFreq) / ln2).round();
      final freqStartIdx = max(freqIdx - nBinsTolerance, 0);
      final freqEndIdx = min(annotationsNSemitones * contoursBinsPerSemitone, freqIdx + nBinsTolerance + 1);

      final pitchBendSubmatrix = List.generate(
        endIdx - startIdx,
        (i) => List.generate(
          freqEndIdx - freqStartIdx,
          (j) => contours[startIdx + i][freqStartIdx + j] *
          freqGaussian[max(0, nBinsTolerance - freqIdx) + j],
        ),
      );

      final pbShift = nBinsTolerance - max(0, nBinsTolerance - freqIdx);
      final bends = pitchBendSubmatrix.map((row) => row.indexOf(row.reduce(max)) - pbShift).toList();

      noteEventsWithPitchBends.add({
        'start': startIdx,
        'end': endIdx,
        'pitch': pitchMidi,
        'amplitude': amplitude,
        'bends': bends,
      });
    }

    return noteEventsWithPitchBends;
  }

  List<double> modelFramesToTime(int nFrames) {
    final originalTimes = List.generate(
      nFrames,
      (i) => i * fftHopSize / expectedAudioSampleRate,
    );

    const audioWindowLength = 2;
    const audioNSamples = expectedAudioSampleRate * audioWindowLength - fftHopSize;
    const annotNFrames = expectedAudioSampleRate ~/ fftHopSize * audioWindowLength;
    final windowNumbers = List.generate(nFrames, (i) => (i / annotNFrames).floor());
    const windowOffset = (fftHopSize / expectedAudioSampleRate) * (annotNFrames - (audioNSamples / fftHopSize)) + 0.0018;

    final times = List.generate(
      nFrames,
      (i) => originalTimes[i] - (windowOffset * windowNumbers[i]),
    );

    return times;
  }

  Future<List<Map<String, Object?>>> predict(List<List<double>> data,
      int originalAudioLength,
      {double onsetThreshold = 0.5,
      double frameThreshold = 0.3,
      double minimalNoteLength = 200,
      double minimalFrequency = 80,
      double maximalFrequency = 1100,
      bool multiplePitchBends = false,
      bool melodiaTrick = true}) async {
    final contour = [];
    final note = [];
    final onset = [];
    for (var window in data) {
      final input = Float32List.fromList(window);
      final inferenceResult = await inference(input, false);
      final noteOutput = inferenceResult['note'] as List<List<List<double>>>;
      final onsetOutput = inferenceResult['onset'] as List<List<List<double>>>;
      final contourOutput = inferenceResult['contour'] as List<List<List<double>>>;
      contour.add(contourOutput[0]);
      note.add(noteOutput[0]);
      onset.add(onsetOutput[0]);
    }
    final noteEvents = postprocess(contour.cast<List<List<double>>>(),
      note.cast<List<List<double>>>(),
      onset.cast<List<List<double>>>(),
      originalAudioLength,
      onsetThreshold,
      frameThreshold,
      minimalNoteLength,
      minimalFrequency,
      maximalFrequency,
      multiplePitchBends,
      melodiaTrick
    );
    return noteEvents;
  }

  Future<List<Map<String, Object?>>> predictFile(
      String filePath,
      {double onsetThreshold = 0.5,
      double frameThreshold = 0.3,
      double minimalNoteLength = 200,
      double minimalFrequency = 80,
      double maximalFrequency = 1100,
      bool multiplePitchBends = false,
      bool melodiaTrick = true}) async {
    final audioData = await loadAudioWithPreProcess(filePath);
    final data = audioData['data'] as List<List<double>>;
    final originalAudioLength = audioData['originalAudioLength'] as int;

    return predict(data, originalAudioLength,
      onsetThreshold: onsetThreshold,
      frameThreshold: frameThreshold,
      minimalNoteLength: minimalNoteLength,
      minimalFrequency: minimalFrequency,
      maximalFrequency: maximalFrequency,
      multiplePitchBends: multiplePitchBends,
      melodiaTrick: melodiaTrick);
  }

  Future<List<Map<String, Object?>>> predictBytes(Uint8List bytes,
      {double onsetThreshold = 0.5,
      double frameThreshold = 0.3,
      double minimalNoteLength = 200,
      double minimalFrequency = 80,
      double maximalFrequency = 1100,
      bool multiplePitchBends = false,
      bool melodiaTrick = true}) async {
    final audioData = await loadAudioBytesWithPreProcess(bytes);
    final data = audioData['data'] as List<List<double>>;
    final originalAudioLength = audioData['originalAudioLength'] as int;

    return predict(data, originalAudioLength,
      onsetThreshold: onsetThreshold,
      frameThreshold: frameThreshold,
      minimalNoteLength: minimalNoteLength,
      minimalFrequency: minimalFrequency,
      maximalFrequency: maximalFrequency,
      multiplePitchBends: multiplePitchBends,
      melodiaTrick: melodiaTrick);
  }

  // Please make sure the sample rate equals 22050, pcm data is mono and 16-bit
  // num of samples should be windowedAudioDataLength - overlappingFrames * fftHopSize * 0.5
  Future<List<Map<String, Object?>>> predictStreamingPcmMono(Uint8List bytes,
      int sampleRate,
      {double onsetThreshold = 0.5,
      double frameThreshold = 0.3,
      double minimalNoteLength = 200,
      double minimalFrequency = 80,
      double maximalFrequency = 1100,
      bool multiplePitchBends = false,
      bool melodiaTrick = true}) async {

    if (sampleRate != expectedAudioSampleRate) {
      throw ArgumentError('Unsupported sample rate');
    }

    List<double> data;
    final int16Data = Int16List.sublistView(Uint8List.fromList(bytes));
    data = int16Data.map((e) => e / 32768.0).toList();
    int originalAudioLength = data.length;
    const overlapLengthHalf = overlappingFrames * fftHopSize ~/ 2;
    if (originalAudioLength != windowedAudioDataLength - overlapLengthHalf) {
      throw ArgumentError('Input data length must be ${windowedAudioDataLength - overlapLengthHalf}');
    }

    if (_lastAudioDataOverlap.isEmpty) {
      _lastAudioDataOverlap = List<double>.filled(overlapLengthHalf, 0.0);
    }
    final paddedData = _lastAudioDataOverlap + data;
    _lastAudioDataOverlap = data.sublist(data.length - overlapLengthHalf);
    originalAudioLength = paddedData.length;
    assert (originalAudioLength == windowedAudioDataLength);
    List<List<double>> input = [paddedData];
    
    return predict(input, originalAudioLength,
      onsetThreshold: onsetThreshold,
      frameThreshold: frameThreshold,
      minimalNoteLength: minimalNoteLength,
      minimalFrequency: minimalFrequency,
      maximalFrequency: maximalFrequency,
      multiplePitchBends: multiplePitchBends,
      melodiaTrick: melodiaTrick);
  }

  void clearLastAudioDataOverlap() {
    _lastAudioDataOverlap = [];
  }
}
