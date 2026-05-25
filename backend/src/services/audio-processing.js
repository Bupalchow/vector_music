import { readFile } from 'node:fs/promises';
import decodeAudio from 'audio-decode';
import Meyda from 'meyda';

export async function audioFileToTuneText(audioFilePath) {
	const fileBuffer = await readFile(audioFilePath);
	const audioBuffer = await decodeAudio(fileBuffer);

	if (!audioBuffer || !audioBuffer.numberOfChannels || !audioBuffer.length) {
		throw new Error('Invalid audio file or unsupported format.');
	}

	const sampleRate = audioBuffer.sampleRate;
	const durationSec = audioBuffer.length / sampleRate;
	const channels = [];
	for (let c = 0; c < audioBuffer.numberOfChannels; c += 1) {
		channels.push(audioBuffer.getChannelData(c));
	}

	const mono = new Float32Array(audioBuffer.length);
	for (let i = 0; i < audioBuffer.length; i += 1) {
		let sum = 0;
		for (let c = 0; c < channels.length; c += 1) {
			sum += channels[c][i] || 0;
		}
		mono[i] = sum / channels.length;
	}

	const bufferSize = 2048;
	const hopSize = 512;
	const featureNames = ['rms', 'zcr', 'spectralCentroid', 'spectralRolloff', 'chroma', 'mfcc'];

	let previousFrame = null;
	const rmsValues = [];
	const zcrValues = [];
	const centroidValues = [];
	const rolloffValues = [];
	const chromaSum = new Array(12).fill(0);
	const mfccSum = new Array(13).fill(0);
	let frameCount = 0;

	for (let start = 0; start + bufferSize <= mono.length; start += hopSize) {
		const frame = mono.slice(start, start + bufferSize);
		const features = Meyda.extract(featureNames, frame, {
			sampleRate,
			bufferSize,
			numberOfMFCCCoefficients: 13,
			melBands: 26,
			previousSignal: previousFrame
		});

		previousFrame = frame;
		if (!features) continue;

		frameCount += 1;
		rmsValues.push(features.rms ?? 0);
		zcrValues.push(features.zcr ?? 0);
		centroidValues.push(features.spectralCentroid ?? 0);
		rolloffValues.push(features.spectralRolloff ?? 0);

		if (Array.isArray(features.chroma) && features.chroma.length === 12) {
			for (let i = 0; i < 12; i += 1) chromaSum[i] += features.chroma[i];
		}

		if (Array.isArray(features.mfcc) && features.mfcc.length) {
			const limit = Math.min(13, features.mfcc.length);
			for (let i = 0; i < limit; i += 1) mfccSum[i] += features.mfcc[i];
		}
	}

	if (!frameCount) {
		throw new Error('Audio is too short to extract tune features.');
	}

	const mean = (arr) => arr.reduce((acc, v) => acc + v, 0) / Math.max(1, arr.length);
	const avgRms = mean(rmsValues);
	const avgZcr = mean(zcrValues);
	const avgCentroid = mean(centroidValues);
	const avgRolloff = mean(rolloffValues);

	const rmsVar = rmsValues.reduce((acc, v) => acc + (v - avgRms) ** 2, 0) / rmsValues.length;
	const dynamicRange = Math.sqrt(rmsVar);

	const onsetEnvelope = [];
	for (let i = 1; i < rmsValues.length; i += 1) {
		onsetEnvelope.push(Math.max(0, rmsValues[i] - rmsValues[i - 1]));
	}

	const framesPerSecond = sampleRate / hopSize;
	const minBpm = 60;
	const maxBpm = 180;
	const minLag = Math.floor((60 / maxBpm) * framesPerSecond);
	const maxLag = Math.ceil((60 / minBpm) * framesPerSecond);

	let bestLag = minLag;
	let bestCorr = -Infinity;

	for (let lag = minLag; lag <= maxLag; lag += 1) {
		let corr = 0;
		for (let i = lag; i < onsetEnvelope.length; i += 1) {
			corr += onsetEnvelope[i] * onsetEnvelope[i - lag];
		}
		if (corr > bestCorr) {
			bestCorr = corr;
			bestLag = lag;
		}
	}

	const tempoBpm = Math.round((60 * framesPerSecond) / Math.max(1, bestLag));

	const keyNames = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
	const majorProfile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88];
	const minorProfile = [6.33, 2.68, 3.52, 5.38, 2.6, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17];
	const chromaNorm = chromaSum.map((v) => v / frameCount);

	let bestKeyIndex = 0;
	let bestMode = 'major';
	let bestScore = -Infinity;

	for (let shift = 0; shift < 12; shift += 1) {
		let majorScore = 0;
		let minorScore = 0;
		for (let i = 0; i < 12; i += 1) {
			const chromaValue = chromaNorm[(i + shift) % 12];
			majorScore += chromaValue * majorProfile[i];
			minorScore += chromaValue * minorProfile[i];
		}

		if (majorScore > bestScore) {
			bestScore = majorScore;
			bestKeyIndex = shift;
			bestMode = 'major';
		}

		if (minorScore > bestScore) {
			bestScore = minorScore;
			bestKeyIndex = shift;
			bestMode = 'minor';
		}
	}

	const normalizedCentroid = avgCentroid / (sampleRate / 2);
	const normalizedRolloff = avgRolloff / (sampleRate / 2);
	const mfccAverages = mfccSum.map((v) => Number((v / frameCount).toFixed(3)));

	const energyLabel = avgRms > 0.12 ? 'high' : avgRms > 0.06 ? 'medium' : 'low';
	const brightnessLabel = normalizedCentroid > 0.35 ? 'bright' : normalizedCentroid > 0.2 ? 'balanced' : 'warm';
	const motionLabel = tempoBpm >= 130 ? 'driving' : tempoBpm >= 100 ? 'steady' : 'laid-back';
	const textureLabel = avgZcr > 0.12 ? 'edgy' : avgZcr > 0.06 ? 'clean' : 'smooth';

	const tuneText = [
		`tempo_bpm:${tempoBpm}`,
		`key:${keyNames[bestKeyIndex]}`,
		`mode:${bestMode}`,
		`duration_sec:${durationSec.toFixed(2)}`,
		`energy_level:${energyLabel}`,
		`groove:${motionLabel}`,
		`brightness:${brightnessLabel}`,
		`texture:${textureLabel}`,
		`dynamic_range:${dynamicRange.toFixed(4)}`,
		`spectral_centroid_norm:${normalizedCentroid.toFixed(4)}`,
		`spectral_rolloff_norm:${normalizedRolloff.toFixed(4)}`,
		`vibe_tags:${[energyLabel, motionLabel, brightnessLabel, textureLabel, bestMode].join('|')}`,
		`mfcc_signature:${mfccAverages.slice(0, 8).join(',')}`,
		`chroma_signature:${chromaNorm.map((v) => v.toFixed(3)).join(',')}`
	].join('\n');

	return tuneText;
}
