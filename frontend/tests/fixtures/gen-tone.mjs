/**
 * Regenerate `tone.wav` — a 0.5s 440Hz mono 16-bit PCM sine.
 *
 * Committed as a binary fixture (browsers decode WAV PCM natively, so
 * the stem-load e2e doesn't need ffmpeg), but kept reproducible here so
 * the bytes aren't a mystery blob. Run: `bun e2e/fixtures/gen-tone.mjs`.
 */
import { writeFileSync } from 'node:fs';

const sampleRate = 44100;
const seconds = 0.5;
const freq = 440;
const n = Math.floor(sampleRate * seconds);

const dataBytes = n * 2;
const buf = Buffer.alloc(44 + dataBytes);

buf.write('RIFF', 0);
buf.writeUInt32LE(36 + dataBytes, 4);
buf.write('WAVE', 8);
buf.write('fmt ', 12);
buf.writeUInt32LE(16, 16); // PCM fmt chunk size
buf.writeUInt16LE(1, 20); // audio format = PCM
buf.writeUInt16LE(1, 22); // channels = mono
buf.writeUInt32LE(sampleRate, 24);
buf.writeUInt32LE(sampleRate * 2, 28); // byte rate
buf.writeUInt16LE(2, 32); // block align
buf.writeUInt16LE(16, 34); // bits per sample
buf.write('data', 36);
buf.writeUInt32LE(dataBytes, 40);

for (let i = 0; i < n; i++) {
  const s = Math.sin((2 * Math.PI * freq * i) / sampleRate);
  buf.writeInt16LE(Math.round(s * 0.6 * 32767), 44 + i * 2);
}

writeFileSync(new URL('./tone.wav', import.meta.url), buf);
console.log(`wrote tone.wav (${buf.length} bytes)`);
