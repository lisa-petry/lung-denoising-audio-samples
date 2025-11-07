// ============================================================================
// STFT (Short-Time Fourier Transform) Implementation
// Based on Python implementation by Frank Zalkow
// ============================================================================

/**
 * Hanning window function
 */
function hanningWindow(size) {
    const window = new Float32Array(size);
    for (let i = 0; i < size; i++) {
        window[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / (size - 1)));
    }
    return window;
}

/**
 * Short-time Fourier transform (returns PSD for each frame)
 * signal: Float32Array (mono)
 * frameSize: NFFT
 * overlapFac: fraction (0..1) or absolute noverlap if > 1 (we accept both)
 * sampleRate: Fs in Hz
 */
function stft(signal, frameSize, overlapFac = 0.5, sampleRate = 4000) {
    // If overlapFac > 1 treat it as number of overlap samples (noverlap)
    const noverlap = (overlapFac > 1) ? Math.floor(overlapFac) : Math.floor(overlapFac * frameSize);
    const hopSize = frameSize - noverlap;
    if (hopSize <= 0) throw new Error('hopSize must be > 0');

    // Window
    const win = hanningWindow(frameSize);

    // Decide number of frames. We WILL zero-pad the end so last partial frame is included (matches many specgram behaviors).
    const numFrames = Math.ceil((signal.length - frameSize) / hopSize) + 1;
    const paddedLength = (numFrames - 1) * hopSize + frameSize;
    const paddedSignal = new Float32Array(paddedLength);
    paddedSignal.set(signal, 0); // no front padding (start at t=0), only pad end with zeros implicitly

    const spectrogram = [];
    for (let frame = 0; frame < numFrames; frame++) {
        const start = frame * hopSize;
        // build windowed frame
        const frameBuf = new Float32Array(frameSize);
        for (let i = 0; i < frameSize; i++) {
            frameBuf[i] = paddedSignal[start + i] * win[i];
        }
        const psd = rfft(frameBuf, win, sampleRate);
        spectrogram.push(psd);
    }

    return spectrogram;
}

/**
 * Real FFT -> returns one-sided Power Spectral Density (PSD) as Float32Array
 * signal: windowed frame (length = n)
 * window: the window that was applied (same length)
 * sampleRate: Fs
 *
 * PSD formula used here:
 *   PSD = (|FFT|^2) / (Fs * U)
 * where U = sum(window^2)
 * Then for one-sided (positive freqs) multiply non-DC and non-Nyquist bins by 2.
 *
 * This is the same scaling used by scipy.signal.spectrogram with 'density' scaling.
 */
function rfft(signal, window, sampleRate) {
    const n = signal.length;
    // compute fft (returns arrays of length fftSize which is power of two >= n)
    const fft = computeFFT(signal); // { real: Float32Array, imag: Float32Array }
    const fftSize = fft.real.length;

    // window normalization U = sum(window^2)
    let windowSumSq = 0;
    for (let i = 0; i < window.length; i++) windowSumSq += window[i] * window[i];

    // PSD scaling denominator: Fs * U
    const denom = sampleRate * windowSumSq;

    // rfft size (positive frequencies)
    const rfftSize = Math.floor(fftSize / 2) + 1;
    const psd = new Float32Array(rfftSize);

    for (let k = 0; k < rfftSize; k++) {
        const re = fft.real[k];
        const im = fft.imag[k];
        let power = re * re + im * im; // |FFT|^2

        // Note: FFT implementation here does NOT normalize by fftSize,
        // so the |FFT|^2 contains the raw energy; dividing by denom gives PSD (density).
        // For one-sided spectrum multiply non-DC/non-Nyquist by 2:
        if (k > 0 && k < rfftSize - 1) power *= 2;

        psd[k] = power / denom;
    }
    return psd;
}

/**
 * computeFFT: in-place Cooley-Tukey radix-2 FFT that returns complex arrays
 * This implementation zero-pads input to nearest power-of-two length.
 * Input: real-valued array (Float32Array)
 * Output: { real: Float32Array, imag: Float32Array } length = fftSize (power of 2)
 */
function computeFFT(signal) {
    // Copy and zero-pad to next power of two
    const n = signal.length;
    let fftSize = 1;
    while (fftSize < n) fftSize <<= 1;

    const real = new Float32Array(fftSize);
    const imag = new Float32Array(fftSize);

    for (let i = 0; i < n; i++) real[i] = signal[i];
    for (let i = n; i < fftSize; i++) real[i] = 0;

    // Bit reversal permutation
    let j = 0;
    for (let i = 1; i < fftSize; i++) {
        let bit = fftSize >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            const tmpR = real[i]; real[i] = real[j]; real[j] = tmpR;
            const tmpI = imag[i]; imag[i] = imag[j]; imag[j] = tmpI;
        }
    }

    // Cooley-Tukey
    for (let len = 2; len <= fftSize; len <<= 1) {
        const halfLen = len >>> 1;
        const theta = -2 * Math.PI / len;
        const wpr = Math.cos(theta);
        const wpi = Math.sin(theta);
        for (let i = 0; i < fftSize; i += len) {
            // initial twiddle
            let wr = 1.0, wi = 0.0;
            for (let k = 0; k < halfLen; k++) {
                const even = i + k;
                const odd = i + k + halfLen;

                const tr = wr * real[odd] - wi * imag[odd];
                const ti = wr * imag[odd] + wi * real[odd];

                real[odd] = real[even] - tr;
                imag[odd] = imag[even] - ti;
                real[even] = real[even] + tr;
                imag[even] = imag[even] + ti;

                // update wr, wi using complex multiplication by w = exp(i*theta*k)
                const tmpWr = wr * wpr - wi * wpi;
                const tmpWi = wr * wpi + wi * wpr;
                wr = tmpWr;
                wi = tmpWi;
            }
        }
    }

    return { real, imag };
}


/**
 * Convert power spectral density to decibels
 * PSD is already a power quantity, so we use 10*log10 instead of 20*log10
 */
function psdToDb(psd) {
    // Add small epsilon to avoid log(0)
    return 10 * Math.log10(Math.max(psd, 1e-20));
}

// ============================================================================
// Audio Player Class for individual player instances
// ============================================================================

class AudioPlayer {
    constructor(element) {
        this.container = element;
        this.audioPath = element.getAttribute('data-audio');
        this.audio = new Audio();

        // Get DOM elements
        this.playBtn = element.querySelector('.play-btn');
        this.playIcon = element.querySelector('.play-icon');
        this.pauseIcon = element.querySelector('.pause-icon');
        this.currentTimeEl = element.querySelector('.current-time');
        this.durationEl = element.querySelector('.duration');
        this.progressBar = element.querySelector('.progress-bar');
        this.progress = element.querySelector('.progress');
        this.waveformCanvas = element.querySelector('.waveform-canvas');
        this.spectrogramCanvas = element.querySelector('.spectrogram-canvas');

        // Audio context and analyser for visualizations
        this.audioContext = null;
        this.analyser = null;
        this.source = null;
        this.dataArray = null;
        this.bufferLength = null;
        this.waveformData = null;
        this.isPlaying = false;
        this.animationId = null;

        // Spectrogram data
        this.spectrogramData = [];
        this.maxSpectrogramSlices = 200;
        this.spectrogramImage = null;
        this.hasStaticSpectrogram = false;

        // STFT data
        this.stftSpectrogram = null;
        this.stftFreqs = null;
        this.stftSampleRate = null;
        this.useSTFT = true; // Set to false to use simple real-time FFT

        this.init();
    }

    async init() {
        // Set up canvas
        this.setupCanvas();

        // Try to load audio
        this.audio.src = this.audioPath;

        // Set up event listeners
        this.setupEventListeners();

        // Load and visualize waveform
        await this.loadAudioBuffer();

        // Try to load static spectrogram image
        await this.loadSpectrogramImage();
    }

    setupCanvas() {
        // Set canvas sizes to match display size
        const waveformRect = this.waveformCanvas.getBoundingClientRect();
        this.waveformCanvas.width = waveformRect.width * window.devicePixelRatio;
        this.waveformCanvas.height = waveformRect.height * window.devicePixelRatio;

        const spectrogramRect = this.spectrogramCanvas.getBoundingClientRect();
        this.spectrogramCanvas.width = spectrogramRect.width * window.devicePixelRatio;
        this.spectrogramCanvas.height = spectrogramRect.height * window.devicePixelRatio;

        // Scale context to match
        const waveformCtx = this.waveformCanvas.getContext('2d');
        waveformCtx.scale(window.devicePixelRatio, window.devicePixelRatio);

        const spectrogramCtx = this.spectrogramCanvas.getContext('2d');
        spectrogramCtx.scale(window.devicePixelRatio, window.devicePixelRatio);
    }

    setupEventListeners() {
        // Play/Pause button
        this.playBtn.addEventListener('click', () => this.togglePlay());

        // Audio events
        this.audio.addEventListener('loadedmetadata', () => {
            this.durationEl.textContent = this.formatTime(this.audio.duration);
        });

        this.audio.addEventListener('timeupdate', () => {
            this.updateProgress();
        });

        this.audio.addEventListener('ended', () => {
            this.pause();
        });

        this.audio.addEventListener('error', (e) => {
            console.error('Error loading audio:', this.audioPath, e);
            this.container.classList.add('error');
        });

        // Progress bar seeking
        this.progressBar.addEventListener('click', (e) => {
            this.seek(e);
        });

        // Window resize
        window.addEventListener('resize', () => {
            this.setupCanvas();
            this.drawStaticWaveform();
            if (this.hasStaticSpectrogram) {
                this.drawStaticSpectrogram();
            }
        });
    }

    async loadAudioBuffer() {
        try {
            const response = await fetch(this.audioPath);

            // Check if the response is ok (status 200-299)
            if (!response.ok) {
                throw new Error(`Failed to load audio file: ${response.status} ${response.statusText}`);
            }

            // Check if the content type is audio
            const contentType = response.headers.get('content-type');
            if (contentType && !contentType.startsWith('audio/')) {
                throw new Error(`Invalid content type: ${contentType}. Expected audio file.`);
            }

            const arrayBuffer = await response.arrayBuffer();

            // Verify that we got some data
            if (arrayBuffer.byteLength === 0) {
                throw new Error('Audio file is empty');
            }

            // Create audio context if not exists
            if (!this.audioContext) {
                this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            }

            // Decode audio data
            const audioBuffer = await this.audioContext.decodeAudioData(arrayBuffer);
            this.waveformData = audioBuffer.getChannelData(0); // Get first channel
            this.stftSampleRate = audioBuffer.sampleRate;

            // Compute STFT spectrogram
            if (this.useSTFT && !this.hasStaticSpectrogram) {
                console.log('Computing STFT spectrogram...');
                const nfft = 256; // FFT size to match user's implementation
                const overlapFac = 0.5; // 50% overlap

                // Compute STFT with PSD (without log scaling)
                const stftResult = stft(this.waveformData, nfft, overlapFac, this.stftSampleRate);

                this.stftSpectrogram = stftResult;
                this.stftFreqs = null; // Linear frequency spacing

                console.log(`STFT computed: ${this.stftSpectrogram.length} time frames, ${this.stftSpectrogram[0].length} frequency bins`);
                console.log(`Sample rate: ${this.stftSampleRate} Hz, nfft: ${nfft}`);

                // Draw the STFT spectrogram
                this.drawPlaceholderSpectrogram();
            }

            // Draw static waveform
            this.drawStaticWaveform();

        } catch (error) {
            console.error('Error loading audio buffer:', this.audioPath, '-', error.message);
            this.drawPlaceholderWaveform();
        }
    }

    async loadSpectrogramImage() {
        // Only load static spectrogram if STFT is disabled
        // When STFT is enabled, we want the interactive visualization
        if (this.useSTFT) {
            console.log('STFT enabled, skipping static spectrogram image');
            return;
        }

        // Generate spectrogram image path from audio path
        // e.g., audio/lung.wav -> audio/lung_spectrogram.png
        const pathWithoutExt = this.audioPath.replace(/\.[^/.]+$/, '');
        const spectrogramPath = `${pathWithoutExt}_spectrogram.png`;

        try {
            // Try to load the image
            const img = new Image();

            await new Promise((resolve, reject) => {
                img.onload = () => {
                    this.spectrogramImage = img;
                    this.hasStaticSpectrogram = true;
                    resolve();
                };
                img.onerror = () => {
                    // Image doesn't exist, that's okay
                    reject();
                };
                img.src = spectrogramPath;
            });

            // Draw the static spectrogram
            this.drawStaticSpectrogram();

        } catch (error) {
            // No static spectrogram available, will use real-time visualization
            console.log('No static spectrogram found for', this.audioPath);
            this.drawPlaceholderSpectrogram();
        }
    }

    drawStaticSpectrogram() {
        if (!this.spectrogramImage) return;

        const canvas = this.spectrogramCanvas;
        const ctx = canvas.getContext('2d');
        const rect = canvas.getBoundingClientRect();
        const width = rect.width;
        const height = rect.height;

        // Clear canvas
        ctx.clearRect(0, 0, width, height);

        // Draw the image to fit the canvas
        ctx.drawImage(this.spectrogramImage, 0, 0, width, height);
    }

    drawPlaceholderSpectrogram() {
        const canvas = this.spectrogramCanvas;
        const ctx = canvas.getContext('2d');
        const rect = canvas.getBoundingClientRect();
        const width = rect.width;
        const height = rect.height;

        ctx.clearRect(0, 0, width, height);

        // If STFT is computed, draw it
        if (this.useSTFT && this.stftSpectrogram) {
            this.drawSTFTSpectrogram();
            return;
        }

        // Draw placeholder text
        ctx.fillStyle = '#dee2e6';
        ctx.font = '14px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('Play to see spectrogram', width / 2, height / 2);
    }

    drawStaticWaveform() {
        if (!this.waveformData) {
            this.drawPlaceholderWaveform();
            return;
        }

        const canvas = this.waveformCanvas;
        const ctx = canvas.getContext('2d');
        const rect = canvas.getBoundingClientRect();
        const width = rect.width;
        const height = rect.height;

        ctx.clearRect(0, 0, width, height);

        // Draw waveform
        const step = Math.ceil(this.waveformData.length / width);
        const amp = height / 2;

        ctx.beginPath();
        ctx.strokeStyle = '#A987FF';
        ctx.lineWidth = 1;

        for (let i = 0; i < width; i++) {
            const min = this.getMinInRange(i * step, (i + 1) * step);
            const max = this.getMaxInRange(i * step, (i + 1) * step);

            const yMin = (1 + min) * amp;
            const yMax = (1 + max) * amp;

            if (i === 0) {
                ctx.moveTo(i, yMin);
            }
            ctx.lineTo(i, yMin);
            ctx.lineTo(i, yMax);
        }

        ctx.stroke();

        // Draw center line
        ctx.beginPath();
        ctx.strokeStyle = '#e9ecef';
        ctx.lineWidth = 1;
        ctx.moveTo(0, height / 2);
        ctx.lineTo(width, height / 2);
        ctx.stroke();
    }

    drawPlaceholderWaveform() {
        const canvas = this.waveformCanvas;
        const ctx = canvas.getContext('2d');
        const rect = canvas.getBoundingClientRect();
        const width = rect.width;
        const height = rect.height;

        ctx.clearRect(0, 0, width, height);

        // Draw placeholder text
        ctx.fillStyle = '#dee2e6';
        ctx.font = '14px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('Audio file not found', width / 2, height / 2);
    }

    getMinInRange(start, end) {
        let min = 1.0;
        for (let i = start; i < end && i < this.waveformData.length; i++) {
            const val = this.waveformData[i];
            if (val < min) min = val;
        }
        return min;
    }

    getMaxInRange(start, end) {
        let max = -1.0;
        for (let i = start; i < end && i < this.waveformData.length; i++) {
            const val = this.waveformData[i];
            if (val > max) max = val;
        }
        return max;
    }

    async togglePlay() {
        if (this.isPlaying) {
            this.pause();
        } else {
            await this.play();
        }
    }

    async play() {
        try {
            // Initialize audio context on first play
            if (!this.source) {
                await this.initializeAudioAnalyser();
            }

            if (this.audioContext.state === 'suspended') {
                await this.audioContext.resume();
            }

            await this.audio.play();
            this.isPlaying = true;
            this.playIcon.classList.add('hidden');
            this.pauseIcon.classList.remove('hidden');
            this.container.classList.add('playing');

            // Start visualization
            this.visualize();

        } catch (error) {
            console.error('Error playing audio:', error);
        }
    }

    pause() {
        this.audio.pause();
        this.isPlaying = false;
        this.playIcon.classList.remove('hidden');
        this.pauseIcon.classList.add('hidden');
        this.container.classList.remove('playing');

        // Stop visualization
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
    }

    async initializeAudioAnalyser() {
        if (!this.audioContext) {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        }

        // Create analyser
        this.analyser = this.audioContext.createAnalyser();
        this.analyser.fftSize = 256;
        this.bufferLength = this.analyser.frequencyBinCount;
        this.dataArray = new Uint8Array(this.bufferLength);

        // Connect audio element to analyser
        this.source = this.audioContext.createMediaElementSource(this.audio);
        this.source.connect(this.analyser);
        this.analyser.connect(this.audioContext.destination);
    }

    visualize() {
        if (!this.isPlaying) return;

        // Draw spectrogram (only if we don't have a static one)
        if (!this.hasStaticSpectrogram) {
            if (this.useSTFT && this.stftSpectrogram) {
                this.drawSTFTSpectrogram();
            } else {
                // Get frequency data for simple real-time spectrogram
                this.analyser.getByteFrequencyData(this.dataArray);
                this.drawSpectrogram();
            }
        }

        // Draw waveform with progress indicator
        this.drawWaveformWithProgress();

        // Continue animation
        this.animationId = requestAnimationFrame(() => this.visualize());
    }

// Replace the existing drawSTFTSpectrogram method in AudioPlayer with this function
drawSTFTSpectrogram() {
    const canvas = this.spectrogramCanvas;
    const ctx = canvas.getContext('2d');
    const rect = canvas.getBoundingClientRect();
    // Use CSS pixel sizes because setupCanvas already scaled the context
    const width = Math.max(1, Math.floor(rect.width));
    const height = Math.max(1, Math.floor(rect.height));

    // Clear canvas first
    ctx.clearRect(0, 0, width, height);

    // Basic guards
    if (!this.stftSpectrogram || !this.stftSpectrogram.length) {
        // nothing to draw
        ctx.fillStyle = '#dee2e6';
        ctx.font = '14px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('Spectrogram not available', width / 2, height / 2);
        return;
    }

    const frames = this.stftSpectrogram.length;
    const bins = this.stftSpectrogram[0].length; // number of freq bins in each frame
    const nfft = 256; // must match STFT used when computing stftSpectrogram
    const sampleRate = this.stftSampleRate || 44100;

    // Choose the max display frequency (Hz) — tweak as you like
    const maxDisplayFreq = 1000;
    const maxFreqBin = Math.min(bins - 1, Math.ceil(maxDisplayFreq * nfft / sampleRate));

    // --- Convert PSD -> dB and find min/max dB in the displayed range ---
    // We'll compute min/max only for the portion we draw (0..maxFreqBin)
    const eps = 1e-12;
    let minDb = Infinity, maxDb = -Infinity;

    // Convert and cache dB values in a temporary Float32Array per frame to avoid repeated Math.log10
    const dbSpec = new Array(frames);
    for (let t = 0; t < frames; t++) {
        const frame = this.stftSpectrogram[t];
        const dbFrame = new Float32Array(maxFreqBin + 1);
        for (let f = 0; f <= maxFreqBin; f++) {
            // PSD -> dB (10*log10)
            const db = 10 * Math.log10(frame[f] + eps);
            dbFrame[f] = db;
            if (db < minDb) minDb = db;
            if (db > maxDb) maxDb = db;
        }
        dbSpec[t] = dbFrame;
    }

    // Set sensible display dB range (clamp to a window for better contrast)
    const vmin = Math.max(minDb, maxDb - 80); // show up to 80 dB dynamic range
    const vmax = maxDb;

    // Quantized color map (you can change levels)
    const colorLevels = 64;
    const colorMap = (function makeQuantizedColorMap(levels) {
        // base colors for a pleasant perceptual-ish gradient
        const base = [
            [0, 0, 0],
            [0, 0, 128],
            [0, 0, 255],
            [0, 128, 255],
            [0, 255, 255],
            [0, 255, 0],
            [255, 255, 0],
            [255, 128, 0],
            [255, 0, 0],
            [255, 255, 255]
        ];
        const map = new Array(levels);
        for (let i = 0; i < levels; i++) {
            const t = i / (levels - 1) * (base.length - 1);
            const i0 = Math.floor(t), i1 = Math.min(i0 + 1, base.length - 1);
            const w = t - i0;
            const r = Math.round(base[i0][0] * (1 - w) + base[i1][0] * w);
            const g = Math.round(base[i0][1] * (1 - w) + base[i1][1] * w);
            const b = Math.round(base[i0][2] * (1 - w) + base[i1][2] * w);
            map[i] = [r, g, b];
        }
        return map;
    })(colorLevels);

    // Prepare pixel buffer via ImageData
    // Note: canvas has been scaled by devicePixelRatio in setupCanvas, but we're drawing with CSS coords.
    // createImageData expects width/height in CSS pixels because context is already scaled
    const imageData = ctx.createImageData(width, height);
    const pixels = imageData.data; // Uint8ClampedArray length = width * height * 4

    // Determine horizontal binning (average frames when frames > width)
    const framesPerPixel = frames / width;

    // For each output pixel column x, compute an averaged frame of dB values
    for (let x = 0; x < width; x++) {
        // frame range for this pixel
        const startFrame = Math.floor(x * framesPerPixel);
        const endFrame = Math.min(frames, Math.floor((x + 1) * framesPerPixel));
        const countFrames = Math.max(1, endFrame - startFrame);

        // accumulate dB per frequency bin (only up to maxFreqBin)
        const avg = new Float32Array(maxFreqBin + 1);
        for (let t = startFrame; t < startFrame + countFrames && t < frames; t++) {
            const fArr = dbSpec[t];
            for (let f = 0; f <= maxFreqBin; f++) avg[f] += fArr[f];
        }
        for (let f = 0; f <= maxFreqBin; f++) avg[f] /= countFrames;

        // map frequency bins -> vertical pixels
        // Use linear mapping; for log scale, replace mapping code below
        for (let y = 0; y < height; y++) {
            // pixel y corresponds to frequency bin (flip so low freq at bottom)
            const frac = y / height; // 0..1 top->bottom
            // Map frac to bin index (0 = top = highest freq, maxFreqBin = bottom = lowest freq)
            const binFloat = (1 - frac) * (maxFreqBin);
            const binLo = Math.floor(binFloat);
            const binHi = Math.min(maxFreqBin, binLo + 1);
            const w = binFloat - binLo;
            // linear interpolate between two nearest bins for vertical smoothing
            const valDb = (1 - w) * avg[binLo] + w * avg[binHi];

            // normalize to 0..1 using vmin..vmax
            const norm = (valDb - vmin) / (vmax - vmin);
            const clamped = Math.max(0, Math.min(1, norm));
            const colorIndex = Math.floor(clamped * (colorLevels - 1));
            const col = colorMap[colorIndex];

            const pixelIdx = 4 * (y * width + x);
            pixels[pixelIdx] = col[0];
            pixels[pixelIdx + 1] = col[1];
            pixels[pixelIdx + 2] = col[2];
            pixels[pixelIdx + 3] = 255; // opaque
        }
    }

    // Draw the ImageData to the canvas. Because we built top->bottom, and we mapped bins accordingly,
    // no further vertical flipping is required.
    ctx.putImageData(imageData, 0, 0);

    // Draw playback progress line on top
    if (!isNaN(this.audio.duration) && this.audio.duration > 0) {
        const currentTime = this.audio.currentTime;
        const duration = this.audio.duration;
        const progressX = (currentTime / duration) * width;

        ctx.save();
        ctx.strokeStyle = 'rgba(255,255,255,0.9)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(progressX, 0);
        ctx.lineTo(progressX, height);
        ctx.stroke();
        ctx.restore();
    }
}


    // Jet colormap approximation (similar to matplotlib)
    jetColormap(value) {
        // Clamp value between 0 and 1
        value = Math.max(0, Math.min(1, value));

        let r, g, b;

        if (value < 0.25) {
            r = 0;
            g = 0;
            b = 0.5 + value * 2;
        } else if (value < 0.5) {
            r = 0;
            g = (value - 0.25) * 4;
            b = 1;
        } else if (value < 0.75) {
            r = (value - 0.5) * 4;
            g = 1;
            b = 1 - (value - 0.5) * 4;
        } else {
            r = 1;
            g = 1 - (value - 0.75) * 4;
            b = 0;
        }

        return `rgb(${Math.floor(r * 255)}, ${Math.floor(g * 255)}, ${Math.floor(b * 255)})`;
    }

    drawSpectrogram() {
        const canvas = this.spectrogramCanvas;
        const ctx = canvas.getContext('2d');
        const rect = canvas.getBoundingClientRect();
        const width = rect.width;
        const height = rect.height;

        // Shift existing spectrogram data to the left
        const imageData = ctx.getImageData(1, 0, width - 1, height);
        ctx.putImageData(imageData, 0, 0);

        // Draw new frequency data column on the right
        const sliceWidth = 1;
        const barHeight = height / this.bufferLength;

        for (let i = 0; i < this.bufferLength; i++) {
            const value = this.dataArray[i];
            const percent = value / 255;

            // Create color based on frequency intensity
            const hue = 240 - (percent * 100); // Blue to cyan
            const saturation = 100;
            const lightness = 20 + (percent * 60);

            ctx.fillStyle = `hsl(${hue}, ${saturation}%, ${lightness}%)`;

            // Draw from bottom to top (low frequencies at bottom)
            const y = height - (i * barHeight) - barHeight;
            ctx.fillRect(width - sliceWidth, y, sliceWidth, barHeight);
        }
    }

    drawWaveformWithProgress() {
        if (!this.waveformData) return;

        const canvas = this.waveformCanvas;
        const ctx = canvas.getContext('2d');
        const rect = canvas.getBoundingClientRect();
        const width = rect.width;
        const height = rect.height;

        ctx.clearRect(0, 0, width, height);

        // Draw full waveform
        const step = Math.ceil(this.waveformData.length / width);
        const amp = height / 2;

        // Calculate progress position
        const progressPos = (this.audio.currentTime / this.audio.duration) * width;

        // Draw played portion (darker blue)
        ctx.beginPath();
        ctx.strokeStyle = '#392d56';
        ctx.lineWidth = 1.5;

        for (let i = 0; i < progressPos && i < width; i++) {
            const min = this.getMinInRange(i * step, (i + 1) * step);
            const max = this.getMaxInRange(i * step, (i + 1) * step);

            const yMin = (1 + min) * amp;
            const yMax = (1 + max) * amp;

            if (i === 0) {
                ctx.moveTo(i, yMin);
            }
            ctx.lineTo(i, yMin);
            ctx.lineTo(i, yMax);
        }

        ctx.stroke();

        // Draw unplayed portion (lighter blue)
        ctx.beginPath();
        ctx.strokeStyle = '#987ae5';
        ctx.lineWidth = 1;

        for (let i = Math.floor(progressPos); i < width; i++) {
            const min = this.getMinInRange(i * step, (i + 1) * step);
            const max = this.getMaxInRange(i * step, (i + 1) * step);

            const yMin = (1 + min) * amp;
            const yMax = (1 + max) * amp;

            if (i === Math.floor(progressPos)) {
                ctx.moveTo(i, yMin);
            }
            ctx.lineTo(i, yMin);
            ctx.lineTo(i, yMax);
        }

        ctx.stroke();

        // Draw progress indicator line
        ctx.beginPath();
        ctx.strokeStyle = '#655199';
        ctx.lineWidth = 2;
        ctx.moveTo(progressPos, 0);
        ctx.lineTo(progressPos, height);
        ctx.stroke();

        // Draw center line
        ctx.beginPath();
        ctx.strokeStyle = '#e9ecef';
        ctx.lineWidth = 1;
        ctx.moveTo(0, height / 2);
        ctx.lineTo(width, height / 2);
        ctx.stroke();
    }

    updateProgress() {
        const percent = (this.audio.currentTime / this.audio.duration) * 100;
        this.progress.style.width = `${percent}%`;
        this.currentTimeEl.textContent = this.formatTime(this.audio.currentTime);
    }

    seek(e) {
        const rect = this.progressBar.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const percent = x / rect.width;
        const time = percent * this.audio.duration;

        this.audio.currentTime = time;
    }

    formatTime(seconds) {
        if (isNaN(seconds)) return '0:00';

        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }
}

// Initialize all audio players when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    const audioPlayers = document.querySelectorAll('.audio-player');

    audioPlayers.forEach(playerElement => {
        new AudioPlayer(playerElement);
    });
});
