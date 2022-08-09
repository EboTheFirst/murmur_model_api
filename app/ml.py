import numpy as np
import librosa
from joblib import load


MAX_SOUND_CLIP_DURATION = 5
OUTCOMES = ['No murmur detected', 'Murmur detected']


from scipy.signal import kaiserord, lfilter, firwin
from numpy import arange
from tensorflow.keras.models import load_model

def filter_audio(audio,sample_rate):
    
    nsamples = sample_rate * MAX_SOUND_CLIP_DURATION
    t = arange(nsamples) / sample_rate

    #------------------------------------------------
    # Create a FIR filter and apply it to x.
    #------------------------------------------------

    # The Nyquist rate of the signal.
    nyq_rate = sample_rate / 2.0

    # The desired width of the transition from pass to stop,
    # relative to the Nyquist rate.  We'll design the filter
    # with a 5 Hz transition width.
    width = 5.0/nyq_rate

    # The desired attenuation in the stop band, in dB.
    ripple_db = 60.0

    # Compute the order and Kaiser parameter for the FIR filter.
    N, beta = kaiserord(ripple_db, width)

    # The cutoff frequency of the filter.
    cutoff_hz = 650.0

    # Use firwin with a Kaiser window to create a lowpass FIR filter.
    taps = firwin(N, cutoff_hz/nyq_rate, window=('kaiser', beta))

    # Use lfilter to filter x with the FIR filter.
    filtered_audio = lfilter(taps, 1.0, audio)
    return filtered_audio

def predict(file):
    audio, sr = librosa.load(file, sr=None, duration=MAX_SOUND_CLIP_DURATION)
    y = filter_audio(audio,sample_rate=sr)
    rms = librosa.feature.rms(y=y)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    to_append = f'{np.mean(chroma_stft)} {np.mean(rms)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
    
    for e in mfcc:
        to_append += f' {np.mean(e)}'
    features = [list(map(float, to_append.split(" ")))]
    
    scaler =load('./filtered_std_scaler.bin')
    features = scaler.transform(features)
    model = load_model("./murmur_1L81-247.h5")

    predicted = model.predict(features)
    certainty = [ max(pred) for pred in predicted]
    predictions = [ np.argmax(np.array(list(map(int,pred == max(pred))))) for pred in predicted]
    
    output = {}
    output['prediction'] = bool(predictions[0])
    output['audio_sample_rate'] = sr

    predictions = [OUTCOMES[p] for p in predictions]
    output['alt_prediction'] = predictions[0]
    output['certainty'] = float(certainty[0])
    # output['audio_array'] = audio.tolist()

    return output
