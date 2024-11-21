# TODO: remove entire file if ended up working with combined audio
# If keeping, add pydub webrtcvad noisereduce to requirements.txt

from pydub import AudioSegment
import webrtcvad
import numpy as np
import noisereduce as nr

def get_speech_activity(audio_segment, frame_duration_ms=30, aggressiveness=3):
    """
    Returns a list of tuples (start_ms, end_ms) for segments where speech is detected.
    """
    vad = webrtcvad.Vad(aggressiveness)
    sample_rate = audio_segment.frame_rate
    assert sample_rate in (8000, 16000, 32000, 48000), "Sample rate must be 8000, 16000, 32000, or 48000 Hz"

    # Convert audio to raw PCM data
    audio = audio_segment.raw_data
    bytes_per_frame = int(sample_rate * (frame_duration_ms / 1000.0) * audio_segment.sample_width)
    frames = [audio[i:i+bytes_per_frame] for i in range(0, len(audio), bytes_per_frame)]
    
    speech_frames = []
    timestamp = 0
    speech = False
    segments = []
    segment_start = 0

    for frame in frames:
        if len(frame) < bytes_per_frame:
            # Pad frame with zeros
            frame += b'\x00' * (bytes_per_frame - len(frame))
        is_speech = vad.is_speech(frame, sample_rate)
        start_ms = timestamp
        end_ms = timestamp + frame_duration_ms
        if is_speech:
            if not speech:
                # Start of a new speech segment
                segment_start = start_ms
            speech = True
        else:
            if speech:
                # End of a speech segment
                segment_end = end_ms - frame_duration_ms  # End at the previous frame
                # Apply energy threshold to filter out low-energy segments
                segment_audio = audio_segment[segment_start:segment_end]
                if is_high_energy(segment_audio):
                    segments.append((segment_start, segment_end))
                speech = False
        timestamp += frame_duration_ms

    # If speech was ongoing at the end of the audio
    if speech:
        segment_end = timestamp
        segment_audio = audio_segment[segment_start:segment_end]
        if is_high_energy(segment_audio):
            segments.append((segment_start, segment_end))
    return segments

def is_high_energy(segment, threshold=30):
    """
    Determines if the audio segment has high enough energy to be considered speech.
    """
    # Calculate RMS (Root Mean Square) energy
    rms = segment.rms
    return rms > threshold

def reduce_noise(audio_segment):
    """
    Applies noise reduction to the audio segment.
    """
    # Convert AudioSegment to numpy array
    samples = np.array(audio_segment.get_array_of_samples())
    # Apply noise reduction
    reduced_noise = nr.reduce_noise(y=samples.astype(np.float32), sr=audio_segment.frame_rate)
    # Convert back to AudioSegment
    reduced_audio_segment = audio_segment._spawn(reduced_noise.astype(np.int16).tobytes())
    return reduced_audio_segment

def find_overlaps(segments1, segments2):
    """
    Given two lists of speech segments [(start_ms, end_ms)], find overlapping segments.
    Returns a list of overlaps (start_ms, end_ms)
    """
    overlaps = []
    idx1, idx2 = 0, 0
    while idx1 < len(segments1) and idx2 < len(segments2):
        s1_start, s1_end = segments1[idx1]
        s2_start, s2_end = segments2[idx2]

        # Check if segments overlap
        latest_start = max(s1_start, s2_start)
        earliest_end = min(s1_end, s2_end)
        overlap = earliest_end - latest_start
        if overlap > 0:
            overlaps.append((latest_start, earliest_end))

        # Move to next segment
        if s1_end <= s2_end:
            idx1 += 1
        else:
            idx2 += 1
    return overlaps

def merge_close_segments(segments, merge_threshold_ms=1000):
    """
    Merges segments that are within a specified threshold in milliseconds.
    """
    if not segments:
        return []

    merged_segments = [segments[0]]

    for current in segments[1:]:
        prev_start, prev_end = merged_segments[-1]
        curr_start, curr_end = current

        if curr_start - prev_end <= merge_threshold_ms:
            # Merge the current segment with the previous one
            merged_segments[-1] = (prev_start, max(prev_end, curr_end))
        else:
            merged_segments.append(current)

    return merged_segments

def main(audio_file_path):
    audio = AudioSegment.from_file(audio_file_path)
    channels = audio.split_to_mono()
    if len(channels) != 2:
        raise ValueError("Audio file must have exactly two channels")

    channel1 = channels[0]
    channel2 = channels[1]

    # Ensure sample rate is acceptable for VAD
    sample_rate = 16000
    channel1 = channel1.set_frame_rate(sample_rate)
    channel2 = channel2.set_frame_rate(sample_rate)

    # Ensure sample width is 16-bit PCM
    channel1 = channel1.set_sample_width(2)
    channel2 = channel2.set_sample_width(2)

    # Apply noise reduction to each channel
    channel1 = reduce_noise(channel1)
    channel2 = reduce_noise(channel2)

    # Get speech activity for each channel with higher aggressiveness
    speech_segments1 = get_speech_activity(channel1, aggressiveness=3)
    speech_segments2 = get_speech_activity(channel2, aggressiveness=3)

    # Now compare the segments to find overlaps
    overlaps = find_overlaps(speech_segments1, speech_segments2)

    # Merge close overlaps
    merged_overlaps = merge_close_segments(overlaps, merge_threshold_ms=1000)

    # Print the merged overlaps
    if merged_overlaps:
        print("Interruptions detected at the following times:")
        for start_ms, end_ms in merged_overlaps:
            print(f"From {start_ms/1000:.2f} s to {end_ms/1000:.2f} s")
    else:
        print("No interruptions detected.")


def split_audio_channels(input_file, output_file_channel1, output_file_channel2):
    # Load the audio file
    audio = AudioSegment.from_file(input_file)
    
    # Check if the audio has at least two channels
    if audio.channels < 2:
        raise ValueError("Audio file must have at least two channels")
    
    # Split the audio into separate channels
    channels = audio.split_to_mono()
    
    # Save each channel to a separate file
    channel1 = channels[0]
    channel2 = channels[1]
    
    channel1.export(output_file_channel1, format="wav")
    channel2.export(output_file_channel2, format="wav")
    
    print(f"Channel 1 saved to {output_file_channel1}")
    print(f"Channel 2 saved to {output_file_channel2}")

# split_audio_channels("speech_testing/audio_files/interruptions.wav", "speech_testing/audio_files/interruptions_channel1.wav", "speech_testing/audio_files/interruptions_channel2.wav")
# main("speech_testing/audio_files/11x_role_flip.wav")