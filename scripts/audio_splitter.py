import os
import librosa
import soundfile as sf

def split_audio(
    input_audio_path,
    output_dir,
    chunk_duration=5.0
):
    os.makedirs(output_dir, exist_ok=True)

    y, sr = librosa.load(input_audio_path, sr=None)

    chunk_length = int(chunk_duration * sr)
    total_chunks = int(len(y) / chunk_length) + 1

    base_name = os.path.splitext(os.path.basename(input_audio_path))[0]

    for i in range(total_chunks):
        start = i * chunk_length
        end = start + chunk_length
        chunk = y[start:end]

        if len(chunk) == 0:
            continue

        output_path = os.path.join(
            output_dir,
            f"{base_name}_chunk_{i:04d}.wav"
        )

        sf.write(output_path, chunk, sr)

    print(f"Saved {total_chunks} chunks to '{output_dir}'")


if __name__ == "__main__":
    input_audio = "george_downsampled_single_channel.wav"
    output_directory = "elevenlabs-dataset/real"

    split_audio(input_audio, output_directory)
