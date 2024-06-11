import librosa
import soundfile as sf
import json
import glob
import collections
import numpy as np


def concat_raw_audio(audio_files, output_file):
    audio_clips = []
    for audio_file in audio_files:
        audio, sr = librosa.load(audio_file, sr=None)
        audio_clips.append(audio)
    audio_clips = np.concatenate(audio_clips)
    sf.write(output_file, audio_clips, sr, subtype="PCM_24")


if __name__ == "__main__":
    met_file = "/home/qiantong/MixEval/mix_eval/data/mixeval-2024-06-01/mixeval-hard-audio-in/close_freeform_hard_for_tts.json"
    with open(met_file, "r") as f:
        meta_data = json.load(f)

    all_audio_files = glob.glob(
        "/home/qiantong/MixEval/mix_eval/data/mixeval-2024-06-01/mixeval-hard-audio-in/audio_gen/raw/*.wav"
    )
    all_audio_files = sorted(all_audio_files)
    assert len(meta_data) == len(all_audio_files), (
        len(meta_data),
        len(all_audio_files),
    )

    q_to_audios = collections.defaultdict(list)
    for i, d in enumerate(meta_data):
        name = f"{d['question_id']}_{d['voice_id']}"
        q_to_audios[name].append(i)

    for name in q_to_audios:
        print(name, q_to_audios[name], len(all_audio_files))
        audio_files = [all_audio_files[i] for i in q_to_audios[name]]
        output_file = f"/home/qiantong/MixEval/mix_eval/data/mixeval-2024-06-01/mixeval-hard-audio-in/audio_gen/{name}.wav"
        concat_raw_audio(audio_files, output_file)

    orig_file = "/home/qiantong/MixEval/mix_eval/data/model_responses/gemma_11_7b_instruct/mixeval_hard/2024-06-01/gemma_11_7b_instruct_close_freeform_hard.jsonl"
    with open(orig_file, "r") as f:
        data = []
        for line in f:
            data.append(json.loads(line))

    with open(
        "/home/qiantong/MixEval/mix_eval/data/mixeval-2024-06-01/mixeval-hard-audio-in/close_freeform_hard.jsonl",
        "w",
    ) as f:
        for i, d in enumerate(data):
            if d["benchmark_name"] == "MATH":
                continue

            d["audio_files"] = [
                f"/home/qiantong/MixEval/mix_eval/data/mixeval-2024-06-01/mixeval-hard-audio-in/audio_gen/{i}_male-ll.wav",
                f"/home/qiantong/MixEval/mix_eval/data/mixeval-2024-06-01/mixeval-hard-audio-in/audio_gen/{i}_female-ll.wav",
            ]
            del d["response"]
            f.write(json.dumps(d) + "\n")
