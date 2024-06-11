import json
import nltk

nltk.download("punkt")

from nltk.tokenize import sent_tokenize, word_tokenize


def split_long_sentences(text, max_words=30):
    # Split text into sentences
    sentences = sent_tokenize(text)
    short_sentences = []

    for sentence in sentences:
        words = word_tokenize(sentence)
        if len(words) <= max_words:
            short_sentences.append(sentence)
        else:
            # Split long sentence into smaller sentences
            new_sentence = []
            current_count = 0
            for word in words:
                if word[0] not in ["'", ",", ";", "."] and len(new_sentence) > 0:
                    word = " " + word
                new_sentence.append(word)
                current_count += 1
                # Split at a logical point
                if current_count >= max_words and word in [
                    ",",
                    ";",
                    "and",
                    "but",
                    "or",
                ]:
                    short_sentences.append("".join(new_sentence))
                    new_sentence = []
                    current_count = 0
            if new_sentence:
                short_sentences.append("".join(new_sentence))

    return short_sentences


if __name__ == "__main__":
    # orig_file = "/home/qiantong/MixEval/mix_eval/data/model_responses/gemma_11_7b_instruct/mixeval_hard/2024-06-01/gemma_11_7b_instruct_close_freeform_hard.jsonl"
    # output_file = "/home/qiantong/MixEval/mix_eval/data/mixeval-2024-06-01/mixeval-hard-audio-in/close_freeform_hard_for_tts.json"
    orig_file = "/home/qiantong/MixEval/mix_eval/data/model_responses/gemma_11_7b_instruct/mixeval_hard/2024-06-01/gemma_11_7b_instruct_close_multichoice_hard.jsonl"
    output_file = "/home/qiantong/MixEval/mix_eval/data/mixeval-2024-06-01/mixeval-hard-audio-in/close_multichoice_hard_for_tts.json"
    with open(orig_file, "r") as f:
        data = []
        for line in f:
            data.append(json.loads(line))

    tts_prompt = []
    for i, d in enumerate(data):
        if d["benchmark_name"] == "MATH":
            continue
        text = d["formated_input"]
        text = text.replace("Options:\n- Yes\n- No\n", "Options: Yes or No\n")
        text = text.replace("(A)", "")
        text = text.replace("(B)", "")
        text = text.replace("(C)", "")
        text = text.replace("(D)", "")
        text = text.split("\n")
        print(text)
        print("-----")
        sentences = []
        for t in text:
            sentences.extend(split_long_sentences(t))
        max_len = 0
        for s in sentences:
            print(s)
            max_len = max(max_len, len(s.split()))

            tts_prompt.append((i, s))
        print(max_len)
        print("-----------------------------------")

    new_tts_prompt = []
    for i, s in tts_prompt:
        item = {
            "question_id": i,
            "text": s,
            "voice_id": "male-ll",
        }
        new_tts_prompt.append(item)
    for i, s in tts_prompt:
        item = {
            "question_id": i,
            "text": s,
            "voice_id": "female-ll",
        }
        new_tts_prompt.append(item)

    with open(output_file, "w") as f:
        json.dump(new_tts_prompt, f, indent=1)
