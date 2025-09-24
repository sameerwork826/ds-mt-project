import json
import os
from itertools import product
from random import Random


def ensure_data_dir(directory_path: str) -> None:
    os.makedirs(directory_path, exist_ok=True)


def write_jsonl(file_path: str, records: list[dict[str, str]]) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def generate_pairs(target_count: int = 1200, seed: int = 42) -> list[dict[str, str]]:
    rng = Random(seed)

    subjects_en = [
        "I",
        "You",
        "We",
        "They",
        "He",
        "She",
        "The boy",
        "The girl",
        "My friend",
        "The teacher",
        "The doctor",
        "The student",
    ]

    subjects_hi = {
        "I": "मैं",
        "You": "तुम",
        "We": "हम",
        "They": "वे",
        "He": "वह",
        "She": "वह",
        "The boy": "लड़का",
        "The girl": "लड़की",
        "My friend": "मेरा दोस्त",
        "The teacher": "अध्यापक",
        "The doctor": "डॉक्टर",
        "The student": "छात्र",
    }

    objects_en = [
        "tea",
        "coffee",
        "rice",
        "books",
        "music",
        "cricket",
        "English",
        "Hindi",
        "the news",
        "a movie",
        "the bus",
        "the train",
    ]

    objects_hi = {
        "tea": "चाय",
        "coffee": "कॉफ़ी",
        "rice": "चावल",
        "books": "किताबें",
        "music": "संगीत",
        "cricket": "क्रिकेट",
        "English": "अंग्रेज़ी",
        "Hindi": "हिंदी",
        "the news": "समाचार",
        "a movie": "एक फ़िल्म",
        "the bus": "बस",
        "the train": "ट्रेन",
    }

    places_en = [
        "at home",
        "at school",
        "at work",
        "in the park",
        "in the market",
        "in Delhi",
        "in Mumbai",
    ]

    places_hi = {
        "at home": "घर पर",
        "at school": "स्कूल में",
        "at work": "काम पर",
        "in the park": "पार्क में",
        "in the market": "बाज़ार में",
        "in Delhi": "दिल्ली में",
        "in Mumbai": "मुंबई में",
    }

    times_en = [
        "today",
        "yesterday",
        "tomorrow",
        "every day",
        "on Monday",
        "at night",
        "in the morning",
    ]

    times_hi = {
        "today": "आज",
        "yesterday": "कल",
        "tomorrow": "कल",
        "every day": "हर दिन",
        "on Monday": "सोमवार को",
        "at night": "रात में",
        "in the morning": "सुबह",
    }

    # Verb templates (simple present, past, future; affirmative, negative, interrogative)
    # English is SVO, Hindi commonly SOV with auxiliaries at the end.
    verbs = [
        {
            "en": "eat",
            "hi_root": "खाना",
            "hi_pres": "खाता/खाती हूँ",
            "hi_past": "खाया/खाई",
            "hi_future": "खाऊँगा/खाऊँगी",
            "object_required": True,
        },
        {
            "en": "drink",
            "hi_root": "पीना",
            "hi_pres": "पीता/पीती हूँ",
            "hi_past": "पीया/पी",
            "hi_future": "पीऊँगा/पीऊँगी",
            "object_required": True,
        },
        {
            "en": "read",
            "hi_root": "पढ़ना",
            "hi_pres": "पढ़ता/पढ़ती हूँ",
            "hi_past": "पढ़ा/पढ़ी",
            "hi_future": "पढ़ूँगा/पढ़ूँगी",
            "object_required": True,
        },
        {
            "en": "go",
            "hi_root": "जाना",
            "hi_pres": "जाता/जाती हूँ",
            "hi_past": "गया/गई",
            "hi_future": "जाऊँगा/जाऊँगी",
            "object_required": False,
        },
        {
            "en": "play",
            "hi_root": "खेलना",
            "hi_pres": "खेलता/खेलती हूँ",
            "hi_past": "खेला/खेले",
            "hi_future": "खेलूँगा/खेलूँगी",
            "object_required": True,
        },
        {
            "en": "watch",
            "hi_root": "देखना",
            "hi_pres": "देखता/देखती हूँ",
            "hi_past": "देखा/देखी",
            "hi_future": "देखूँगा/देखूँगी",
            "object_required": True,
        },
        {
            "en": "learn",
            "hi_root": "सीखना",
            "hi_pres": "सीखता/सीखती हूँ",
            "hi_past": "सीखा/सीखी",
            "hi_future": "सीखूँगा/सीखूँगी",
            "object_required": True,
        },
    ]

    tenses = ["present", "past", "future"]
    polarities = ["affirmative", "negative", "question"]

    def gender_variant(subject_en: str, form_tuple: str) -> str:
        # form_tuple contains two forms separated by '/', masc/fem
        masc, fem = form_tuple.split("/")
        feminine_subjects = {"She", "The girl"}
        return fem if subject_en in feminine_subjects else masc

    def build_en_sentence(subject: str, verb_en: str, obj: str | None, place: str | None, time_exp: str | None, tense: str, polarity: str) -> str:
        # Very simple English templates
        base_v = verb_en
        past_v = {
            "eat": "ate",
            "drink": "drank",
            "read": "read",
            "go": "went",
            "play": "played",
            "watch": "watched",
            "learn": "learned",
        }[verb_en]
        future_aux = "will"

        def vp(tense_: str) -> str:
            if tense_ == "present":
                return base_v
            if tense_ == "past":
                return past_v
            return base_v  # used with 'will'

        parts: list[str] = []
        # Questions use do-support simplistically
        if polarity == "question":
            if tense == "past":
                parts.append("Did")
                parts.append(subject)
                parts.append(base_v)
            elif tense == "future":
                parts.append("Will")
                parts.append(subject)
                parts.append(base_v)
            else:
                parts.append("Do")
                parts.append(subject)
                parts.append(base_v)
            if obj:
                parts.append(obj)
            if place:
                parts.append(place)
            if time_exp:
                parts.append(time_exp)
            sentence = " ".join(parts) + "?"
            return sentence

        if polarity == "negative":
            if tense == "past":
                parts = [subject, "did not", base_v]
            elif tense == "future":
                parts = [subject, future_aux, "not", base_v]
            else:
                parts = [subject, "do not", base_v]
        else:
            if tense == "future":
                parts = [subject, future_aux, vp("future")]
            else:
                parts = [subject, vp(tense)]

        if obj:
            parts.append(obj)
        if place:
            parts.append(place)
        if time_exp:
            parts.append(time_exp)

        return " ".join(parts) + "."

    def build_hi_sentence(subject: str, verb: dict, obj_en: str | None, place_en: str | None, time_en: str | None, tense: str, polarity: str) -> str:
        subj_hi = subjects_hi[subject]
        obj_hi = objects_hi.get(obj_en, None) if obj_en else None
        place_hi = places_hi.get(place_en, None) if place_en else None
        time_hi = times_hi.get(time_en, None) if time_en else None

        # Choose gendered verb where applicable (using मैं default masculine)
        if tense == "present":
            verb_form = gender_variant(subject, verb["hi_pres"]) if "/" in verb["hi_pres"] else verb["hi_pres"]
        elif tense == "past":
            verb_form = gender_variant(subject, verb["hi_past"]) if "/" in verb["hi_past"] else verb["hi_past"]
        else:
            verb_form = gender_variant(subject, verb["hi_future"]) if "/" in verb["hi_future"] else verb["hi_future"]

        # Base SOV order in Hindi; time/place commonly at start or end
        parts_hi: list[str] = []

        if polarity == "question":
            # Simple question with "क्या" at start
            parts_hi.append("क्या")
            if time_hi:
                parts_hi.append(time_hi)
            parts_hi.append(subj_hi)
            if obj_hi:
                parts_hi.append(obj_hi)
            if place_hi:
                parts_hi.append(place_hi)
            parts_hi.append(verb_form)
            sentence_hi = " ".join(parts_hi) + "?"
            return sentence_hi

        if time_hi:
            parts_hi.append(time_hi)
        parts_hi.append(subj_hi)
        if obj_hi:
            parts_hi.append(obj_hi)
        if place_hi:
            parts_hi.append(place_hi)

        if polarity == "negative":
            # Use 'नहीं' before the verb
            parts_hi.append("नहीं")
        parts_hi.append(verb_form)

        return " ".join(parts_hi) + "।"

    records: list[dict[str, str]] = []

    # Build combinations and then sample up to target_count deterministically
    combos = []
    for subject, verb, obj, place, time_exp, tense, pol in product(
        subjects_en,
        verbs,
        objects_en + [None],
        places_en + [None],
        times_en + [None],
        tenses,
        polarities,
    ):
        if not verb["object_required"] and obj is not None:
            # allow object optionally; keep as is
            pass
        if verb["object_required"] and obj is None:
            continue
        # Keep sentences reasonable: avoid too sparse (both place and time missing) sometimes
        if place is None and time_exp is None and rng.random() < 0.5:
            continue

        en = build_en_sentence(subject, verb["en"], obj, place, time_exp, tense, pol)
        hi = build_hi_sentence(subject, verb, obj, place, time_exp, tense, pol)
        combos.append({"src": en, "tgt": hi})

    rng.shuffle(combos)

    # ensure at least target_count; if not enough, cycle
    while len(records) < target_count:
        for rec in combos:
            records.append(rec)
            if len(records) >= target_count:
                break

    return records


def main() -> None:
    output_dir = os.path.join(os.path.dirname(__file__), "data")
    output_path = os.path.join(output_dir, "train_en_hi.jsonl")

    ensure_data_dir(output_dir)
    pairs = generate_pairs(target_count=1200)
    write_jsonl(output_path, pairs)
    print(f"Wrote {len(pairs)} pairs to {output_path}")


if __name__ == "__main__":
    main()


