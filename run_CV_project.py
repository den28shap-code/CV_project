from dataclasses import asdict

from MVP_CV_project import OCRPipeline, SQLiteStore


def main():
    image_path = "images/IMG_5870.JPG"

    pipeline = OCRPipeline(
        languages=["en", "ru"],
        use_gpu=False,
    )

    store = SQLiteStore("recognition.db")

    result = pipeline.run(
        image_path=image_path,
        mode="display_value",   # или "all_text"
        crop=None,
        min_confidence=0.15,
        save_debug=True,
        debug_dir="debug",
    )

    row_id = store.save(result)

    print("\n=== RESULT ===")
    print("db_row_id       :", row_id)
    print("source_path     :", result.source_path)
    print("mode            :", result.mode)
    print("raw_text        :", result.raw_text)
    print("normalized_text :", result.normalized_text)
    print("confidence      :", result.confidence)
    print("saved_at        :", result.saved_at)
    print("meta            :", result.meta)

    print("\nCandidates:")
    for cand in result.candidates[:10]:
        print("  -", asdict(cand))


if __name__ == "__main__":
    main()