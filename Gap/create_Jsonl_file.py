from typing import Any
from llm import prompts,invoke_llm,clean_json
import time
import json

FILES_PATH = "files/"
FILE_NAME = "maanim_desc"# "maanim"

def _safe_get_str(v: Any) -> str:
    """המרה בטוחה למחרוזת וניקוי רווחים/שורות."""
    if v is None:
        return ""
    s = str(v).strip()
    # הגנה מפני ערכים כמו "NULL"/"None" שהגיעו כמחרוזת
    return "" if s.lower() in {"null", "none"} else s

def summarize_maneh(text: str) -> str:
    """
    פנייה ל-LLM כדי להחזיר תמצות קצר (כמה מילים) למהות המענה.
    מצפה שה-LLM יחזיר JSON עם שדה "summary" (או "תמצות").
    נופל חזרה לטקסט הגולמי במקרה של כשל בפרסינג.
    """
    if not text:
        return ""

    if "brief_maneh_summary" not in prompts:
        raise KeyError(f"Prompts missing 'brief_maneh_summary'")

    prompt_tmpl = prompts["brief_maneh_summary"]
    full_prompt = prompt_tmpl.format(text=text)
    raw_response = invoke_llm(full_prompt)
  
    # ניסיון לפרסינג JSON
    try:
        cleaned = clean_json(raw_response)
        data = json.loads(cleaned)
        summary = data.get("summary") or data.get("תמצות") or ""
        return _safe_get_str(summary)
    except Exception:
        # נפילה חזרה: ניקוי טקסט ישיר
        return _safe_get_str(raw_response)

def create_jsonl_with_summaries(sleep_between_calls: float = 0.1,
                                separator: str = ". פירוט המענה "):
    """
    קורא {files_path}{file_name}.json (מערך של אובייקטים),
    עבור כל פריט:
      - מייצר תמצות קצרה מ"תקציר_המענה" (אם קיים),
      - משרשר את התמצות לסוף "תאור",
      - כותב לשורת JSONL המכילה רק {"תאור": ...}
    שומר ל-{files_path}{file_name}.jsonl
    """

     # נתיב קובץ המקור JSON
    input_file = f"{FILES_PATH}{FILE_NAME}.json" 
    # נתיב קובץ היעד JSONL
    output_file = f"{FILES_PATH}{FILE_NAME}.jsonl"


    # קריאה
    with open(input_file, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)
    # כתיבה
    with open(output_file, "w", encoding="utf-8") as out:
        for idx, item in enumerate(data, start=1):
            desc = _safe_get_str(item.get("תאור"))
            brief_src = _safe_get_str(item.get("תקציר_המענה"))
            summary = ""
            if brief_src and brief_src.strip().lower() != "no description":
                summary = summarize_maneh(brief_src)

            # מיזוג: "תאור" המקורי + התמצות (אם קיימת)
            if summary:
                merged = f"{desc}{separator}{summary}" if desc else summary
            else:
                merged = desc

            json_line = json.dumps({"תאור": merged}, ensure_ascii=False)
            out.write(json_line + "\n")

            # האטה קלה כדי לא להציף את ה-LLM (ניתן לאפס ל-0)
            if sleep_between_calls and brief_src:
                time.sleep(sleep_between_calls)

    print(f"נוצר בהצלחה: {output_file}")
    return output_file  


def create_jsonl_file():
    # נתיב קובץ המקור JSON
    input_file = f"{FILES_PATH}{FILE_NAME}.json" 
    # נתיב קובץ היעד JSONL
    output_file = f"{FILES_PATH}{FILE_NAME}.jsonl"

    # קורא את הקובץ JSON
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # יוצר את קובץ JSONL
    with open(output_file, "w", encoding="utf-8") as f:
        for item in data:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + "\n")

    print(f"נוצר בהצלחה: {output_file}")
    return output_file

if __name__ == "__main__":
    create_jsonl_with_summaries()