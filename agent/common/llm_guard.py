from json_repair import repair_json

def sanitize_json_output(model_output: str) -> dict:
  return repair_json(model_output, ensure_ascii=False, return_objects=True)