"""OCR helpers tuned for Chinese automotive-finance documents."""

import re
from typing import Any, Iterable

import cv2
import numpy as np


FINANCE_REPLACEMENTS = {
    "车架号 ": "车架号：",
    "VIN码 ": "VIN码：",
    "贷款金额 ": "贷款金额：",
    "首付金额 ": "首付金额：",
    "合同编号 ": "合同编号：",
    "发动机号 ": "发动机号：",
}


def get_ocr(use_cuda: bool | None = None) -> "RapidOCR":
    """Create an OCR engine, preferring Paddle on GPU and ONNX on CPU."""
    if use_cuda is None:
        try:
            import torch

            use_cuda = torch.cuda.is_available()
        except ImportError:
            use_cuda = False
    try:
        from rapidocr_paddle import RapidOCR

        return RapidOCR(
            det_use_cuda=use_cuda,
            cls_use_cuda=use_cuda,
            rec_use_cuda=use_cuda,
        )
    except (ImportError, RuntimeError):
        from rapidocr_onnxruntime import RapidOCR

        return RapidOCR()


def preprocess_finance_image(image: Any) -> np.ndarray:
    """Improve scans of contracts, ID/vehicle documents and repayment tables."""
    if isinstance(image, str):
        image = cv2.imdecode(np.fromfile(image, dtype=np.uint8), cv2.IMREAD_COLOR)
    else:
        image = np.asarray(image)
    if image is None or image.size == 0:
        raise ValueError("OCR image is empty or unreadable")
    if image.ndim == 2:
        gray = image
    else:
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Upscaling materially helps small interest rates, dates, IDs and VIN characters.
    short_side = min(gray.shape[:2])
    if short_side < 1600:
        scale = min(2.5, 1600 / max(short_side, 1))
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.fastNlMeansDenoising(gray, None, 8, 7, 21)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def normalize_finance_text(text: str) -> str:
    """Normalize common OCR variants without changing financially meaningful digits."""
    text = text.replace("％", "%").replace("￥", "¥").replace("：", "：")
    text = re.sub(r"(?<=\d)\s+(?=\d)", "", text)
    text = re.sub(r"(\d)\s*[,.]\s*(\d{2})(?!\d)", r"\1.\2", text)
    text = re.sub(r"\b[Vv][Ii][Nn]\b", "VIN", text)
    for source, target in FINANCE_REPLACEMENTS.items():
        text = text.replace(source, target)
    return re.sub(r"[ \t]+", " ", text).strip()


def extract_finance_fields(text: str) -> dict[str, str]:
    """Extract high-value fields for retrieval; every value still requires review."""
    patterns = {
        '合同编号': r'(?:合同编号|合同号)\s*[：:]?\s*([A-Za-z0-9_-]{5,40})',
        'VIN': r'(?:VIN(?:码)?|车架号)\s*[：:]?\s*([A-HJ-NPR-Z0-9]{17})',
        '贷款金额': r'(?:贷款金额|贷款本金)\s*[：:]?\s*[¥￥]?\s*([0-9,.]+\s*元?)',
        '首付金额': r'(?:首付金额|首付款)\s*[：:]?\s*[¥￥]?\s*([0-9,.]+\s*元?)',
        '年化利率': r'(?:年化利率|年利率)\s*[：:]?\s*([0-9.]+\s*%)',
        '贷款期限': r'(?:贷款期限|融资期限)\s*[：:]?\s*([0-9]+\s*(?:期|个月|月|年))',
        '月供': r'(?:月供|每期还款额)\s*[：:]?\s*[¥￥]?\s*([0-9,.]+\s*元?)',
    }
    fields = {}
    for name, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            fields[name] = match.group(1).strip()
    return fields


def _line_key(line: list) -> tuple[float, float]:
    box = np.asarray(line[0], dtype=float)
    return float(box[:, 1].mean()), float(box[:, 0].min())


def format_ocr_result(result: Iterable[list] | None, min_confidence: float = 0.55) -> str:
    """Sort OCR boxes into reading order and retain confidence-worthy text."""
    if not result:
        return ""
    accepted = []
    for line in result:
        if len(line) < 2:
            continue
        confidence = float(line[2]) if len(line) > 2 else 1.0
        value = normalize_finance_text(str(line[1]))
        if value and confidence >= min_confidence:
            accepted.append(line)
    accepted.sort(key=_line_key)

    rows: list[list[list]] = []
    for line in accepted:
        y, _ = _line_key(line)
        height = max(1.0, np.ptp(np.asarray(line[0], dtype=float)[:, 1]))
        if not rows or abs(y - _line_key(rows[-1][0])[0]) > height * 0.65:
            rows.append([line])
        else:
            rows[-1].append(line)
    output = []
    for row in rows:
        row.sort(key=lambda item: _line_key(item)[1])
        output.append(" | ".join(normalize_finance_text(str(item[1])) for item in row))
    return "\n".join(output)


def recognize_finance_image(ocr: Any, image: Any, min_confidence: float = 0.55) -> str:
    processed = preprocess_finance_image(image)
    result, _ = ocr(processed)
    text = format_ocr_result(result, min_confidence=min_confidence)
    fields = extract_finance_fields(text)
    if fields:
        summary = '\n'.join(f'- {name}: {value}' for name, value in fields.items())
        text += f'\n\n[结构化字段（OCR结果，必须与原件复核）]\n{summary}'
    return text
