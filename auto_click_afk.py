"""
Auto Click AFK - Cookie Run Classic
=====================================
สคริปต์นี้ใช้ OpenCV จับคู่รูปภาพ (template matching) เพื่อหาปุ่ม/ไอคอนบนหน้าจอเกม
(เช่นปุ่มรับรางวัล, OK, Battle, Skip) แล้วคลิกให้อัตโนมัติ เหมาะสำหรับฟาร์ม AFK
บน Emulator (BlueStacks / LDPlayer) บน PC ของคุณเอง

วิธีใช้งานคร่าวๆ:
1. ติดตั้งไลบรารีที่ต้องใช้ (ดู README.md)
2. แคปรูปปุ่มที่ต้องการให้คลิก เก็บไว้ในโฟลเดอร์ templates/ (ดูวิธีแคปใน README)
3. แก้ไขค่าตั้งต้นด้านล่าง (CONFIG) ให้ตรงกับพฤติกรรมที่ต้องการ
4. รันสคริปต์ขณะเปิดเกมอยู่บนหน้าจอ (อย่าลดหน้าต่างเกม เพราะสคริปต์จับภาพหน้าจอจริง)
5. กด Ctrl+C ที่หน้าต่าง terminal เพื่อหยุดสคริปต์เมื่อไหร่ก็ได้

หมายเหตุ: สคริปต์นี้ควบคุมเมาส์จริงบนเครื่องคุณ ควรทดสอบด้วย DRY_RUN=True ก่อน
เพื่อดูว่าตรวจจับปุ่มถูกต้องหรือไม่ ก่อนเปิดให้คลิกจริง
"""

import time
import sys
import os
import math
import random
from dataclasses import dataclass, field

import cv2
import numpy as np
import pyautogui

pyautogui.MINIMUM_DURATION = 0
pyautogui.MINIMUM_SLEEP = 0
pyautogui.PAUSE = 0

# ============================== CONFIG ==============================

@dataclass
class Config:
    # โฟลเดอร์เก็บรูป template ของปุ่มที่จะให้สคริปต์ตามหา
    templates_dir: str = "templates"

    # ความมั่นใจขั้นต่ำในการจับคู่รูป (0.0 - 1.0) ยิ่งสูงยิ่งเป๊ะ แต่พลาดง่ายถ้าธีม/ขนาดต่าง
    match_confidence: float = 0.85

    # หน่วงเวลาระหว่างรอบสแกนหน้าจอ (วินาที)
    scan_interval: float = 1.0

    # หน่วงเวลาหลังคลิกแต่ละครั้ง กันคลิกรัวเกินไป (วินาที)
    click_delay: float = 0.5

    # จำกัดพื้นที่สแกนหน้าจอ (left, top, width, height) ใส่ None เพื่อสแกนทั้งจอ
    # ใช้ตอนอยากจำกัดเฉพาะกรอบหน้าต่าง emulator เพื่อความไว/แม่นยำขึ้น
    region: tuple | None = None

    # True = ไม่คลิกจริง แค่บอกว่าเจอปุ่มอะไรตรงไหน (ใช้ทดสอบก่อนใช้งานจริง)
    dry_run: bool = False

    # สุ่มตำแหน่งคลิกรอบจุดกึ่งกลางปุ่ม เป็นสัดส่วนของขนาดปุ่ม (0.25 = สุ่มได้ไม่เกิน 25% ของด้านที่สั้นกว่า)
    # ป้องกันไม่ให้คลิกพิกัดเดิมเป๊ะทุกครั้ง แต่ยังคงอยู่ในขอบเขตปุ่มเสมอ
    click_jitter_ratio: float = 0.25

    # จำนวนขั้นของการขยับเมาส์แบบโค้งก่อนคลิก (สุ่มในช่วงนี้ทุกครั้ง)
    move_steps_range: tuple = (15, 30)

    # ความโค้งของเส้นทางเมาส์ เทียบกับระยะทางตรง (0 = เส้นตรง, ยิ่งมากยิ่งโค้ง)
    move_curviness: float = 0.35

    # รายชื่อไฟล์ template (ไม่ต้องใส่ .png) เรียงตามลำดับความสำคัญ
    # ปุ่มที่อยู่ก่อนในลิสต์จะถูกคลิกก่อนถ้าเจอพร้อมกันหลายปุ่มในรอบเดียวกัน
    priority_order: list = field(default_factory=lambda: [
        "confirm_reconnect",
        "open_all",
        "confirm",
        "ok",
        "play_1",
        "play_2",
    ])


CONFIG = Config()

# ============================== CORE LOGIC ==============================

def load_templates(templates_dir: str) -> dict:
    """โหลดรูปภาพทุกไฟล์ในโฟลเดอร์ templates/ เก็บเป็น dict {ชื่อไฟล์: ndarray}"""
    templates = {}
    if not os.path.isdir(templates_dir):
        print(f"[!] ไม่พบโฟลเดอร์ '{templates_dir}' กรุณาสร้างและใส่รูปปุ่มก่อน")
        sys.exit(1)

    for fname in os.listdir(templates_dir):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(templates_dir, fname)
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                print(f"[!] โหลดรูปไม่ได้: {path}")
                continue
            name = os.path.splitext(fname)[0]
            templates[name] = img

    if not templates:
        print(f"[!] โฟลเดอร์ '{templates_dir}' ว่างเปล่า ใส่รูปปุ่ม (.png) อย่างน้อย 1 รูปก่อนรัน")
        sys.exit(1)

    print(f"[i] โหลด template ทั้งหมด {len(templates)} รูป: {list(templates.keys())}")
    return templates


def screenshot_bgr(region=None) -> np.ndarray:
    """แคปหน้าจอปัจจุบัน คืนค่าเป็นภาพ BGR (สำหรับ OpenCV)"""
    shot = pyautogui.screenshot(region=region)
    frame = cv2.cvtColor(np.array(shot), cv2.COLOR_RGB2BGR)
    return frame


def find_best_match(frame: np.ndarray, template: np.ndarray, threshold: float):
    """หาตำแหน่งที่ template match กับ frame ดีที่สุด คืน (x, y, w, h, score) หรือ None"""
    result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    if max_val >= threshold:
        h, w = template.shape[:2]
        x, y = max_loc
        return (x, y, w, h, max_val)
    return None


def bezier_point(p0, p1, p2, t):
    """จุดบนเส้นโค้ง quadratic Bezier ที่ t (0..1) ระหว่าง p0 -> p2 โดยโค้งผ่าน p1"""
    x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t ** 2 * p2[0]
    y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t ** 2 * p2[1]
    return (x, y)


def move_mouse_naturally(start: tuple, end: tuple, steps_range=(15, 30), curviness=0.35):
    """ขยับเมาส์จาก start ไป end แบบเป็นเส้นโค้ง (ไม่ใช่เส้นตรง) ผ่านหลายจุดย่อย
    เพื่อเลียนแบบการขยับเมาส์ของคนจริงแทนการกระโดดไปจุดหมายทันที"""
    screen_w, screen_h = pyautogui.size()
    # เผื่อขอบไว้ 2px กันชนมุมจอพอดี (มุมจอ 0,0 คือจุดที่ทำให้ fail-safe ของ pyautogui ทำงาน)
    margin = 2

    def clamp(px, py):
        cx = min(max(px, margin), screen_w - 1 - margin)
        cy = min(max(py, margin), screen_h - 1 - margin)
        return (cx, cy)

    start = clamp(*start)
    end = clamp(*end)
    dx, dy = end[0] - start[0], end[1] - start[1]
    dist = math.hypot(dx, dy)

    if dist < 2:
        pyautogui.moveTo(end[0], end[1])
        return

    # จุดกึ่งกลางเส้นตรง แล้วเลื่อนออกด้านข้าง (แนวตั้งฉาก) แบบสุ่ม เพื่อสร้างจุดควบคุมของเส้นโค้ง
    mid = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
    perp = (-dy, dx)
    perp_len = math.hypot(*perp) or 1
    perp_norm = (perp[0] / perp_len, perp[1] / perp_len)
    offset_mag = dist * curviness * random.uniform(0.3, 1.0) * random.choice([-1, 1])
    control = clamp(mid[0] + perp_norm[0] * offset_mag, mid[1] + perp_norm[1] * offset_mag)

    steps = random.randint(*steps_range)
    for i in range(1, steps + 1):
        t = i / steps
        # easing เล็กน้อยให้ช่วงต้น/ท้ายช้ากว่าตรงกลาง เหมือนการเคลื่อนไหวของคนจริง
        eased_t = t * t * (3 - 2 * t)
        px, py = bezier_point(start, control, end, eased_t)
        px, py = clamp(px, py)
        pyautogui.moveTo(px, py)
        time.sleep(random.uniform(0.004, 0.015))


def click_at(x: int, y: int, w: int = 0, h: int = 0, region_offset=(0, 0),
             dry_run=True, jitter_ratio=0.25, move_steps_range=(15, 30), move_curviness=0.35):
    # สุ่มตำแหน่งคลิกรอบจุดกึ่งกลางปุ่ม แต่ไม่เกินขอบเขตปุ่ม
    max_jitter = int(min(w, h) * jitter_ratio) if (w and h) else 0
    jx = random.randint(-max_jitter, max_jitter) if max_jitter > 0 else 0
    jy = random.randint(-max_jitter, max_jitter) if max_jitter > 0 else 0

    abs_x = x + jx + region_offset[0]
    abs_y = y + jy + region_offset[1]

    # เผื่อไว้กันพิกัดหลุดขอบจอ (เช่นถ้า region offset คำนวณผิดพลาด) ซึ่งจะไปชน fail-safe ของ pyautogui ที่มุมจอ
    screen_w, screen_h = pyautogui.size()
    abs_x = min(max(abs_x, 1), screen_w - 2)
    abs_y = min(max(abs_y, 1), screen_h - 2)

    if dry_run:
        print(f"    -> [DRY RUN] จะขยับเมาส์แบบโค้งแล้วคลิกที่ ({abs_x}, {abs_y})")
    else:
        start = pyautogui.position()
        move_mouse_naturally((start.x, start.y), (abs_x, abs_y),
                              steps_range=move_steps_range, curviness=move_curviness)
        time.sleep(random.uniform(0.05, 0.15))  # หน่วงสั้นๆ ก่อนกดปุ่มเมาส์ เหมือนคนจริง
        pyautogui.click()
        print(f"    -> คลิกที่ ({abs_x}, {abs_y})")


def run(config: Config):
    templates = load_templates(config.templates_dir)
    ordered_names = [n for n in config.priority_order if n in templates]
    ordered_names += [n for n in templates if n not in ordered_names]  # เพิ่มตัวที่ไม่ได้ระบุลำดับไว้ท้ายสุด

    region_offset = (config.region[0], config.region[1]) if config.region else (0, 0)

    print("=" * 60)
    print("Cookie Run Classic - Auto Click AFK")
    print(f"โหมด: {'DRY RUN (ไม่คลิกจริง)' if config.dry_run else 'LIVE (คลิกจริง!)'}")
    print("กด Ctrl+C เพื่อหยุด")
    print("=" * 60)

    if not config.dry_run:
        print("[!] เริ่มคลิกจริงใน 5 วินาที... สลับไปที่หน้าต่างเกมตอนนี้!")
        time.sleep(5)

    try:
        while True:
            frame = screenshot_bgr(config.region)
            clicked_this_round = False

            for name in ordered_names:
                match = find_best_match(frame, templates[name], config.match_confidence)
                if match:
                    x, y, w, h, score = match
                    center_x, center_y = x + w // 2, y + h // 2
                    print(f"[{time.strftime('%H:%M:%S')}] เจอ '{name}' (score={score:.2f}) ที่ ({center_x},{center_y})")
                    click_at(
                        center_x, center_y, w, h,
                        region_offset=region_offset,
                        dry_run=config.dry_run,
                        jitter_ratio=config.click_jitter_ratio,
                        move_steps_range=config.move_steps_range,
                        move_curviness=config.move_curviness,
                    )
                    clicked_this_round = True
                    time.sleep(config.click_delay)
                    break  # คลิกทีละปุ่มต่อรอบ แล้ววนสแกนใหม่ (กันคลิกซ้อนผิดจังหวะ)

            if not clicked_this_round:
                pass  # ไม่เจอปุ่มไหนเลยในรอบนี้ รอสแกนรอบถัดไป

            time.sleep(config.scan_interval)

    except KeyboardInterrupt:
        print("\n[i] หยุดสคริปต์แล้ว")


if __name__ == "__main__":
    run(CONFIG)
